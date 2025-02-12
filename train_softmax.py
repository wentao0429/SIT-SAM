# set up environment
import torch

import numpy as np
import random
import datetime
import logging
import matplotlib.pyplot as plt
import os
import time

from sympy.abc import alpha

join = os.path.join
from tqdm import tqdm
from torch.backends import cudnn
import torch.distributed as dist
import torch.nn.functional as F
import torchio as tio
from torch.utils.data.distributed import DistributedSampler
from segment_anything.build_sam3D_softmax import sam_model_registry3D
import argparse
from torch.cuda import amp
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from monai.losses import DiceCELoss, DiceLoss, DiceFocalLoss
from contextlib import nullcontext
from utils.click_method import get_next_click3D_torch_2
from utils.data_loader_cls import Dataset_Union_ALL, Union_Dataloader, Dataset_Union_GROUP
from utils.data_paths import img_datas
from utils.data_paths import all_classes, all_classes_merged

import wandb

# %% set up parser
parser = argparse.ArgumentParser()
parser.add_argument('--task_name', type=str, default='union_train')
parser.add_argument('--click_type', type=str, default='random')
parser.add_argument('--multi_click', action='store_true', default=False)
parser.add_argument('--model_type', type=str, default='vit_b_ori')
parser.add_argument('--checkpoint', type=str, default='./work_dir/SAM/sam_vit_b.pth')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--work_dir', type=str, default='./work_dir')
parser.add_argument('--group_start', type=int, default=40)

# train
parser.add_argument('--num_workers', type=int, default=32)
parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0, 1, 2, 3])
parser.add_argument('--multi_gpu', action='store_true', default=False)
parser.add_argument('--resume', action='store_true', default=False)

# lr_scheduler
parser.add_argument('--lr_scheduler', type=str, default='multisteplr')
parser.add_argument('--step_size', type=list, default=[1000, 1500])
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--num_epochs', type=int, default=200)
parser.add_argument('--img_size', type=int, default=128)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--accumulation_steps', type=int, default=8)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=0.1)
parser.add_argument('--port', type=int, default=12312)

args = parser.parse_args()

device = args.device
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in args.gpu_ids])
logger = logging.getLogger(__name__)
LOG_OUT_DIR = join(args.work_dir, args.task_name)
click_methods = {
    'random': get_next_click3D_torch_2,
}
MODEL_SAVE_PATH = join(args.work_dir, args.task_name)
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
handler = logging.FileHandler(os.path.join(LOG_OUT_DIR, 'track.log'))
handler.setLevel(logging.INFO)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.info('Start training...')


def build_model(args):
    sam_model = sam_model_registry3D[args.model_type](checkpoint=None).to(device)
    if args.multi_gpu:
        sam_model = DDP(sam_model, device_ids=[args.rank], output_device=args.rank)
    return sam_model


def get_dataloaders(args):
    ## 分组要改
    train_dataset = Dataset_Union_GROUP(
        paths=img_datas[0],
        all_classes=all_classes,
        group_start=args.group_start,
        group_end=args.group_start + 19,
        transform=tio.Compose([
            tio.ToCanonical(),
            tio.Resample(target=(1.5, 1.5, 1.5)),
            tio.CropOrPad(mask_name='label', target_shape=(args.img_size, args.img_size, args.img_size)),
            # crop only object region
            # tio.RandomFlip(axes=(0, 1, 2)),
        ]),
        threshold=5000)

    if args.multi_gpu:
        train_sampler = DistributedSampler(train_dataset)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    train_dataloader = Union_Dataloader(
        dataset=train_dataset,
        sampler=train_sampler,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    return train_dataloader


class BaseTrainer:
    def __init__(self, model, dataloaders, args, writer=None):

        self.model = model
        self.dataloaders = dataloaders
        self.args = args
        self.best_loss = np.inf
        self.best_dice = 0.0
        self.step_best_loss = np.inf
        self.step_best_dice = 0.0
        self.losses = []
        self.dices = []
        self.ious = []
        self.set_loss_fn()
        self.set_optimizer()
        self.set_lr_scheduler()
        # self.init_checkpoint(join(self.args.work_dir, self.args.task_name, 'sam_model_latest.pth'))
        self.init_checkpoint(args.checkpoint)
        for name, param in model.named_parameters():
            if "mask_decoder" not in name:  
                param.requires_grad = True  

        self.norm_transform = tio.ZNormalization(masking_method=lambda x: x > 0)
        self.writer = writer  # 保存writer

    def set_loss_fn(self):
        weights = torch.tensor([0.1] + [1.0] * 117).to(device)  # Assign a lower weight to the background class

        
        self.seg_loss = DiceFocalLoss(softmax=True, to_onehot_y=True, squared_pred=True, include_background=True)

    def set_optimizer(self):
        if self.args.multi_gpu:
            sam_model = self.model.module
        else:
            sam_model = self.model

        self.optimizer = torch.optim.AdamW([
            # {'params': sam_model.image_encoder.parameters()},  # , 'lr': self.args.lr * 0.1},
            # {'params': sam_model.prompt_encoder.parameters()},  # 'lr': self.args.lr * 0.1},
            {'params': sam_model.mask_decoder.parameters()},  # 'lr': self.args.lr * 0.1},
        ], lr=self.args.lr, betas=(0.9, 0.999), weight_decay=self.args.weight_decay)

    def set_lr_scheduler(self):
        if self.args.lr_scheduler == "multisteplr":
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                                     self.args.step_size,
                                                                     self.args.gamma)
        elif self.args.lr_scheduler == "steplr":
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                                self.args.step_size[0],
                                                                self.args.gamma)
        elif self.args.lr_scheduler == 'coswarm':
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=15,
                                                                                     eta_min=1e-6)
        else:
            self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, 0.1)

    def init_checkpoint(self, ckp_path):
        last_ckpt = None
        if os.path.exists(ckp_path):
            if self.args.multi_gpu:
                dist.barrier()
                last_ckpt = torch.load(ckp_path, map_location=self.args.device)
            else:
                last_ckpt = torch.load(ckp_path, map_location=self.args.device)

        if last_ckpt:
            if self.args.multi_gpu:
                self.model.module.load_state_dict(last_ckpt['model_state_dict'], strict=False)
            else:
                self.model.load_state_dict(last_ckpt['model_state_dict'], strict=False)
            if not self.args.resume:
                self.start_epoch = 0
            else:
                self.start_epoch = last_ckpt['epoch']
                self.optimizer.load_state_dict(last_ckpt['optimizer_state_dict'])
                self.lr_scheduler.load_state_dict(last_ckpt['lr_scheduler_state_dict'])
                self.losses = last_ckpt['losses']
                self.dices = last_ckpt['dices']
                self.best_loss = last_ckpt['best_loss']
                self.best_dice = last_ckpt['best_dice']
            print(f"Loaded checkpoint from {ckp_path} (epoch {self.start_epoch})")
        else:
            self.start_epoch = 0
            print(f"No checkpoint found at {ckp_path}, start training from scratch")

    def save_checkpoint(self, epoch, state_dict, describe="last"):
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": state_dict,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
            "losses": self.losses,
            "dices": self.dices,
            "best_loss": self.best_loss,
            "best_dice": self.best_dice,
            "args": self.args,
            "used_datas": img_datas,
        }, join(MODEL_SAVE_PATH, f"sam_model_{describe}.pth"))

    def batch_forward(self, sam_model, image_embedding, gt3D, low_res_masks, points=None):
        # print(low_res_masks.shape)
        with torch.no_grad():
            sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                points=points,
                boxes=None,
                masks=low_res_masks,
            )
        low_res_masks, iou_predictions = sam_model.mask_decoder(
            image_embeddings=image_embedding.to(device),  # (B, 256, 64, 64)
            image_pe=sam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )
        # print(low_res_masks.shape, low_res_masks[0][0][16][16][16], low_res_masks[0][12][16][16][16]) # torch.Size([4, 117, 32, 32, 32]) tensor(-0.3) tensor(0.3)
        # start = time.time()
        prev_masks = F.interpolate(low_res_masks, size=gt3D.shape[-3:], mode='trilinear', align_corners=False)
        # print(f'Interpolate time: {time.time() - start}')
        return low_res_masks, prev_masks

    def get_points(self, prev_masks, gt3D, class_id, num_click):
        # if num_click != 0:
        #     prev_masks = torch.argmax(torch.softmax(prev_masks, 1), 1)
        #     for i in range(prev_masks.shape[0]):
        #         prev_masks[i] = (prev_masks[i] == class_id[i]).float()
        # print(prev_masks.shape, gt3D.shape)
        batch_points, batch_labels = click_methods[self.args.click_type](prev_masks, gt3D)

        points_co = torch.cat(batch_points, dim=0).to(device)
        points_la = torch.cat(batch_labels, dim=0).to(device)

        self.click_points.append(points_co)
        self.click_labels.append(points_la)

        points_multi = torch.cat(self.click_points, dim=1).to(device)
        labels_multi = torch.cat(self.click_labels, dim=1).to(device)

        if self.args.multi_click:
            points_input = points_multi
            labels_input = labels_multi
        else:
            points_input = points_co
            labels_input = points_la
        return points_input, labels_input

    def interaction(self, sam_model, image_embedding, gt3D, gt3D_id, num_clicks, class_id):
        return_loss = 0
        prev_masks = torch.zeros_like(gt3D).to(gt3D.device)
        low_res_masks = F.interpolate(prev_masks.float(),
                                      size=(args.img_size // 4, args.img_size // 4, args.img_size // 4))
        random_insert = np.random.randint(2, 11)
        prev_masks_all = None  # 初始化 prev_masks_all

        for num_click in range(num_clicks):
            points_input, labels_input = self.get_points(prev_masks, gt3D, class_id, num_click)

            # if num_click == random_insert or num_click == num_clicks - 1:
            if num_click == random_insert:
                low_res_masks, prev_masks = self.batch_forward(sam_model, image_embedding, gt3D, low_res_masks,
                                                               points=None)
            else:
                low_res_masks, prev_masks = self.batch_forward(sam_model, image_embedding, gt3D, low_res_masks,
                                                               points=[points_input, labels_input])

            # 这里对low_res_masks的处理是为了得到下一次点击时候(batch_forward)的prompt encoder的mask
            output_masks = torch.zeros(
                (low_res_masks.shape[0], 1, low_res_masks.shape[2], low_res_masks.shape[3],
                 low_res_masks.shape[4]),
                dtype=low_res_masks.dtype).to(low_res_masks.device)
            for i in range(low_res_masks.shape[0]):
                output_masks[i, 0, :, :] = low_res_masks[i, class_id[i], :, :]
            low_res_masks = output_masks
            low_res_masks = torch.sigmoid(low_res_masks)
            # low_res_masks = torch.softmax(low_res_masks, dim=1)
            # print(prev_masks.shape, gt3D_id.shape, low_res_masks.shape)
            loss = self.seg_loss(prev_masks, gt3D_id)

            return_loss += loss

            # 这样处理后会导致变成二值的, 即值为0和1
            prev_masks = torch.argmax(torch.softmax(prev_masks, 1), 1)
            prev_masks_all = prev_masks.clone()
            # print(torch.unique(prev_masks_all), torch.unique(class_id))
            for i in range(prev_masks.shape[0]):
                # print(class_id[i])
                prev_masks[i] = (prev_masks[i] == class_id[i]).float()
                # print(prev_masks[i].shape, gt3D[i].shape)
        return prev_masks, return_loss, prev_masks_all

    def get_dice_score(self, prev_masks, gt3D):
        def compute_dice(mask_pred, mask_gt):
            mask_threshold = 0.5

            mask_pred = (mask_pred > mask_threshold)
            mask_gt = (mask_gt > 0)

            volume_sum = mask_gt.sum() + mask_pred.sum()
            if volume_sum == 0:
                return np.NaN
            volume_intersect = (mask_gt & mask_pred).sum()
            return 2 * volume_intersect / volume_sum

        pred_masks = (prev_masks > 0.5)
        true_masks = (gt3D > 0)
        dice_list = []
        for i in range(true_masks.shape[0]):
            dice_list.append(compute_dice(pred_masks[i], true_masks[i]))
        return (sum(dice_list) / len(dice_list)).item()

    def train_epoch(self, epoch, num_clicks):
        epoch_loss = 0
        epoch_iou = 0  
        epoch_dice = 0
        self.model.train()
        if self.args.multi_gpu:
            sam_model = self.model.module
        else:
            sam_model = self.model
            self.args.rank = -1  

        if not self.args.multi_gpu or (self.args.multi_gpu and self.args.rank == 0):
            tbar = tqdm(self.dataloaders)
        else:
            tbar = self.dataloaders

        self.optimizer.zero_grad()
        step_loss = 0
        for step, (image3D, gt3D, cls_classes_id) in enumerate(tbar):
            cls_classes_id = cls_classes_id - args.group_start

            my_context = self.model.no_sync if self.args.rank != -1 and step % self.args.accumulation_steps != 0 else nullcontext

            with my_context():

                image3D = self.norm_transform(image3D.squeeze(dim=1))  # (N, C, W, H, D)
                image3D = image3D.unsqueeze(dim=1)

                image3D = image3D.to(device)
                gt3D = gt3D.to(device).type(torch.long)
                gt3D_id = gt3D * (cls_classes_id.to(device) + 1).unsqueeze(dim=-1).unsqueeze(dim=-1).unsqueeze(
                    dim=-1).unsqueeze(dim=-1).type(torch.long)
                with amp.autocast():
                    with torch.no_grad():
                        image_embedding = sam_model.image_encoder(image3D)

                    self.click_points = []
                    self.click_labels = []

                    pred_list = []

                    # print(gt3D_id.shape, cls_classes_id.shape) # torch.Size([4, 1, 128, 128, 128]) torch.Size([4])
                    # print(torch.unique(cls_classes_id), torch.unique(gt3D_id))

                    prev_masks, loss, prev_masks_all = self.interaction(sam_model, image_embedding, gt3D, gt3D_id, 5,
                                                                        cls_classes_id.to(device) + 1)

                epoch_loss += loss.item()

                cur_loss = loss.item()

                loss /= self.args.accumulation_steps  

                self.scaler.scale(loss).backward()

            if step % self.args.accumulation_steps == 0 and step != 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                print_loss = step_loss / self.args.accumulation_steps
                step_loss = 0
                # print(torch.unique(prev_masks), torch.unique(gt3D), torch.sum(prev_masks), torch.sum(gt3D))
                print_dice = self.get_dice_score(prev_masks, gt3D)
                epoch_dice += print_dice
            else:
                step_loss += cur_loss

            if not self.args.multi_gpu or (self.args.multi_gpu and self.args.rank == 0):
                if step % self.args.accumulation_steps == 0 and step != 0:
                    # print(f'Epoch: {epoch}, Step: {step}, Loss: {print_loss}')
                    tbar.set_description(f'Loss: {print_loss:.4f}, '
                                         f'Dice: {print_dice:.4f}, '
                                         f'Unique_all: {torch.unique(prev_masks_all)}'
                                         f'Unique_gt: {torch.unique(gt3D_id)}'
                                         f'Unique_cls: {torch.unique(cls_classes_id)}')

        epoch_loss /= step
        epoch_dice = epoch_dice * self.args.accumulation_steps / step  

        return epoch_loss, epoch_iou, epoch_dice, pred_list

    def eval_epoch(self, epoch, num_clicks):
        return 0

    def plot_result(self, plot_data, description, save_name):
        plt.plot(plot_data)
        plt.title(description)
        plt.xlabel('Epoch')
        plt.ylabel(f'{save_name}')
        plt.savefig(join(MODEL_SAVE_PATH, f'{save_name}.png'))
        plt.close()

    def train(self):
        self.scaler = amp.GradScaler()
        for epoch in range(self.start_epoch, self.args.num_epochs):
            print(f'Epoch: {epoch}/{self.args.num_epochs - 1}')

            if self.args.multi_gpu:
                dist.barrier()
                self.dataloaders.sampler.set_epoch(epoch)
            num_clicks = np.random.randint(1, 21)
            epoch_loss, epoch_iou, epoch_dice, pred_list = self.train_epoch(epoch, num_clicks)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            if self.args.multi_gpu:
                dist.barrier()

            if not self.args.multi_gpu or (self.args.multi_gpu and self.args.rank == 0):
                self.losses.append(epoch_loss)
                self.dices.append(epoch_dice)
                print(f'EPOCH: {epoch}, Loss: {epoch_loss}')
                print(f'EPOCH: {epoch}, Dice: {epoch_dice}')
                logger.info(f'Epoch\t {epoch}\t : loss: {epoch_loss}, dice: {epoch_dice}')
                # wandb.log({"loss": epoch_loss, "dice": epoch_dice})

                if self.args.multi_gpu:
                    state_dict = self.model.module.state_dict()
                else:
                    state_dict = self.model.state_dict()

                # save latest checkpoint
                self.save_checkpoint(
                    epoch,
                    state_dict,
                    describe='latest'
                )

                # save train loss best checkpoint
                if epoch_loss < self.best_loss:
                    self.best_loss = epoch_loss
                    self.save_checkpoint(
                        epoch,
                        state_dict,
                        describe='loss_best'
                    )

                # save train dice best checkpoint
                if epoch_dice > self.best_dice:
                    self.best_dice = epoch_dice
                    self.save_checkpoint(
                        epoch,
                        state_dict,
                        describe='dice_best'
                    )

                self.plot_result(self.losses, 'Dice + Cross Entropy Loss', 'Loss')
                # self.writer.close()
                # self.plot_result(self.dices, 'Dice', 'Dice')
        logger.info('=====================================================================')
        logger.info(f'Best loss: {self.best_loss}')
        logger.info(f'Best dice: {self.best_dice}')
        logger.info(f'Total loss: {self.losses}')
        logger.info(f'Total dice: {self.dices}')
        logger.info('=====================================================================')
        logger.info(f'args : {self.args}')
        logger.info(f'Used datasets : {img_datas}')
        logger.info('=====================================================================')


def init_seeds(seed=0, cuda_deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True


def device_config(args):
    try:
        if not args.multi_gpu:
            # Single GPU
            if args.device == 'mps':
                args.device = torch.device('mps')
            else:
                args.device = torch.device(f"cuda:{args.gpu_ids[0]}")
        else:
            args.nodes = 1
            args.ngpus_per_node = len(args.gpu_ids)
            args.world_size = args.nodes * args.ngpus_per_node

    except RuntimeError as e:
        print(e)


def main():
    mp.set_sharing_strategy('file_system')
    device_config(args)
    if args.multi_gpu:
        mp.spawn(
            main_worker,
            nprocs=args.world_size,
            args=(args,)
        )
    else:
        random.seed(2023)
        np.random.seed(2023)
        torch.manual_seed(2023)
        
        dataloaders = get_dataloaders(args)
        model = build_model(args)
        # Create trainer
        trainer = BaseTrainer(model, dataloaders, args)
        # Train
        trainer.train()


def main_worker(rank, args):
    setup(rank, args.world_size)

    torch.cuda.set_device(rank)
    args.num_workers = int(args.num_workers / args.ngpus_per_node)
    args.device = torch.device(f"cuda:{rank}")
    args.rank = rank

    init_seeds(2023 + rank)

    cur_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO if rank in [-1, 0] else logging.WARN,
        filemode='w',
        filename=os.path.join(LOG_OUT_DIR, f'output_{cur_time}.log'))


    dataloaders = get_dataloaders(args)
    model = build_model(args)
    trainer = BaseTrainer(model, dataloaders, args)
    trainer.train()
    cleanup()


def setup(rank, world_size):
    # initialize the process group
    dist.init_process_group(
        backend='nccl',
        init_method=f'tcp://127.0.0.1:{args.port}',
        world_size=world_size,
        rank=rank
    )


def cleanup():
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
