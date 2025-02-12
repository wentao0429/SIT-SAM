# set up environment
import datetime
import logging
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb

join = os.path.join
from tqdm import tqdm
from torch.backends import cudnn
import torch.distributed as dist
import torch.nn.functional as F
import torchio as tio
from torch.utils.data.distributed import DistributedSampler
from segment_anything.build_sam3D import sam_model_registry3D
from segment_anything.build_cls_knn import classifier
import argparse
from torch.cuda import amp
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from contextlib import nullcontext
from utils.click_method import get_next_click3D_torch_2
from utils.data_loader_cls import Dataset_Union_ALL, Union_Dataloader
from utils.data_paths import img_datas, img_val_datas
from utils.data_paths import all_classes_merged, all_classes

# %% set up parser
parser = argparse.ArgumentParser()
parser.add_argument('--task_name', type=str, default='union_train')
parser.add_argument('--click_type', type=str, default='random')
parser.add_argument('--multi_click', action='store_true', default=False)
parser.add_argument('--model_type', type=str, default='vit_b_ori')
parser.add_argument('--checkpoint', type=str, default='./work_dir/SAM/sam_vit_b.pth')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--work_dir', type=str, default='./work_dir')

# train
parser.add_argument('--num_workers', type=int, default=64)
parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0, 1])
parser.add_argument('--multi_gpu', action='store_true', default=False)
parser.add_argument('--resume', action='store_true', default=False)
parser.add_argument('--KNN', action='store_true', default=False)

# lr_scheduler
parser.add_argument('--lr_scheduler', type=str, default='multisteplr')
parser.add_argument('--step_size', type=list, default=[1200, 3000])
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--num_epochs', type=int, default=200)
parser.add_argument('--img_size', type=int, default=128)
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--accumulation_steps', type=int, default=20)
parser.add_argument('--lr', type=float, default=8e-4)
parser.add_argument('--weight_decay', type=float, default=0.1)
parser.add_argument('--port', type=int, default=12361)
parser.add_argument('--num_classes', type=int, default=117)

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
    # print(img_datas[0])
    train_dataset = Dataset_Union_ALL(
        paths=img_datas[0],
        all_classes=all_classes,
        transform=tio.Compose([
            tio.ToCanonical(),
            tio.Resample(target=(1.5, 1.5, 1.5)),
            tio.CropOrPad(mask_name='label', target_shape=(args.img_size, args.img_size, args.img_size)),
        ]),
        threshold=30)  

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
        drop_last=True,
        persistent_workers=True
    )
    return train_dataloader


def get_val_dataloader(args):
    val_dataset = Dataset_Union_ALL(
        paths=img_val_datas[0],
        all_classes=all_classes,
        transform=tio.Compose([
            tio.ToCanonical(),
            tio.Resample(target=(1.5, 1.5, 1.5)),
            tio.CropOrPad(mask_name='label', target_shape=(args.img_size, args.img_size, args.img_size)),
        ]),
        threshold=0)

    if args.multi_gpu:
        val_sampler = DistributedSampler(val_dataset)
        shuffle = False
    else:
        val_sampler = None
        shuffle = True

    val_dataloader = Union_Dataloader(
        dataset=val_dataset,
        sampler=val_sampler,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True
    )
    return val_dataloader


class BaseTrainer:
    def __init__(self, model, dataloaders, classifier, args, writer=None):

        self.model = model
        self.dataloaders = dataloaders
        self.val_dataloader = get_val_dataloader(args)
        self.classifier = classifier(input_dim=528, num_classes=args.num_classes, mlp_ratio=2.0).to(device)
        if args.multi_gpu:
            self.classifier = DDP(self.classifier, device_ids=[args.rank], output_device=args.rank)
        self.args = args
        self.best_loss = np.inf
        self.best_cls_loss = np.inf
        self.best_dice = 0.0
        self.best_acc = 0.0
        self.step_best_loss = np.inf
        self.step_best_cls_loss = np.inf
        self.step_best_dice = 0.0
        self.step_best_acc = 0.0
        self.losses = []
        self.cls_losses = []  
        self.dices = []
        self.accs = []  
        self.ious = []
        self.set_loss_fn()
        self.set_optimizer()
        self.set_lr_scheduler()
        # self.init_checkpoint(join(self.args.work_dir, self.args.task_name, 'sam_model_latest.pth'))
        self.init_checkpoint(args.checkpoint)
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = True
        self.norm_transform = tio.ZNormalization(masking_method=lambda x: x > 0)
        self.writer = writer  

    def set_loss_fn(self):
        # self.seg_loss = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
        self.cls_loss = torch.nn.CrossEntropyLoss()

    def set_optimizer(self):
        if self.args.multi_gpu:
            # sam_model = self.model.module
            cls_model = self.classifier.module
        else:
            # sam_model = self.model
            cls_model = self.classifier

        self.cls_optimizer = torch.optim.AdamW(cls_model.parameters(), lr=self.args.lr, betas=(0.9, 0.999),
                                               weight_decay=self.args.weight_decay)

    def set_lr_scheduler(self):
        if self.args.lr_scheduler == "multisteplr":
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.cls_optimizer,
                                                                     self.args.step_size,
                                                                     self.args.gamma)
        elif self.args.lr_scheduler == "steplr":
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.cls_optimizer,
                                                                self.args.step_size[0],
                                                                self.args.gamma)
        elif self.args.lr_scheduler == 'coswarm':
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.cls_optimizer, T_0=10,
                                                                                     eta_min=1e-6)
        else:
            self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(self.cls_optimizer, 0.1)

    def init_checkpoint(self, ckp_path):
        last_ckpt = None
        if os.path.exists(ckp_path):
            if self.args.multi_gpu:
                dist.barrier()
                last_ckpt = torch.load(ckp_path, map_location=self.args.device)
            else:
                last_ckpt = torch.load(ckp_path, map_location=self.args.device)

        if last_ckpt:
            if 'model_state_dict' in last_ckpt:
                if self.args.multi_gpu:
                    self.model.module.load_state_dict(last_ckpt['model_state_dict'])
                else:
                    self.model.load_state_dict(last_ckpt['model_state_dict'])
            if 'classifier_state_dict' in last_ckpt and hasattr(self, 'classifier'):
                if self.args.multi_gpu:
                    self.classifier.module.load_state_dict(last_ckpt['classifier_state_dict'])
                else:
                    self.classifier.load_state_dict(last_ckpt['classifier_state_dict'])

            if not self.args.resume:
                self.start_epoch = 0
            else:

                # self.optimizer.load_state_dict(last_ckpt['optimizer_state_dict'])
                if 'cls_optimizer_state_dict' in last_ckpt:  
                    self.cls_optimizer.load_state_dict(last_ckpt['cls_optimizer_state_dict'])
                    self.start_epoch = last_ckpt['epoch']
                else:
                    self.start_epoch = 0
                self.lr_scheduler.load_state_dict(last_ckpt['lr_scheduler_state_dict'])
                self.losses = last_ckpt['losses']
                self.dices = last_ckpt['dices']
                if 'cls_losses' in last_ckpt:
                    self.cls_losses = last_ckpt['cls_losses']
                if 'accs' in last_ckpt:
                    self.accs = last_ckpt['accs']
                self.best_loss = last_ckpt['best_loss']
                self.best_dice = last_ckpt['best_dice']
                if 'best_cls_loss' in last_ckpt:
                    self.best_cls_loss = last_ckpt['best_cls_loss']
                if 'best_acc' in last_ckpt:
                    self.best_acc = last_ckpt['best_acc']

            print(f"Loaded checkpoint from {ckp_path} (epoch {self.start_epoch})")
        else:
            self.start_epoch = 0
            print(f"No checkpoint found at {ckp_path}, start training from scratch")

    def save_checkpoint(self, epoch, state_dict, classifier_state_dict, describe="last"):
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": state_dict,
            "classifier_state_dict": classifier_state_dict,
            # "optimizer_state_dict": self.optimizer.state_dict(),
            "cls_optimizer_state_dict": self.cls_optimizer.state_dict(),
            "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
            "losses": self.losses,
            "dices": self.dices,
            "cls_losses": self.cls_losses,
            "accs": self.accs,
            "best_loss": self.best_loss,
            "best_dice": self.best_dice,
            "best_cls_loss": self.best_cls_loss,
            "best_acc": self.best_acc,
            "args": self.args,
            "used_datas": img_datas,
        }, join(MODEL_SAVE_PATH, f"sam_cls_model_{describe}.pth"))

    def batch_forward(self, sam_model, image_embedding, gt3D, low_res_masks, points=None):
        features = {}
        hook_handles = []

        def get_features(name):
            def hook(model, input, output):
                features[name] = output

            return hook

        features['feature1'] = image_embedding
        handle = sam_model.mask_decoder.output_upscaling[2].register_forward_hook(get_features('feature2'))
        hook_handles.append(handle)
        handle = sam_model.mask_decoder.output_upscaling[4].register_forward_hook(get_features('feature3'))
        hook_handles.append(handle)
        sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
            points=points,
            boxes=None,
            masks=low_res_masks,
        )
        low_res_masks, iou_predictions = sam_model.mask_decoder(
            image_embeddings=image_embedding.to(device),
            image_pe=sam_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        prev_masks = F.interpolate(low_res_masks, size=gt3D.shape[-3:], mode='trilinear', align_corners=False)

        for handle in hook_handles:
            handle.remove()

        return low_res_masks, prev_masks, features

    def get_points(self, prev_masks, gt3D):
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

    def interaction(self, sam_model, classifer, image_embedding, gt3D, num_clicks):
        return_loss = torch.tensor(1000000).to(device)  
        prev_masks = torch.zeros_like(gt3D).to(gt3D.device)
        cls_pred = None
        low_res_masks = F.interpolate(prev_masks.float(),
                                      size=(args.img_size // 4, args.img_size // 4, args.img_size // 4))
        random_insert = np.random.randint(2, 9)
        classifer = self.classifier
        for num_click in range(num_clicks):
            points_input, labels_input = self.get_points(prev_masks, gt3D)

            if num_click == random_insert or num_click == num_clicks - 1:
                if num_click == num_clicks - 1:
                    cls = True
                else:
                    cls = False
                low_res_masks, prev_masks, features = self.batch_forward(sam_model, image_embedding, gt3D,
                                                                         low_res_masks,
                                                                         points=None)  
                if cls:
                    cls_pred = classifer(features, low_res_masks)
                    features.clear()

            else:
                low_res_masks, prev_masks, features = self.batch_forward(sam_model, image_embedding, gt3D,
                                                                         low_res_masks,
                                                                         points=[points_input, labels_input])
                features.clear()
            # loss = self.seg_loss(prev_masks, gt3D)
            # return_loss += loss
        return prev_masks, return_loss, cls_pred

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
        epoch_cls_loss = 0  
        epoch_iou = 0  
        epoch_dice = 0
        epoch_acc = 0  
        total_correct = 0
        total_samples = 0
        self.model.train()
        self.classifier.train()  
        if self.args.multi_gpu:
            sam_model = self.model.module
            classifier = self.classifier.module
        else:
            sam_model = self.model
            classifier = self.classifier
            self.args.rank = -1 

        if not self.args.multi_gpu or (self.args.multi_gpu and self.args.rank == 0):
            tbar = tqdm(self.dataloaders)
        else:
            tbar = self.dataloaders

        self.cls_optimizer.zero_grad()
        # self.optimizer.zero_grad()
        step_loss = 0
        step_cls_loss = 0
        for step, (image3D, gt3D, cls_label) in enumerate(tbar):

            my_context = self.model.no_sync if self.args.rank != -1 and step % self.args.accumulation_steps != 0 else nullcontext

            with my_context():

                image3D = self.norm_transform(image3D.squeeze(dim=1))  # (N, C, W, H, D)
                image3D = image3D.unsqueeze(dim=1)

                image3D = image3D.to(device)
                gt3D = gt3D.to(device).type(torch.long)
                cls_label = cls_label.to(device)
                with amp.autocast():
                    image_embedding = sam_model.image_encoder(image3D)

                    self.click_points = []
                    self.click_labels = []

                    pred_list = []

                    prev_masks, loss, cls_pred = self.interaction(sam_model, classifier, image_embedding, gt3D,
                                                                  num_clicks=num_clicks)
                    cls_loss = self.cls_loss(cls_pred, cls_label)
                    _, pred = torch.max(cls_pred, dim=1)
                    correct = torch.sum(pred == cls_label).item()
                    total_correct += correct
                    total_samples += cls_label.size(0)
                    print_acc = correct / cls_label.size(0)

                # epoch_loss += loss.item()
                epoch_cls_loss += cls_loss.item()

                # cur_loss = loss.item()
                cur_cls_loss = cls_loss.item()

                # loss /= self.args.accumulation_steps  
                cls_loss /= self.args.accumulation_steps

                self.scaler.scale(cls_loss).backward()  

            if step % self.args.accumulation_steps == 0 and step != 0:
                self.scaler.step(self.cls_optimizer)
                self.scaler.update()
                self.cls_optimizer.zero_grad()
                # self.optimizer.zero_grad()

                print_loss = step_cls_loss / self.args.accumulation_steps  
                # step_loss = 0
                step_cls_loss = 0
            else:
                # step_loss += cur_loss
                step_cls_loss += cur_cls_loss

            if not self.args.multi_gpu or (self.args.multi_gpu and self.args.rank == 0):
                if step % self.args.accumulation_steps == 0 and step != 0:
                    tbar.set_postfix({'Epoch': epoch, 'Step': step, 'Loss': print_loss, 'Acc': print_acc})
                    wandb.log({"Step Loss": print_loss})
                    if print_loss < self.step_best_loss:
                        self.step_best_loss = print_loss
                    # self.writer.add_scalar(f'Step Loss/train/Epoch_{epoch}', print_loss, step)

        epoch_loss /= step
        epoch_cls_loss /= step
        epoch_acc = total_correct / total_samples
        return epoch_loss, epoch_cls_loss, epoch_iou, epoch_dice, epoch_acc, pred_list

    def eval_epoch(self, num_clicks):
        self.model.eval()
        self.classifier.eval()
        if self.args.multi_gpu:
            sam_model = self.model.module
            classifier = self.classifier.module
        else:
            sam_model = self.model
            classifier = self.classifier
            self.args.rank = -1  

        epoch_loss = torch.tensor(0.0).to(self.args.device)
        epoch_cls_loss = torch.tensor(0.0).to(self.args.device)
        total_correct = torch.tensor(0).to(self.args.device)
        total_samples = torch.tensor(0).to(self.args.device)

        device_val = torch.device('cuda:0') 

        if not self.args.multi_gpu or (self.args.multi_gpu and self.args.rank == 0):
            tbar = tqdm(self.val_dataloader)
        else:
            tbar = self.val_dataloader

        with torch.no_grad():
            for step, (image3D, gt3D, cls_label) in enumerate(tbar):
                image3D = self.norm_transform(image3D.squeeze(dim=1))  # (N, C, W, H, D)
                image3D = image3D.unsqueeze(dim=1)

                image3D = image3D.to(device)
                gt3D = gt3D.to(device).type(torch.long)
                cls_label = cls_label.to(device)

                with amp.autocast():
                    image_embedding = sam_model.image_encoder(image3D)

                    self.click_points = []
                    self.click_labels = []

                    prev_masks, loss, cls_pred = self.interaction(sam_model, classifier, image_embedding, gt3D,
                                                                  num_clicks=num_clicks)
                    cls_loss = self.cls_loss(cls_pred, cls_label)

                _, pred = torch.max(cls_pred, dim=1)
                correct = torch.sum(pred == cls_label).item()
                total_correct += correct
                total_samples += cls_label.size(0)

                epoch_loss += loss.item()
                epoch_cls_loss += cls_loss.item()

            if self.args.multi_gpu:
                torch.distributed.all_reduce(epoch_loss)
                torch.distributed.all_reduce(epoch_cls_loss)
                torch.distributed.all_reduce(total_correct)
                torch.distributed.all_reduce(total_samples)

            epoch_loss /= step
            epoch_cls_loss /= step
            epoch_acc = total_correct / total_samples

        return epoch_loss, epoch_cls_loss, epoch_acc

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
            epoch_loss, epoch_cls_loss, epoch_iou, epoch_dice, epoch_acc, pred_list = self.train_epoch(epoch,
                                                                                                       num_clicks)

            # Add validation after each epoch
            val_epoch_loss, val_epoch_cls_loss, val_epoch_acc = self.eval_epoch(num_clicks=3)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            if self.args.multi_gpu:
                dist.barrier()

            if not self.args.multi_gpu or (self.args.multi_gpu and self.args.rank == 0):
                self.losses.append(epoch_loss)
                self.cls_losses.append(epoch_cls_loss)
                self.dices.append(epoch_dice)
                self.accs.append(epoch_acc)
                # print(f'EPOCH: {epoch}, Loss: {epoch_loss}')
                print(f'EPOCH: {epoch}, cls_Loss: {val_epoch_cls_loss}')
                print(f'EPOCH: {epoch}, acc: {val_epoch_acc}')

                logger.info(
                    f'Epoch\t {epoch}\t : cls loss: {epoch_cls_loss}, acc: {epoch_acc}, acc: {epoch_acc}')
                # self.writer.add_scalar('Loss/train', epoch_loss, epoch)  
                # self.writer.add_scalar('Acc/train', epoch_acc, epoch)  
                wandb.log({"Val Cls Loss": val_epoch_cls_loss, "Val Acc": val_epoch_acc})
                wandb.log({"Cls Loss": epoch_cls_loss, "Acc": epoch_acc})
                if self.args.multi_gpu:
                    state_dict = self.model.module.state_dict()
                    classifier_state_dict = self.classifier.module.state_dict()
                else:
                    state_dict = self.model.state_dict()
                    classifier_state_dict = self.classifier.state_dict()

                # save latest checkpoint
                self.save_checkpoint(
                    epoch,
                    state_dict,
                    classifier_state_dict,
                    describe='latest'
                )

                # save train loss best checkpoint
                if val_epoch_cls_loss < self.best_cls_loss:
                    self.best_cls_loss = val_epoch_cls_loss
                    self.save_checkpoint(
                        epoch,
                        state_dict,
                        classifier_state_dict,
                        describe='val_epoch_cls_loss'
                    )
                    if args.KNN:
                        if self.args.multi_gpu:
                            if self.args.rank == 0:  
                                self.classifier.module.knn_memory[0].save(f'./{args.task_name}_loss/knn.memories/state1')

                # save train dice best checkpoint
                if val_epoch_acc > self.best_acc:
                    self.best_acc = val_epoch_acc
                    self.save_checkpoint(
                        epoch,
                        state_dict,
                        classifier_state_dict,
                        describe='val_epoch_acc'
                    )
                    if args.KNN:
                        if self.args.multi_gpu:
                            if self.args.rank == 0: 
                                self.classifier.module.knn_memory[0].save(f'./{args.task_name}_acc/knn.memories/state1')

                self.plot_result(self.cls_losses, 'classifier Loss', 'Loss')
                self.plot_result(self.accs, 'Acc', 'Acc')
                # self.writer.close()
            if args.KNN:
                if self.args.multi_gpu:
                    if self.args.rank == 0:  
                        self.classifier.module.knn_memory[0].save(f'./{args.task_name}/knn.memories/state1')
            # else:
            #     self.classifier.knn_memory[0].save('./knnp/knn.memories/state1')
            # self.classifier.knn_memory[1].save('./knn/knn.memories/state2')

        logger.info('=====================================================================')
        logger.info(f'Best loss: {self.best_loss}')
        logger.info(f'Best dice: {self.best_dice}')
        logger.info(f'Best acc: {self.best_acc}')
        logger.info(f'Best cls loss: {self.best_cls_loss}')
        logger.info(f'Total loss: {self.losses}')
        logger.info(f'Total dice: {self.dices}')
        logger.info(f'Total acc: {self.accs}')
        logger.info(f'Total cls loss: {self.cls_losses}')
        logger.info('=====================================================================')
        logger.info(f'args : {self.args}')
        logger.info(f'Used datasets : {img_datas}')
        logger.info('=====================================================================')


def init_seeds(seed=0, cuda_deterministic=True):
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
        # log_dir = os.path.join("./tf-logs/", args.task_name)
        # writer = SummaryWriter(log_dir=log_dir, comment=args.task_name)
        # Load datasets
        dataloaders = get_dataloaders(args)
        # val_dataloaders = get_val_dataloader(args)
        # Build model
        model = build_model(args)
        # Create trainer
        trainer = BaseTrainer(model, dataloaders, classifier, args)
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

    # start a new wandb run to track this script
    if rank == 0:  # Only initialize wandb run for the main process
        wandb.init(
            # set the wandb project where this run will be logged
            project="Acc 117",
            name=args.task_name,  # set the run name as the task name
            resume='allow',
            # id='ugt0ugpj',
            tags=["experiment", "baseline", "cls"],

            # track hyperparameters and run metadata
            config={
                "learning_rate": args.lr,
                "architecture": "SAM Med 3D",
                "dataset": "Total Segment Merged",
                "epochs": args.num_epochs,
                "batch_size": args.batch_size,
                "loss_function": "CrossEntropyLoss",
                "optimizer": "AdamW",
                "lr_scheduler": args.lr_scheduler,
            }
        )

    dataloaders = get_dataloaders(args)
    # val_dataloaders = get_val_dataloader(args)
    model = build_model(args)
    trainer = BaseTrainer(model, dataloaders, classifier, args)
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
