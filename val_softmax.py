import torch

import argparse
import os
import torch.nn.functional as F
import torchio as tio
import numpy as np
from torchmetrics import Accuracy, F1Score, AUROC, ConfusionMatrix, Precision, Recall
from tqdm import tqdm

from segment_anything.build_cls_knn import classifier
from segment_anything.build_sam3D import sam_model_registry3D
from utils.click_method import get_next_click3D_torch_ritm, get_next_click3D_torch_2
from utils.data_loader_cls import Dataset_Union_ALL, Union_Dataloader
from utils.data_paths import all_classes_merged, all_classes
from utils.data_paths import img_test_datas

import time
from fvcore.nn import FlopCountAnalysis

num_cores = os.cpu_count()
print(f"num_cores: {num_cores}")

parser = argparse.ArgumentParser()
parser.add_argument('-tdp', '--test_data_path', type=str, default='./data/validation')
parser.add_argument('-vp', '--vis_path', type=str, default='./visualization')
parser.add_argument('-cp', '--checkpoint_path', type=str, default='./ckpt/sam_med3d.pth')
parser.add_argument('--save_name', type=str, default='union_out_dice.py')

parser.add_argument('--image_size', type=int, default=256)
parser.add_argument('--crop_size', type=int, default=128)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('-mt', '--model_type', type=str, default='vit_b_ori')
parser.add_argument('-nc', '--num_clicks', type=int, default=1)
parser.add_argument('-pm', '--point_method', type=str, default='default')
parser.add_argument('-dt', '--data_type', type=str, default='Ts')

parser.add_argument('--threshold', type=int, default=0)
parser.add_argument('--dim', type=int, default=3)
parser.add_argument('--split_idx', type=int, default=0)
parser.add_argument('--split_num', type=int, default=1)
parser.add_argument('--ft2d', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=2023)
parser.add_argument('--num_classes', type=int, default=117)

args = parser.parse_args()

SEED = args.seed
torch.manual_seed(SEED)
np.random.seed(SEED)

click_methods = {
    'default': get_next_click3D_torch_ritm,
    'ritm': get_next_click3D_torch_ritm,
    'random': get_next_click3D_torch_2,
}

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def batch_forward(sam_model, image_embedding, gt3D, low_res_masks, points=None):
    features = {}
    # 注册钩子并保存句柄以便稍后移除
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

    # 使用完毕后移除所有钩子
    for handle in hook_handles:
        handle.remove()

    return low_res_masks, prev_masks, features


# @timer_decorator
# @profile
def get_points(prev_masks, gt3D, click_points, click_labels, device='cuda'):
    click_type = "random"
    batch_points, batch_labels = click_methods[click_type](prev_masks, gt3D)

    points_co = torch.cat(batch_points, dim=0).to(device)
    points_la = torch.cat(batch_labels, dim=0).to(device)

    click_points.append(points_co)
    click_labels.append(points_la)

    points_input = points_co
    labels_input = points_la
    return points_input, labels_input, click_points, click_labels


# @timer_decorator
# @profile
def interaction(sam_model, classifier, image_embedding, gt3D, num_clicks, click_points, click_labels, click_type,
                device='cuda'):
    return_loss = 1000000  # TODO:应该为0
    prev_masks = torch.zeros_like(gt3D).to(gt3D.device)
    cls_pred = None
    low_res_masks = F.interpolate(prev_masks.float(),
                                  size=(args.crop_size // 4, args.crop_size // 4, args.crop_size // 4))
    classifier = classifier.to(device)
    for num_click in range(num_clicks):
        points_input, labels_input, click_points, click_labels = get_points(prev_masks, gt3D, click_points,
                                                                            click_labels, device)

        if num_click == num_clicks - 1:
            low_res_masks, prev_masks, features = batch_forward(sam_model, image_embedding, gt3D,
                                                                low_res_masks,
                                                                points=[points_input, labels_input])  # 为了节省显存,还可以优化一下

            cls_pred = classifier(features, low_res_masks)
            # flops = FlopCountAnalysis(classifier, (features, low_res_masks))
            # print(f"Flops: {flops.total()/1e9:.1f} G")
            # 清除features中的所有键，因为它们已经不再需要
            features.clear()

        else:
            low_res_masks, prev_masks, features = batch_forward(sam_model, image_embedding, gt3D,
                                                                low_res_masks,
                                                                points=[points_input, labels_input])
            # 如果不需要保留features中的数据，可以在这里清理
            features.clear()
    return prev_masks, return_loss, cls_pred, click_points, click_labels


# @timer_decorator
# @profile
def finetune_model_predict3D(img3D, gt3D, sam_model_tune, classifier, cls_label, device='cuda', click_method='random',
                             num_clicks=5,
                             prev_masks=None):
    with torch.no_grad():
        cls_loss = torch.nn.CrossEntropyLoss()
        img3D = norm_transform(img3D.squeeze(dim=1))  # (N, 1, W, H, D)
        img3D = img3D.unsqueeze(dim=1)
        image3D = img3D.to(device)
        gt3D = gt3D.to(device).type(torch.long)
        cls_label = cls_label.to(device)
        # start = time.time()
        image_embedding = sam_model_tune.image_encoder(image3D)

        click_points = []
        click_labels = []

        prev_masks, loss, cls_pred, click_points, click_labels = interaction(sam_model_tune, classifier,
                                                                             image_embedding, gt3D,
                                                                             num_clicks, click_points, click_labels,
                                                                             click_method)
        # print(f"Time: {time.time() - start}")
        cls_loss = cls_loss(cls_pred, cls_label)
        _, pred = torch.max(cls_pred, dim=1)
        correct = torch.sum(pred == cls_label).item()
        samples = cls_label.size(0)
        acc = correct / samples

        return cls_loss, correct, samples, acc, cls_pred, cls_label


if __name__ == "__main__":
    infer_transform = [
        tio.ToCanonical(),
        tio.Resample(target=(1.5, 1.5, 1.5)),
        tio.CropOrPad(mask_name='label', target_shape=(args.crop_size, args.crop_size, args.crop_size)),
    ]

    test_dataset = Dataset_Union_ALL(
        paths=img_test_datas[0],
        all_classes=all_classes,
        mode="test",
        data_type=args.data_type,
        transform=tio.Compose(infer_transform),
        threshold=0,
        split_num=args.split_num,
        split_idx=args.split_idx,
        pcc=False,
    )

    test_dataloader = Union_Dataloader(
        dataset=test_dataset,
        sampler=None,
        batch_size=16,
        num_workers=64,
        pin_memory=True,
        drop_last=True,
    )

    checkpoint_path = args.checkpoint_path

    device = args.device

    accuracy = Accuracy(task='multiclass', num_classes=args.num_classes).to(device)
    f1_score = F1Score(task='multiclass', num_classes=args.num_classes, average='macro').to(device)
    auc = AUROC(task='multiclass', num_classes=args.num_classes).to(device)
    confmat = ConfusionMatrix(task='multiclass', num_classes=args.num_classes).to(device)
    precision = Precision(task='multiclass', num_classes=args.num_classes, average='macro').to(device)
    recall = Recall(task='multiclass', num_classes=args.num_classes, average='macro').to(device)

    sam_model_tune = sam_model_registry3D[args.model_type](checkpoint=None).to(device)

    classifier = classifier(input_dim=528, num_classes=args.num_classes, mlp_ratio=2.0).to(device)

    num_params = count_parameters(sam_model_tune)
    print(f"Number of parameters in SAM: {num_params}")
    num_params = count_parameters(classifier)
    print(f"Number of parameters in Classifier: {num_params}")

    if checkpoint_path is not None:
        model_dict = torch.load(checkpoint_path, map_location=device)
        sam_state_dict = model_dict['model_state_dict']
        sam_model_tune.load_state_dict(sam_state_dict)
        classifier.load_state_dict(model_dict['classifier_state_dict'])
        sam_model_tune.eval()
        classifier.eval()
    test_dataloader = tqdm(test_dataloader)
    for batch_data in test_dataloader:
        image3D, gt3D, _, cls_id, _ = batch_data
        # print(cls_id)
        norm_transform = tio.ZNormalization(masking_method=lambda x: x > 0)
        cls_loss, correct, samples, acc, cls_pred, cls_label = finetune_model_predict3D(
            image3D, gt3D, sam_model_tune, classifier, cls_id, device=device,
            click_method=args.point_method, num_clicks=args.num_clicks,
            prev_masks=None)

        # print(torch.max(cls_pred.softmax(dim=-1), dim=1))

        accuracy(cls_pred.softmax(dim=-1), cls_label)
        f1_score(cls_pred.softmax(dim=-1), cls_label)
        auc(cls_pred.softmax(dim=-1), cls_label)
        confmat(cls_pred.softmax(dim=-1), cls_label)
        precision(cls_pred.softmax(dim=-1), cls_label)
        recall(cls_pred.softmax(dim=-1), cls_label)

        # print('Accuracy: ', acc)
        test_dataloader.set_postfix({'Accuracy_torchmetrics:': accuracy.compute(), 'F1 Score:': f1_score.compute()})

    print('Accuracy: ', accuracy.compute())
    print('F1 Score: ', f1_score.compute())
    print('AUC: ', auc.compute())
    print('Precision: ', precision.compute())
    print('Recall: ', recall.compute())
    confusion_matrix = confmat.compute().cpu().numpy()

    # Get the directory of the checkpoint file
    ckpt_dir = os.path.dirname(args.checkpoint_path)

    log_file = os.path.join(ckpt_dir, 'test.log')
    with open(log_file, 'a') as f:
        f.write(f'Number of Clicks: {args.num_clicks}\n')
        f.write(f'Accuracy: {accuracy.compute()}\n')
        f.write(f'F1 Score: {f1_score.compute()}\n')
        f.write(f'AUC: {auc.compute()}\n')
        f.write(f'Precision: {precision.compute()}\n')
        f.write(f'Recall: {recall.compute()}\n')

    # Join the directory with the filename
    output_file = os.path.join(ckpt_dir, f'confusion_matrix_{args.num_clicks}.csv')

    # Save the confusion matrix to the file
    np.savetxt(output_file, confusion_matrix.astype(float), delimiter=',', fmt='%d')

    print("Done")
