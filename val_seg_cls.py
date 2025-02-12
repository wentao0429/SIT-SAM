import os

join = os.path.join
import torch
from segment_anything.build_sam3D import sam_model_registry3D
from segment_anything.utils.transforms3D import ResizeLongestSide3D
from tqdm import tqdm
import argparse
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchio as tio
import numpy as np
from collections import OrderedDict
import json
import pickle
from utils.click_method import get_next_click3D_torch_ritm, get_next_click3D_torch_2
from utils.data_loader_cls import Dataset_Union_ALL
import nibabel as nib
from utils.data_paths import all_classes_merged, all_classes
from segment_anything.build_cls import classifier

from monai.metrics import compute_surface_dice

parser = argparse.ArgumentParser()
parser.add_argument('-tdp', '--test_data_path', type=str, default='./data/validation')
parser.add_argument('-vp', '--vis_path', type=str, default='./74_seg_cls_1')
parser.add_argument('-cp', '--checkpoint_path', type=str, default='./ckpt/sam_med3d.pth')
parser.add_argument('--save_name', type=str, default='74_seg_cls_1.py')

parser.add_argument('--image_size', type=int, default=256)
parser.add_argument('--crop_size', type=int, default=128)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('-mt', '--model_type', type=str, default='vit_b_ori')
parser.add_argument('-nc', '--num_clicks', type=int, default=5)
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
print("set seed as", SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

if torch.cuda.is_available():
    torch.cuda.init()

click_methods = {
    'default': get_next_click3D_torch_ritm,
    'ritm': get_next_click3D_torch_ritm,
    'random': get_next_click3D_torch_2,
}


def compute_iou(pred_mask, gt_semantic_seg):
    in_mask = np.logical_and(gt_semantic_seg, pred_mask)
    out_mask = np.logical_or(gt_semantic_seg, pred_mask)
    iou = np.sum(in_mask) / np.sum(out_mask)
    return iou


def compute_dice(mask_gt, mask_pred):
    """Compute soerensen-dice coefficient.
    Returns:
    the dice coeffcient as float. If both masks are empty, the result is NaN
    """
    volume_sum = mask_gt.sum() + mask_pred.sum()
    if volume_sum == 0:
        return np.NaN
    volume_intersect = (mask_gt & mask_pred).sum()
    return 2 * volume_intersect / volume_sum


def postprocess_masks(low_res_masks, image_size, original_size):
    ori_h, ori_w = original_size
    masks = F.interpolate(
        low_res_masks,
        (image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )
    if args.ft2d and ori_h < image_size and ori_w < image_size:
        top = (image_size - ori_h) // 2
        left = (image_size - ori_w) // 2
        masks = masks[..., top: ori_h + top, left: ori_w + left]
        pad = (top, left)
    else:
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        pad = None
    return masks, pad


def sam_decoder_inference(target_size, points_coords, points_labels, model, image_embeddings, mask_inputs=None,
                          multimask=False):
    with torch.no_grad():
        sparse_embeddings, dense_embeddings = model.prompt_encoder(
            points=(points_coords.to(model.device), points_labels.to(model.device)),
            boxes=None,
            masks=mask_inputs,
        )

        low_res_masks, iou_predictions = model.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask,
        )

    if multimask:
        max_values, max_indexs = torch.max(iou_predictions, dim=1)
        max_values = max_values.unsqueeze(1)
        iou_predictions = max_values
        low_res = []
        for i, idx in enumerate(max_indexs):
            low_res.append(low_res_masks[i:i + 1, idx])
        low_res_masks = torch.stack(low_res, 0)
    masks = F.interpolate(low_res_masks, (target_size, target_size), mode="bilinear", align_corners=False, )
    return masks, low_res_masks, iou_predictions


def repixel_value(arr, is_seg=False):
    if not is_seg:
        min_val = arr.min()
        max_val = arr.max()
        new_arr = (arr - min_val) / (max_val - min_val + 1e-10) * 255.
    return new_arr


def random_point_sampling(mask, get_point=1):
    if isinstance(mask, torch.Tensor):
        mask = mask.numpy()
    fg_coords = np.argwhere(mask == 1)[:, ::-1]
    bg_coords = np.argwhere(mask == 0)[:, ::-1]

    fg_size = len(fg_coords)
    bg_size = len(bg_coords)

    if get_point == 1:
        if fg_size > 0:
            index = np.random.randint(fg_size)
            fg_coord = fg_coords[index]
            label = 1
        else:
            index = np.random.randint(bg_size)
            fg_coord = bg_coords[index]
            label = 0
        return torch.as_tensor([fg_coord.tolist()], dtype=torch.float), torch.as_tensor([label], dtype=torch.int)
    else:
        num_fg = get_point // 2
        num_bg = get_point - num_fg
        fg_indices = np.random.choice(fg_size, size=num_fg, replace=True)
        bg_indices = np.random.choice(bg_size, size=num_bg, replace=True)
        fg_coords = fg_coords[fg_indices]
        bg_coords = bg_coords[bg_indices]
        coords = np.concatenate([fg_coords, bg_coords], axis=0)
        labels = np.concatenate([np.ones(num_fg), np.zeros(num_bg)]).astype(int)
        indices = np.random.permutation(get_point)
        coords, labels = torch.as_tensor(coords[indices], dtype=torch.float), torch.as_tensor(labels[indices],
                                                                                              dtype=torch.int)
        return coords, labels


def finetune_model_predict3D(img3D, gt3D, sam_model_tune, classifier, cls_label, device='cuda', click_method='random',
                             num_clicks=5,
                             prev_masks=None):
    img3D = norm_transform(img3D.squeeze(dim=1))  # (N, C, W, H, D)
    img3D = img3D.unsqueeze(dim=1)
    classifier = classifier.to(device)
    cls_label = cls_label.to(device)

    click_points = []
    click_labels = []

    pred_list = []

    iou_list = []
    dice_list = []
    nsd_list = []
    acc_list = []

    if prev_masks is None:
        prev_masks = torch.zeros_like(gt3D).to(device)
    low_res_masks = F.interpolate(prev_masks.float(),
                                  size=(args.crop_size // 4, args.crop_size // 4, args.crop_size // 4))

    with torch.no_grad():
        image_embedding = sam_model_tune.image_encoder(img3D.to(device))  # (1, 384, 16, 16, 16)
    for num_click in range(num_clicks):
        with torch.no_grad():
            if (num_click > 1):
                click_method = "random"
            features = {}
            # 注册钩子并保存句柄以便稍后移除
            hook_handles = []

            def get_features(name):
                def hook(model, input, output):
                    features[name] = output

                return hook

            batch_points, batch_labels = click_methods[click_method](prev_masks.to(device), gt3D.to(device))

            points_co = torch.cat(batch_points, dim=0).to(device)
            points_la = torch.cat(batch_labels, dim=0).to(device)

            click_points.append(points_co)
            click_labels.append(points_la)

            points_input = points_co
            labels_input = points_la

            features['feature1'] = image_embedding
            handle = sam_model_tune.mask_decoder.output_upscaling[2].register_forward_hook(get_features('feature2'))
            hook_handles.append(handle)
            handle = sam_model_tune.mask_decoder.output_upscaling[4].register_forward_hook(get_features('feature3'))
            hook_handles.append(handle)

            sparse_embeddings, dense_embeddings = sam_model_tune.prompt_encoder(
                points=[points_input, labels_input],
                boxes=None,
                masks=low_res_masks.to(device),
            )
            low_res_masks, _ = sam_model_tune.mask_decoder(
                image_embeddings=image_embedding.to(device),  # (B, 384, 64, 64, 64)
                image_pe=sam_model_tune.prompt_encoder.get_dense_pe(),  # (1, 384, 64, 64, 64)
                sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 384)
                dense_prompt_embeddings=dense_embeddings,  # (B, 384, 64, 64, 64)
                multimask_output=False,
            )
            cls_pred = classifier(features, low_res_masks)
            cls_pred = torch.nn.functional.softmax(cls_pred, dim=1)
            # 清除features中的所有键，因为它们已经不再需要
            features.clear()

            prev_masks = F.interpolate(low_res_masks, size=gt3D.shape[-3:], mode='trilinear', align_corners=False)

            medsam_seg_prob = torch.sigmoid(prev_masks)  # (B, 1, 64, 64, 64)
            # convert prob to mask
            medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
            medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)
            pred_list.append(medsam_seg)

            iou_list.append(round(compute_iou(medsam_seg, gt3D[0][0].detach().cpu().numpy()), 4))
            dice_list.append(round(compute_dice(gt3D[0][0].detach().cpu().numpy().astype(np.uint8), medsam_seg), 4))
            gt_tensor = torch.from_numpy(gt3D.detach().cpu().numpy().astype(np.uint8))
            medsam_tensor = torch.from_numpy(medsam_seg.astype(np.uint8)).unsqueeze(0).unsqueeze(0)
            nsd_list.append(
                round(compute_surface_dice(gt_tensor, medsam_tensor, [1.0]).item(), 4)
            )
            acc_list.append(round(cls_pred[0][cls_label[0]].item(), 4))

    return pred_list, click_points, click_labels, iou_list, dice_list, nsd_list, acc_list


if __name__ == "__main__":
    all_dataset_paths = args.test_data_path
    infer_transform = [
        tio.ToCanonical(),
        tio.Resample(target=(1.5, 1.5, 1.5)),
        tio.CropOrPad(mask_name='label', target_shape=(args.crop_size, args.crop_size, args.crop_size)),
    ]

    test_dataset = Dataset_Union_ALL(
        paths=all_dataset_paths,
        all_classes=all_classes_merged,
        mode="test",
        data_type=args.data_type,
        transform=tio.Compose(infer_transform),
        threshold=0,
        split_num=args.split_num,
        split_idx=args.split_idx,
        pcc=False,
    )  # 注意

    test_dataloader = DataLoader(
        dataset=test_dataset,
        sampler=None,
        batch_size=1,  # 注意这里的batch size只能是1
        shuffle=False,
        num_workers=16,
    )

    checkpoint_path = args.checkpoint_path

    device = args.device
    print("device:", device)

    sam_model_tune = sam_model_registry3D[args.model_type](checkpoint=None).to(device)
    classifier = classifier(input_dim=528, num_classes=args.num_classes, mlp_ratio=2.0).to(device)

    if checkpoint_path is not None:
        model_dict = torch.load(checkpoint_path, map_location=device)
        state_dict = model_dict['model_state_dict']
        sam_model_tune.load_state_dict(state_dict)
        classifier.load_state_dict(model_dict['classifier_state_dict'])

    sam_trans = ResizeLongestSide3D(sam_model_tune.image_encoder.img_size)

    all_iou_list = []
    all_dice_list = []
    all_nsd_list = []
    all_acc_list = []

    out_dice = dict()
    out_dice_all = OrderedDict()

    for batch_data in tqdm(test_dataloader):
        image3D, gt3D, img_name, class_id, affine = batch_data
        # print(affine)
        gt_voxel_value = (gt3D[0][0] > 0.5).sum().item()
        sz = image3D.size()
        if sz[2] < args.crop_size or sz[3] < args.crop_size or sz[4] < args.crop_size:
            print("[ERROR] wrong size", sz, "for", img_name)
        modality = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(img_name[0]))))
        dataset = os.path.basename(os.path.dirname(os.path.dirname(img_name[0])))
        vis_root = os.path.join(os.path.dirname(__file__), args.vis_path, modality, dataset)
        pred_path = os.path.join(vis_root,
                                 os.path.basename(img_name[0]).replace(".nii.gz", f"_pred{args.num_clicks - 1}.nii.gz"))
        if os.path.exists(pred_path):
            iou_list, dice_list, nsd_list = [], [], []
            for iter in range(args.num_clicks):
                curr_pred_path = os.path.join(vis_root,
                                              os.path.basename(img_name[0]).replace(".nii.gz", f"_pred{iter}.nii.gz"))
                medsam_seg = nib.load(curr_pred_path).get_fdata()
                iou_list.append(round(compute_iou(medsam_seg, gt3D[0][0].detach().cpu().numpy()), 4))
                dice_list.append(
                    round(compute_dice(gt3D[0][0].detach().cpu().numpy().astype(np.uint8), medsam_seg.astype(np.uint8)),
                          4))
                gt_tensor = torch.from_numpy(gt3D.detach().cpu().numpy().astype(np.uint8))
                medsam_tensor = torch.from_numpy(medsam_seg.astype(np.uint8)).unsqueeze(0).unsqueeze(0)
                nsd_list.append(
                    round(compute_surface_dice(gt_tensor, medsam_tensor, [1.0]).item(), 4)  # TODO: 在total的文章中是3.0
                )

        else:
            norm_transform = tio.ZNormalization(masking_method=lambda x: x > 0)
            if args.dim == 3:
                seg_mask_list, points, labels, iou_list, dice_list, nsd_list, acc_list = finetune_model_predict3D(
                    image3D, gt3D, sam_model_tune, classifier, class_id, device=device,
                    click_method=args.point_method, num_clicks=args.num_clicks,
                    prev_masks=None)
                print(acc_list)
            os.makedirs(vis_root, exist_ok=True)
            points = [p.cpu().numpy() for p in points]
            labels = [l.cpu().numpy() for l in labels]
            pt_info = dict(points=points, labels=labels)
            print("save to", os.path.join(vis_root, os.path.basename(img_name[0]).replace(".nii.gz", "_pred.nii.gz")))
            pt_path = os.path.join(vis_root, os.path.basename(img_name[0]).replace(".nii.gz", "_pt.pkl"))
            pickle.dump(pt_info, open(pt_path, "wb"))
            for idx, pred3D in enumerate(seg_mask_list):
                out = nib.Nifti1Image(pred3D, affine[0])  # Use nibabel to create Nifti image
                nib.save(out, os.path.join(vis_root, os.path.basename(img_name[0]).replace(".nii.gz",
                                                                                           f"_pred{idx}.nii.gz")))

        all_iou_list.append(max(iou_list))
        # all_iou_list.append(sum(iou_list) / len(iou_list) if iou_list else 0)
        all_dice_list.append(max(dice_list))
        # all_dice_list.append(sum(dice_list) / len(dice_list) if dice_list else 0)
        all_nsd_list.append(max(nsd_list))
        # all_nsd_list.append(sum(nsd_list) / len(nsd_list) if nsd_list else 0)
        all_acc_list.append(max(acc_list))
        # all_acc_list.append(sum(acc_list) / len(acc_list) if acc_list else 0)
        print(dice_list)
        print(acc_list)
        out_dice[img_name] = max(dice_list)
        cur_dice_dict = OrderedDict()
        cur_iou_dict = OrderedDict()
        cur_nsd_dict = OrderedDict()
        cur_acc_dict = OrderedDict()
        for i, dice in enumerate(dice_list):
            cur_dice_dict[f'{i}_dice'] = dice
        for i, iou in enumerate(iou_list):
            cur_iou_dict[f'{i}_iou'] = iou
        for i, nsd_v in enumerate(nsd_list):
            cur_nsd_dict[f'{i}_nsd'] = nsd_v
        for i, acc in enumerate(acc_list):
            cur_acc_dict[f'{i}_acc'] = acc
        cur_dict = {
            "dice": cur_dice_dict,
            "iou": cur_iou_dict,
            "gt_voxel_value": gt_voxel_value,
            "nsd": cur_nsd_dict,
            "acc": cur_acc_dict
        }
        out_dice_all[img_name[0]] = cur_dict

    print('Mean IoU : ', sum(all_iou_list) / len(all_iou_list))
    print('Mean Dice: ', sum(all_dice_list) / len(all_dice_list))
    print('Mean NSD: ', sum(all_nsd_list) / len(all_nsd_list))
    print('Mean ACC: ', sum(all_acc_list) / len(all_acc_list))

    final_dice_dict = OrderedDict()
    for k, v in out_dice_all.items():
        organ = k.split('/')[-4]
        final_dice_dict[organ] = OrderedDict()
    for k, v in out_dice_all.items():
        organ = k.split('/')[-4]
        final_dice_dict[organ][k] = v

    if (args.split_num > 1):
        args.save_name = args.save_name.replace('.py', f'_s{args.split_num}i{args.split_idx}.py')

    print("Save to", args.save_name)
    with open(args.save_name, 'w') as f:
        f.writelines(f'# mean dice: \t{np.mean(all_dice_list)}\n')
        f.writelines('dice_Ts = {')
        for k, v in out_dice.items():
            f.writelines(f'\'{str(k[0])}\': {v},\n')
        f.writelines('}')

    with open(args.save_name.replace('.py', '.json'), 'w') as f:
        json.dump(final_dice_dict, f, indent=4)

    print("Done")
