from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchio as tio
import numpy as np
import os
import torch
import nibabel as nib
from prefetch_generator import BackgroundGenerator
from data_paths import all_classes


class Dataset_Union_ALL(Dataset):
    def __init__(self, paths, all_classes, mode='train', data_type='Tr', image_size=128,
                 transform=None, threshold=500,
                 split_num=1, split_idx=0, pcc=False):
        self.paths = paths
        self.data_type = data_type
        self.split_num = split_num
        self.split_idx = split_idx
        self.all_classes = all_classes
        self.class_name_to_id = {class_name: i for i, class_name in enumerate(self.all_classes)}

        self._set_file_paths(self.paths)
        self.image_size = image_size
        self.transform = transform
        self.threshold = threshold
        self.mode = mode
        self.pcc = pcc
        # print(self.label_paths)
        self.last_image = None
        self.last_path = None

    def __len__(self):
        return len(self.label_paths)

    def __getitem__(self, index):
        if self.image_paths[index] == self.last_path:
            # If it is, use the last loaded image
            nib_image = self.last_image
        else:
            # If not, load the new image and save it
            nib_image = nib.load(self.image_paths[index])
            self.last_image = nib_image
            self.last_path = self.image_paths[index]

        cls_classes_name = self.class_labels[index]
        cls_classes_id = self.class_name_to_id[cls_classes_name]
        cls_classes_id = torch.tensor(cls_classes_id, dtype=torch.long)
        nib_label = nib.load(self.label_paths[index])
        image_data = np.expand_dims(nib_image.get_fdata(), axis=0)
        label_data = np.expand_dims(nib_label.get_fdata(), axis=0)

        subject = tio.Subject(
            image=tio.ScalarImage(tensor=image_data),
            label=tio.LabelMap(tensor=label_data),
        )

        # if '/ct_' in self.image_paths[index]:
        subject = tio.Clamp(-1000, 1000)(subject)

        if self.transform:
            try:
                subject = self.transform(subject)
            except:
                print(self.image_paths[index])

        if (self.pcc):
            print("using pcc setting")
            # crop from random click point
            random_index = torch.argwhere(subject.label.data == 1)
            if (len(random_index) >= 1):
                random_index = random_index[np.random.randint(0, len(random_index))]
                # print(random_index)
                crop_mask = torch.zeros_like(subject.label.data)
                # print(crop_mask.shape)
                crop_mask[random_index[0]][random_index[1]][random_index[2]][random_index[3]] = 1
                subject.add_image(tio.LabelMap(tensor=crop_mask,
                                               affine=subject.label.affine),
                                  image_name="crop_mask")
                subject = tio.CropOrPad(mask_name='crop_mask',
                                        target_shape=(self.image_size, self.image_size, self.image_size))(subject)

        if subject.label.data.sum() <= self.threshold:
            return self.__getitem__(np.random.randint(self.__len__()))

        if self.mode == "train" and self.data_type == 'Tr':
            return subject.image.data.clone().detach(), subject.label.data.clone().detach()
        else:
            return subject.image.data.clone().detach(), subject.label.data.clone().detach(), cls_classes_id, cls_classes_name

    def _set_file_paths(self, paths):
        self.image_paths = []
        self.label_paths = []
        self.class_labels = []

        all_dataset_paths = [os.path.join(paths, case) for case in os.listdir(paths)]
        # 确保只包含目录
        all_dataset_paths = [path for path in all_dataset_paths if os.path.isdir(path)]

        for case_path in all_dataset_paths:
            ct_path = os.path.join(case_path, 'ct.nii')
            # segmentations_path = os.path.join(case_path, 'merged_segmentations')
            segmentations_path = os.path.join(case_path, 'segmentations')

            if os.path.exists(segmentations_path):
                for seg_file in os.listdir(segmentations_path):
                    seg_path = os.path.join(segmentations_path, seg_file)
                    self.image_paths.append(ct_path)
                    self.label_paths.append(seg_path)

                    # 假设类别信息可以从seg_file的文件名中获取
                    class_label = self._get_class_from_filename(seg_file)
                    self.class_labels.append(class_label)

    def _get_class_from_filename(self, filename):
        # 这是一个示例方法，你需要根据你的文件名格式来实现它
        # 假设文件名的格式是"classname.nii.gz"
        class_name = filename.split('.nii.gz')[0]
        return class_name


class Union_Dataloader(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class Test_Single(Dataset):
    def __init__(self, paths, image_size=128, transform=None, threshold=500):
        self.paths = paths

        self._set_file_paths(self.paths)
        self.image_size = image_size
        self.transform = transform
        self.threshold = threshold

    def __len__(self):
        return len(self.label_paths)

    def __getitem__(self, index):

        nib_image = nib.load(self.image_paths[index])
        nib_label = nib.load(self.label_paths[index])

        # 添加一个额外的维度
        image_data = np.expand_dims(nib_image.get_fdata(), axis=0)
        label_data = np.expand_dims(nib_label.get_fdata(), axis=0)

        subject = tio.Subject(
            image=tio.ScalarImage(tensor=image_data),
            label=tio.LabelMap(tensor=label_data),
        )

        subject = tio.Subject(
            image=tio.ScalarImage(tensor=nib_image.get_fdata()),
            label=tio.LabelMap(tensor=nib_label.get_fdata()),
        )

        if '/ct_' in self.image_paths[index]:
            subject = tio.Clamp(-1000, 1000)(subject)

        if self.transform:
            try:
                subject = self.transform(subject)
            except:
                print(self.image_paths[index])

        if subject.label.data.sum() <= self.threshold:
            return self.__getitem__(np.random.randint(self.__len__()))

        return subject.image.data.clone().detach(), subject.label.data.clone().detach(), self.image_paths[index]

    def _set_file_paths(self, paths):
        self.image_paths = []
        self.label_paths = []

        self.image_paths.append(paths)
        self.label_paths.append(paths.replace('images', 'labels'))


if __name__ == "__main__":
    test_dataset = Dataset_Union_ALL(
        paths=
        '../data/test/',
        data_type='Ts',
        all_classes=all_classes,
        transform=tio.Compose([
            tio.ToCanonical(),
            tio.CropOrPad(mask_name='label', target_shape=(128, 128, 128)),
        ]),
        threshold=0)

    test_dataloader = Union_Dataloader(
        dataset=test_dataset,
        sampler=None,
        batch_size=1,
        shuffle=True,
        num_workers=64,
    )

    for i, data in enumerate(test_dataloader):
        images, labels, image_id, image_name = data
        print(f'Images: {images.shape}')
        print(f'Labels: {labels.shape}')
        print(f'image_id: {image_id}')
        print(f'image_name: {image_name}')
        break
