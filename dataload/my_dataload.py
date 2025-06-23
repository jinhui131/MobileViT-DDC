from PIL import Image
import torch
from torch.utils.data import Dataset
import random
import os
from sklearn.model_selection import train_test_split

class my_dataload(Dataset):
    def __init__(self, data_path: str, split='train', transform=None, val_ratio=0.2):
        self.data_path = data_path
        self.transform = transform
        self.split = split

        random.seed(0)  # 保证随机结果可复现
        assert os.path.exists(data_path), f"dataset root: {data_path} does not exist."

        # 遍历文件夹，一个文件夹对应一个类别
        flower_class = [cla for cla in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, cla))]
        self.num_class = len(flower_class)
        # 排序，保证顺序一致
        flower_class.sort()
        # 生成类别名称以及对应的数字索引
        class_indices = dict((cla, idx) for idx, cla in enumerate(flower_class))

        all_images_path = []  # 存储所有图片路径
        all_images_label = []  # 存储所有图片对应索引信息
        supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
        # 遍历每个文件夹下的文件
        for cla in flower_class:
            cla_path = os.path.join(data_path, cla)
            # 遍历获取 supported 支持的所有文件路径
            images = [os.path.join(cla_path, i) for i in os.listdir(cla_path) if os.path.splitext(i)[-1] in supported]
            # 获取该类别对应的索引
            image_class = class_indices[cla]
            # 写入列表
            for img_path in images:
                all_images_path.append(img_path)
                all_images_label.append(image_class)

        if len(all_images_path) == 0:
            raise ValueError("No images found in the dataset.")

        # 将数据集按比例划分为训练集和验证集
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            all_images_path, all_images_label, test_size=val_ratio,
            stratify=all_images_label, random_state=0)

        if split == 'train':
            self.images_path = train_paths
            self.images_label = train_labels
        elif split == 'val':
            self.images_path = val_paths
            self.images_label = val_labels
        else:
            raise ValueError("split should be 'train' or 'val'")
        print(f"{len(self.images_path)} {split} images were found in the dataset.")

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, idx):
        img = Image.open(self.images_path[idx])
        label = self.images_label[idx]
        if img.mode != 'RGB':
            img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels
