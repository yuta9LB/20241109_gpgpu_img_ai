import os
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms

# VOC2012で用いるラベル
CLASSES = ['backgrounds','aeroplane','bicycle','bird','boat','bottle',
            'bus','car' ,'cat','chair','cow', 
            'diningtable','dog','horse','motorbike','person', 
            'potted plant', 'sheep', 'sofa', 'train', 'monitor','unlabeld'
            ]
# カラーパレットの作成
COLOR_PALETTE = np.array(Image.open("./VOCdevkit/VOC2012_sample/SegmentationClass/2007_000170.png").getpalette()).reshape(-1,3)
COLOR_PALETTE = COLOR_PALETTE.tolist()[:len(CLASSES)]

class DataTransforme(torch.nn.Module):
    def __init__(self, input_size, data_augmentation):
        super().__init__()
        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize(size=(input_size, input_size), interpolation=transforms.InterpolationMode.NEAREST)
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.data_augmentation = data_augmentation
        self.random_flip = transforms.RandomHorizontalFlip(p=0.5)
        self.random_rotation = transforms.RandomRotation(15)

    def forward(self, img, gt):
        # データ拡張が有効ならば適用
        if self.data_augmentation and gt is not None:
            img, gt = self.augment(img, gt)

        img_tensor = self.to_tensor(img)
        img_resized = self.resize(img_tensor)
        img_normalized = self.normalize(img_resized)

        gt_np = np.asarray(gt)
        gt_np = np.where(gt_np == 255, 0, gt_np)
        gt_tensor = torch.tensor(gt_np, dtype=torch.long)
        gt_tensor = gt_tensor.unsqueeze(0)
        gt_resized = self.resize(gt_tensor)

        return img_normalized, gt_resized
    
    def augment(self, img, gt):
        # 同じシードでランダム変換を適用
        seed = torch.randint(0, 2**32, (1,)).item()
        
        # ランダム水平反転
        torch.manual_seed(seed)
        img = self.random_flip(img)
        torch.manual_seed(seed)
        gt = self.random_flip(gt)
        
        # ランダム回転
        torch.manual_seed(seed)
        img = self.random_rotation(img)
        torch.manual_seed(seed)
        gt = self.random_rotation(gt)
        
        return img, gt
    
class VOCDataset(Dataset):
    def __init__(
            self, 
            img_list: list, 
            img_dir: str,
            gt_dir: str,
            data_augmentation=False,
    ):
        self.img_list = img_list
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.preprocessing = DataTransforme(input_size=256, data_augmentation=data_augmentation)
        self.img_fps = [os.path.join(img_dir, img_id)+".jpg" for img_id in self.img_list]
        self.gt_fps = [os.path.join(gt_dir, gt_id)+".png" for gt_id in self.img_list]
        self.CLASSES = CLASSES
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in self.CLASSES]

    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, i):
        img = Image.open(self.img_fps[i])
        gt = Image.open(self.gt_fps[i])
        
        # 前処理
        img, gt = self.preprocessing(img, gt)

        return img, gt
