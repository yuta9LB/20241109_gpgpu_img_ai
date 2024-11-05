import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# VOC2012で用いるラベル
CLASSES = ['backgrounds','aeroplane','bicycle','bird','boat','bottle',
            'bus','car' ,'cat','chair','cow', 
            'diningtable','dog','horse','motorbike','person', 
            'potted plant', 'sheep', 'sofa', 'train', 'monitor','unlabeld'
            ]

class DataTransforme(torch.nn.Module):
    """
    ## 画像とアノテーションの前処理クラス
    訓練時と検証時で異なる動作をする。

    - input_size: Int
        - リサイズ後の画像サイズ
    """
    # def __init__(self, input_size):
    #     super().__init__()
    #     self.to_tensor = transforms.ToTensor()
    #     self.resize  =transforms.Resize(size=(input_size, input_size))
    #     self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    # def forward(self, img, norm):
    #     img = np.array(img).astype(np.float32)
    #     img = self.to_tensor(img)
    #     img = self.resize(img)
    #     if norm==True:
    #         img = self.normalize(img)
    #     return img


    def __init__(self, input_size):

        super().__init__()
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.resize = transforms.Resize(size=(input_size, input_size))
    
    def forward(self, img):
        img_np = np.array(img).astype(np.float32)
        tensor = self.to_tensor(img_np)
        # if type=='input':
        #     tensor = self.normalize(tensor)
        resized_tensor = self.resize(tensor)
        return resized_tensor

class VOCDataset(Dataset):
    """
    ## VOC2012のDatasetを作成するクラス
    PyTorchのDatasetクラスを継承。

    - img_list: List
        - 画像のIDを格納したリスト
    - phase: 'train' or 'test'
    - transform: Object
        - 前処理クラスのインスタンス
    """
    def __init__(
            self, 
            img_list: list, 
            phase: str, 
            img_dir: str,
            gt_dir: str,
            augmentation=None,
            preprocessing=None,
    ):
        self.img_list = img_list
        self.img_dir = img_dir
        self.gt_dir = gt_dir

        self.phase = phase
        self.augmentation = augmentation 
        self.preprocessing = DataTransforme(input_size=256)
        self.Onehot = True

        self.img_fps = [os.path.join(img_dir, img_id)+".jpg" for img_id in self.img_list]
        self.gt_fps = [os.path.join(gt_dir, gt_id)+".png" for gt_id in self.img_list]

        # クラスに対応した数字配列を用意
        self.CLASSES = CLASSES
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in self.CLASSES]

    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, i):
        img = np.asarray(Image.open(self.img_fps[i]))
        gt = np.asarray(Image.open(self.gt_fps[i]))
        
        gt = np.where(gt == 255, len(self.CLASSES), gt)  # unlabeledのパレットインデックスを255番から最後番(今回は22番)に変更
        
        #onehotベクトルに変換
        gts = [(gt == v) for v in self.class_values]
        gt = np.stack(gts, axis=-1).astype('float')  # one-hotベクトルに変換

        img = self.preprocessing(img)
        gt = self.preprocessing(gt)

        return img, gt
