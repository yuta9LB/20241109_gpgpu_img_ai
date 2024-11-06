import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

# VOC2012で用いるラベル
CLASSES = ['backgrounds','aeroplane','bicycle','bird','boat','bottle',
            'bus','car' ,'cat','chair','cow', 
            'diningtable','dog','horse','motorbike','person', 
            'potted plant', 'sheep', 'sofa', 'train', 'monitor','unlabeld'
            ]

# カラーパレットの作成
COLOR_PALETTE = np.array(Image.open("./VOCdevkit/VOC2012/SegmentationClass/2007_000170.png").getpalette()).reshape(-1,3)
COLOR_PALETTE = COLOR_PALETTE.tolist()[:len(CLASSES)]
    
class DiceCrossEntropyLoss(nn.Module):
    def __init__(self, num_classes=22, weight=None, dice_weight=0.5):
        super(DiceCrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.weight = weight
        self.dice_weight = dice_weight

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.weight)
        # クロスエントロピー損失の計算
        ce_loss = ce_loss(inputs, targets.squeeze(1))  # targetsの次元を調整してCE計算
        
        # ダイス損失の計算
        dice_loss = self.dice_loss(inputs, targets.squeeze(1))

        # 複合損失の計算
        total_loss = (1 - self.dice_weight) * ce_loss + self.dice_weight * dice_loss
        return total_loss

    def dice_loss(self, inputs, targets):
        # ダイス損失の計算部分
        smooth = 1e-5  # ゼロ除算防止用
        inputs = torch.softmax(inputs, dim=1)  # クラスごとの確率に変換
        target_onehot = F.one_hot(targets, num_classes=inputs.shape[1]).permute(0, 3, 1, 2).float()  # one-hot表現に変換
        
        intersection = (inputs * target_onehot).sum(dim=(2, 3))
        union = inputs.sum(dim=(2, 3)) + target_onehot.sum(dim=(2, 3))
        
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return 1 - dice.mean()
