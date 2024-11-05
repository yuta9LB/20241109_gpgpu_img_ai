import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

# VOC2012で用いるラベル
CLASSES = ['backgrounds','aeroplane','bicycle','bird','boat','bottle',
            'bus','car' ,'cat','chair','cow', 
            'diningtable','dog','horse','motorbike','person', 
            'potted plant', 'sheep', 'sofa', 'train', 'monitor','unlabeld'
            ]

# カラーパレットの作成
COLOR_PALETTE = np.array(Image.open("./VOCdevkit/VOC2012/SegmentationClass/2007_000170.png").getpalette()).reshape(-1,3)
COLOR_PALETTE = COLOR_PALETTE.tolist()[:len(CLASSES)]

def segmentation2gt(gt, device):
    """
    One-hotベクトルからRGB画像に変換（バッチ対応）。
    Args:
        pred (torch.Tensor): 形状が (B, C, H, W) の予測テンソル
        color_palette (list): 各クラスのRGB値リスト
    Returns:
        rgb_images (torch.Tensor): 形状が (B, H, W, 3) のRGB画像
    """
    # onehotベクトルからカラーインデックスを取り出して、該当するRGB3次元を付与
    # gt.shape = (H,W,label_onehot)

    tmp_gt = np.asarray(gt)
    H,W = gt.shape[:2]
    img = [[0 for j in range(W)] for i in range(H)]

    for height in range(H):
        for width in range(W):
            index = np.argmax(gt[height,width,:])
            rgb = COLOR_PALETTE[index]
            img[height][width] = rgb
    img = np.asarray(img)
    print(img.shape)
    return img

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)  # ロジットをシグモイドで確率化
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        return 1 - dice

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(CombinedLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()  # BCE損失（ロジットを受け取るバージョン）
        self.dice_loss = DiceLoss()  # Dice損失
        self.alpha = alpha  # BCEとDiceのバランスを取るパラメータ

    def forward(self, pred, target):
        bce = self.bce_loss(pred, target)  # BCE誤差
        dice = self.dice_loss(pred, target)  # Dice誤差
        combined = self.alpha * bce + (1 - self.alpha) * dice  # 総合的な損失
        return combined

class BCE_and_Tversky_AveLoss():
    def __init__(self):
        self.TverskyLoss = smp.losses.TverskyLoss(mode='multilabel', log_loss=False)
        self.BCELoss = smp.losses.SoftBCEWithLogitsLoss()

    def __call__(self, pred, target):
        return 0.5*self.BCELoss(pred, target) + 0.5*self.TverskyLoss(pred, target)
    
class DiceCrossEntropyLoss(nn.Module):
    def __init__(self, num_classes=22, weight=None, dice_weight=0.5):
        super(DiceCrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.weight = weight
        self.dice_weight = dice_weight
        self.ce_loss = nn.CrossEntropyLoss(weight=self.weight)

    def forward(self, inputs, targets):
        # クロスエントロピー損失の計算
        ce_loss = self.ce_loss(inputs, targets.argmax(dim=1))  # targetsの次元を調整してCE計算
        
        # ダイス損失の計算
        dice_loss = self.dice_loss(inputs, targets)

        # 複合損失の計算
        total_loss = (1 - self.dice_weight) * ce_loss + self.dice_weight * dice_loss
        return total_loss

    def dice_loss(self, inputs, targets):
        # ダイス損失の計算部分
        smooth = 1e-5  # ゼロ除算防止用
        inputs = F.softmax(inputs, dim=1)  # 確率分布に変換済みのinputs

        # ダイススコアの計算
        intersection = torch.sum(inputs * targets, dim=(2, 3))  # 対応ピクセルの積和
        union = torch.sum(inputs, dim=(2, 3)) + torch.sum(targets, dim=(2, 3))  # 全体の和
        dice_score = (2. * intersection + smooth) / (union + smooth)
        
        # ダイス損失の計算
        dice_loss = 1 - dice_score.mean()
        return dice_loss

