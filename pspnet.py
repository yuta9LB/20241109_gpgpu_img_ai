import torch
from torch import nn
import torch.nn.functional as F

class ConvBatchNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, bias):
        super(ConvBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)
        self.batchnorm = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        return x
    
class ConvBatchNormRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, bias):
        super(ConvBatchNormRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        return x
    
class FeatureMap_Conv(nn.Module):
    def __init__(self):
        super(FeatureMap_Conv, self).__init__()
        self.cbnr1 = ConvBatchNormRelu(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1, dilation=1, bias=False)
        self.cbnr2 = ConvBatchNormRelu(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
        self.cbnr3 = ConvBatchNormRelu(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.cbnr1(x)
        x = self.cbnr2(x)
        x = self.cbnr3(x)
        outputs = self.maxpool(x)
        return outputs

class bottleNeckPSP(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, stride, dilation):
        super(bottleNeckPSP, self).__init__()
        self.cbnr1 = ConvBatchNormRelu(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)
        self.cbnr2 = ConvBatchNormRelu(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.cbnr3 = ConvBatchNormRelu(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)

        # スキップ結合
        self.cbn = ConvBatchNorm(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, dilation=1, bias=False)

        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x1 = self.cbnr1(x)
        x1 = self.cbnr2(x1)
        x1 = self.cbnr3(x1)
        x2 = self.cbn(x)
        return self.relu(x1 + x2)

class bottleNeckIdentifyPSP(nn.Module):
    def __init__(self, in_channels, mid_channels, stride, dilation):
        super(bottleNeckIdentifyPSP, self).__init__()

        self.cbnr1 = ConvBatchNormRelu(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)
        self.cbnr2 = ConvBatchNormRelu(mid_channels, mid_channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=False)
        self.cbn = ConvBatchNorm(mid_channels, in_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.cbnr1(x)
        x1 = self.cbnr2(x1)
        x1 = self.cbn(x1)
        return self.relu(x1 + x)

class PyramidPooling(nn.Module):
    def __init__(self, in_channels, pool_sizes, height, width):
        super(PyramidPooling, self).__init__()

        # forwardで使用する画像サイズ
        self.height = height
        self.width = width

        # 各畳み込み層の出力チャネル数
        out_channels = int(in_channels / len(pool_sizes))

        # 各畳み込み層を作成
        # この実装方法は愚直すぎてfor文で書きたいところですが、分かりやすさを優先しています
        # pool_sizes: [6, 3, 2, 1]
        self.avpool_1 = nn.AdaptiveAvgPool2d(output_size=pool_sizes[0])
        self.cbnr1 = ConvBatchNormRelu(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)

        self.avpool_2 = nn.AdaptiveAvgPool2d(output_size=pool_sizes[1])
        self.cbnr2 = ConvBatchNormRelu(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)

        self.avpool_3 = nn.AdaptiveAvgPool2d(output_size=pool_sizes[2])
        self.cbnr3 = ConvBatchNormRelu(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)

        self.avpool_4 = nn.AdaptiveAvgPool2d(output_size=pool_sizes[3])
        self.cbnr4 = ConvBatchNormRelu(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)

    def forward(self, x):
        x1 = self.cbnr1(self.avpool_1(x))
        x1 = F.interpolate(x1, size=(self.height, self.width), mode="bilinear", align_corners=True)

        x2 = self.cbnr2(self.avpool_2(x))
        x2 = F.interpolate(x2, size=(self.height, self.width), mode="bilinear", align_corners=True)

        x3 = self.cbnr3(self.avpool_3(x))
        x3 = F.interpolate(x3, size=(self.height, self.width), mode="bilinear", align_corners=True)

        x4 = self.cbnr4(self.avpool_4(x))
        x4 = F.interpolate(x4, size=(self.height, self.width), mode="bilinear", align_corners=True)

        # 最終的に結合させる、dim=1でチャネル数の次元で結合
        output = torch.cat([x, x1, x2, x3, x4], dim=1)

        return output

class ResidualBlockPSP(nn.Sequential):
    def __init__(self, n_blocks, in_channels, mid_channels, out_channels, stride, dilation):
        super(ResidualBlockPSP, self).__init__()

        # bottleNeckPSPの用意
        self.add_module("block1", bottleNeckPSP(in_channels, mid_channels, out_channels, stride, dilation))

        # bottleNeckIdentifyPSPの繰り返しの用意
        for i in range(n_blocks-1):
            self.add_module("block"+str(i+2), bottleNeckIdentifyPSP(out_channels, mid_channels, stride, dilation))

class DecodePSPFeature(nn.Module):
    def __init__(self, height, width, n_classes):
        super(DecodePSPFeature, self).__init__()

        # forwardで使用する画像サイズ
        self.height = height
        self.width = width

        self.cbnr = ConvBatchNormRelu(in_channels=4096, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
        self.dropout = nn.Dropout2d(p=0.1)
        self.classification = nn.Conv2d(in_channels=512, out_channels=n_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.cbnr(x)
        x = self.dropout(x)
        x = self.classification(x)
        output = F.interpolate(x, size=(self.height, self.width), mode="bilinear", align_corners=True)

        return output

class AuxiliaryPSPlayers(nn.Module):
    def __init__(self, in_channels, height, width, n_classes):
        super(AuxiliaryPSPlayers, self).__init__()

        # forwardで使用する画像サイズ
        self.height = height
        self.width = width

        self.cbr = ConvBatchNormRelu(in_channels=in_channels, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
        self.dropout = nn.Dropout2d(p=0.1)
        self.classification = nn.Conv2d(in_channels=256, out_channels=n_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.cbr(x)
        x = self.dropout(x)
        x = self.classification(x)
        output = F.interpolate(x, size=(self.height, self.width), mode="bilinear", align_corners=True)

        return output

class PSPNet(nn.Module):
    def __init__(self, n_classes):
        super(PSPNet, self).__init__()

        # パラメータ設定
        block_config = [3, 4, 6, 3]
        input_size = 256

        # 4つのモジュールを構成するサブネットワーク
        self.feature_conv = FeatureMap_Conv()
        self.feature_res1 = ResidualBlockPSP(n_blocks=block_config[0], in_channels=128, mid_channels=64, out_channels=256, stride=1, dilation=1)
        self.feature_res2 = ResidualBlockPSP(n_blocks=block_config[1], in_channels=256, mid_channels=128, out_channels=512, stride=2, dilation=1)
        self.feature_dilated_res1 = ResidualBlockPSP(n_blocks=block_config[2], in_channels=512, mid_channels=256, out_channels=1024, stride=1, dilation=2)
        self.feature_dilated_res2 = ResidualBlockPSP(n_blocks=block_config[3], in_channels=1024, mid_channels=512, out_channels=2048, stride=1, dilation=4)

        self.pyramid_pooling = PyramidPooling(in_channels=2048, pool_sizes=[6, 3, 2, 1], height=input_size//8, width=input_size//8)
        self.decode_feature = DecodePSPFeature(height=input_size, width=input_size, n_classes=n_classes)

        self.aux = AuxiliaryPSPlayers(in_channels=1024, height=input_size, width=input_size, n_classes=n_classes)

    def forward(self, x):
        x = self.feature_conv(x)
        x = self.feature_res1(x)
        x = self.feature_res2(x)
        x = self.feature_dilated_res1(x)

        output_aux = self.aux(x) # Featuere モジュールの途中をAuxモジュールへ

        x = self.feature_dilated_res2(x)

        x = self.pyramid_pooling(x)
        output = self.decode_feature(x)

        return output, output_aux