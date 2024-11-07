import os
import math
import tqdm
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader

from unet import UNet
from dataset import VOCDataset
from util import DiceCrossEntropyLoss

# VOC2012で用いるラベル
CLASSES = ['backgrounds','aeroplane','bicycle','bird','boat','bottle',
            'bus','car' ,'cat','chair','cow', 
            'diningtable','dog','horse','motorbike','person', 
            'potted plant', 'sheep', 'sofa', 'train', 'monitor','unlabeld'
            ]

# カラーパレットの作成
COLOR_PALETTE = np.array(Image.open("./VOCdevkit/VOC2012/SegmentationClass/2007_000170.png").getpalette()).reshape(-1,3)
COLOR_PALETTE = COLOR_PALETTE.tolist()[:len(CLASSES)]

def train(dataloader, model, optimizer, criterion, device):
    model.train()
    size = len(dataloader.dataset)
    batch_size = dataloader.batch_size
    step_total = math.ceil(size/batch_size)
    total_loss = 0.0

    with tqdm.tqdm(enumerate(dataloader), total=step_total) as pbar:
        for batch, item in pbar:
            inp, gt = item[0].to(device), item[1].to(device)

            # 損失誤差を計算
            pred = model(inp)

            loss = criterion(pred, gt, device)
            total_loss += loss.item()

            # バックプロパゲーション
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # プログレスバーに損失を表示
            pbar.set_postfix(loss=loss.item())

        avg_loss = total_loss / step_total
        print(f"Train Loss: {avg_loss:.4f}")
    
    return avg_loss

def test(dataloader, model, criterion, device):
    model.eval()
    size = len(dataloader.dataset)
    batch_size = dataloader.batch_size
    step_total = math.ceil(size/batch_size)
    total_loss = 0.0

    with torch.no_grad():
        for item in dataloader:
            inp, gt = item[0].to(device), item[1].to(device)
            pred = model(inp)
            loss = criterion(pred, gt, device)
            total_loss += loss.item()

        avg_loss = total_loss / step_total
        print(f"Val Loss: {avg_loss:.4f}")

    return avg_loss

def main():
    save_dir = './savedir'
    save_name_prefix = 'savename'
    csv_path = os.path.join(save_dir, f'{save_name_prefix}.csv')

    initial_epoch = 0
    chkp_path = None #'weights/unet_241106_00.pth'
    continuation = False # 続きから再開する場合True

    train_list_path = 'VOCdevkit/VOC2012_sample/listfile/train_list_1464.txt'
    val_list_path = 'VOCdevkit/VOC2012_sample/listfile/val_list_1449.txt'

    img_dir = 'VOCdevkit/VOC2012_sample/JPEGImages'
    gt_dir = 'VOCdevkit/VOC2012_sample/SegmentationClass'

    with open(train_list_path, 'r') as f:
        train_list = f.read().splitlines()
    with open(val_list_path, 'r') as g:
        val_list = g.read().splitlines()

    train_ds = VOCDataset(img_list=train_list, img_dir=img_dir, gt_dir=gt_dir, data_augmentation=True)
    val_ds = VOCDataset(img_list=val_list, img_dir=img_dir, gt_dir=gt_dir)
    print(f"len(train_data): {train_ds.__len__()}")
    print(f"len(test_data): {val_ds.__len__()}")

    train_dl = DataLoader(train_ds, batch_size=16, num_workers=8, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=16, num_workers=8, shuffle=True)

    epochs = 100
    device = 'cuda' if torch.cuda.is_available()  else "cpu"
    print("Using {} device".format(device))

    model = UNet()
    model = model.to(device)

    # 重みの読み込み。
    if chkp_path is not None:
        checkpoint = torch.load(chkp_path, map_location='cpu', weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if continuation == True:
            initial_epoch = checkpoint["epoch"] + 1
    
    weight = None # torch.tensor([0.3, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5])
    if weight is not None:
        weight = weight.to(device)
    criterion =  DiceCrossEntropyLoss(weight=weight, dice_weight=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for epoch in range(initial_epoch, epochs):
        print(f"Epoch {epoch}\n-------------------------------")
        train_loss = train(train_dl, model, optimizer, criterion, device)
        val_loss = test(val_dl, model, criterion, device)

        torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            },
        os.path.join(save_dir, f'{save_name_prefix}_ep{epoch}.pth'),
        )
        
        if not os.path.exists(csv_path):
            with open(csv_path, 'w') as f:
                f.write('epoch,train_loss,val_loss\n')
        with open(csv_path, 'a') as f:
            f.write(f'{epoch},{train_loss},{val_loss}\n')

if __name__ == '__main__':
    main()