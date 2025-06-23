import os
import argparse
import matplotlib.pyplot as plt
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.metrics import precision_score, recall_score, f1_score
import albumentations as A
from albumentations.pytorch import ToTensorV2
from thop import profile, clever_format  # 导入 thop 库

from dataload.my_dataload import my_dataload
from models import find_model_using_name

from utils.lr_methods import warmup
from utils.train_engin import train_one_epoch, evaluate


# 模型FLOPs和Params输出
def compute_flops_params(model, input_tensor):
    flops, params = profile(model, inputs=(input_tensor,))
    flops, params = clever_format([flops, params], "%.3f")  # 格式化输出
    print(f"FLOPs: {flops}, Params: {params}")


parser = argparse.ArgumentParser()
parser.add_argument('--num_classes', type=int, default=5, help='分类数量')
parser.add_argument('--epochs', type=int, default=200, help='训练的epoch数量')
parser.add_argument('--batch_size', type=int, default=64, help='训练的batch size')
parser.add_argument('--lr', type=float, default=0.001, help='初始学习率')
parser.add_argument('--seed', default=True, action='store_true', help='固定初始化参数')
parser.add_argument('--tensorboard', default=True, action='store_true', help='使用tensorboard进行可视化')
parser.add_argument('--use_amp', default=False, action='store_true', help='使用混合精度训练')
parser.add_argument('--data_train_path', type=str, default="/home/jh/MobileVit/newdata_en")
parser.add_argument('--model', type=str, default="mobile_vit_adda_x_small", help='选择训练的模型')
parser.add_argument('--device', default='cuda', help='设备id (如: 0 或 0,1 或 cpu)')
parser.add_argument('--model-name', default='', help='创建模型名称')
parser.add_argument('--weights', type=str, default='', help='初始权重文件路径')









opt = parser.parse_args()


# Albumentations的图像增强封装
class AlbumentationsTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img):
        # 将PIL图像转换为numpy数组
        img = np.array(img)
        # 使用Albumentations的transform进行数据增强
        augmented = self.transform(image=img)
        # 返回增强后的图像
        return augmented['image']


if opt.seed:
    def seed_torch(seed=7):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        print('随机种子已固定')


    seed_torch()


def main(args):
    torch.autograd.set_detect_anomaly(True)  # 添加这一行来启用异常检测
    print(f"Using data path: {args.data_train_path}")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    log_dir = os.path.join("/home/jh/MobileVit/bast/login/coatnet_0", args.model_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    print(f"日志目录: {log_dir}")
    tb_writer = SummaryWriter(log_dir, comment='MobileVIT')
    print(f"TensorBoard writer 创建于 {log_dir}")

    albumentations_transform = {
        "train": A.Compose([
            A.Resize(224, 224),
            A.RandomResizedCrop((224, 224)),  # 将 (224, 224) 作为元组传入
            # A.HorizontalFlip(p=0.4),  # 水平翻转，40% 概率，适合刺绣图案的左右对称性
            # A.VerticalFlip(p=0.4),  # 垂直翻转，40% 概率，进一步增加多样性
            # A.RandomRotate90(p=0.4),  # 随机旋转90度，40% 概率，适合图像在不同角度下的情况
            # A.Rotate(limit=20, p=0.4),  # 随机旋转（±20度），减小旋转幅度，避免过多形变
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # 标准化
            ToTensorV2(),  # 转换为Tensor，方便后续输入模型
        ]),
        "val": A.Compose([
            A.Resize(224, 224),  # 调整尺寸
            # A.CenterCrop(224, 224),  # 中心裁剪
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # 标准化
            ToTensorV2(),  # 转换为Tensor
        ])
    }

    # 将 Albumentations 的transform传入到自定义的数据增强类中
    train_transform = AlbumentationsTransform(albumentations_transform["train"])
    val_transform = AlbumentationsTransform(albumentations_transform["val"])

    train_dataset = my_dataload(args.data_train_path, split='train', transform=train_transform)
    val_dataset = my_dataload(args.data_train_path, split='val', transform=val_transform)

    if args.num_classes != train_dataset.num_class:
        raise ValueError("数据集有 {} 个类别，但输入了 {}".format(train_dataset.num_class, args.num_classes))

    nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])
    print('每个进程使用 {} 个数据加载工作线程'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True,
                                               num_workers=nw, collate_fn=train_dataset.collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True,
                                             num_workers=nw, collate_fn=val_dataset.collate_fn)

    model = find_model_using_name(args.model, num_classes=args.num_classes).to(device)

    if args.weights != "":
        assert os.path.exists(args.weights), "权重文件: '{}' 不存在.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)
        model_keys = list(model.state_dict().keys())
        del_keys = [k for k in model_keys if k.startswith('classifier')]
        for k in del_keys:
            if k in weights_dict:
                del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))

    pg = [
        {"params": [p for n, p in model.named_parameters() if 'deform_conv' in n], "lr": args.lr * 0.1,
         "weight_decay": 0.001},
        {"params": [p for n, p in model.named_parameters() if 'deform_conv' not in n], "lr": args.lr,
         "weight_decay": 0.00001}
    ]
    optimizer = optim.AdamW(pg)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    best_acc = 0.
    best_metrics = {"precision": 0., "recall": 0., "f1": 0.}
    train_losses = []
    val_losses = []
    save_path = os.path.join(os.getcwd(), '//home/jh/MobileVit/bjst/weights/coatnet_0', args.model)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model=model, optimizer=optimizer, data_loader=train_loader,
                                                device=device, epoch=epoch, use_amp=args.use_amp, lr_method=warmup, )
        val_loss, val_acc = evaluate(model=model, data_loader=val_loader, device=device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print('[epoch %d] train_loss: %.3f  train_acc: %.3f  val_loss: %.3f  val_acc: %.3f' % (
            epoch + 1, train_loss, train_acc, val_loss, val_acc))

        if opt.tensorboard:
            tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
            tb_writer.add_scalar(tags[0], train_loss, epoch)
            tb_writer.add_scalar(tags[1], train_acc, epoch)
            tb_writer.add_scalar(tags[2], val_loss, epoch)
            tb_writer.add_scalar(tags[3], val_acc, epoch)
            tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        scheduler.step(val_loss)

        if val_acc > best_acc:
            best_acc = val_acc
            # 计算精度、召回率、F1分数
            y_true = []
            y_pred = []
            with torch.no_grad():
                for data, labels in val_loader:
                    data, labels = data.to(device), labels.to(device)
                    outputs = model(data)
                    _, preds = torch.max(outputs, 1)
                    y_true.extend(labels.cpu().numpy())
                    y_pred.extend(preds.cpu().numpy())

            precision = precision_score(y_true, y_pred, average='macro')
            recall = recall_score(y_true, y_pred, average='macro')
            f1 = f1_score(y_true, y_pred, average='macro')

            best_metrics = {"precision": precision, "recall": recall, "f1": f1}
            torch.save(model.state_dict(), os.path.join(save_path, "coatnet_0.pth"))
            print(f"最佳模型保存于 {os.path.join(save_path, 'coatnet_0.pth')}")

    tb_writer.close()

    print('最佳验证集: 准确率: {:.5f}, 精准率: {:.5f}, 召回率: {:.5f}, F1得分: {:.5f}'.format(
        best_acc, best_metrics["precision"], best_metrics["recall"], best_metrics["f1"]))
    with open(os.path.join(save_path, "kansformer_flower_log.txt"), 'a') as f:
        f.writelines(
            '最佳验证集: 准确率: {:.5f}, 精准率: {:.5f}, 召回率: {:.5f}, F1得分: {:.5f}\n'.format(
                best_acc, best_metrics["precision"], best_metrics["recall"], best_metrics["f1"]))

    plt.figure()
    plt.plot(range(1, args.epochs + 1), train_losses, label='train_loss')
    plt.plot(range(1, args.epochs + 1), val_losses, label='val_loss')
    plt.xlabel('Epoch #')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training/Validation Loss')
    plt.savefig(os.path.join(save_path, "training_validation_loss.png"))
    plt.show()


main(opt)
