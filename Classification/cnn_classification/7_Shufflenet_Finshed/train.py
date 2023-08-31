import os
import math
import argparse

import sys
import json
import pickle
import random
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler
from model import shufflenet_v2_x1_0
from my_dataset import MyDataSet
from utils import train_one_epoch, evaluate


def read_split_data(root: str, val_rate: float = 0.2):
    # 保证随机结果可复现
    random.seed(0)
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # 遍历文件夹，一个文件夹对应一个类别
    # os.path.isdir()用于判断某一对象(需提供绝对路径)是否为目录
    # os.listdir()返回一个列表，其中包含有指定路径下的目录和文件的名称
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]

    # 排序，保证各平台顺序一致
    flower_class.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        # 遍历获取supported支持的所有文件路径
        # os.path.splitext(“文件路径”)   分离文件名与扩展名
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # 排序，保证各平台顺序一致
        images.sort()
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))
        # 按比例随机采样验证样本
        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:  # 如果该路径在采样的验证集样本中则存入验证集
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:  # 否则存入训练集
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))
    assert len(train_images_path) > 0, "number of training images must greater than 0."
    assert len(val_images_path) > 0, "number of validation images must greater than 0."

    plot_image = False
    if plot_image:
        # 绘制每种类别个数柱状图
        plt.bar(range(len(flower_class)), every_class_num, align='center')
        # 将横坐标0,1,2,3,4替换为相应的类别名称
        plt.xticks(range(len(flower_class)), flower_class)
        # 在柱状图上添加数值标签
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        # 设置x坐标
        plt.xlabel('image class')
        # 设置y坐标
        plt.ylabel('number of images')
        # 设置柱状图的标题
        plt.title('flower class distribution')
        plt.show()

    return train_images_path, train_images_label, val_images_path, val_images_label

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(args)

    """
        args = {namespace}
        1.命名空间指的是变量存储的位置，每一个变量都需要存储到指定的命名空间当中
        2.每一个作用域都会有一个它对应的命名空间
        3.全局命名空间，用来保存全局变量。函数命名空间用来保存函数中的变量
        4.命名空间实际上就是一个字典，是一个专门用来存储变量的字典
    """
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    tb_writer = SummaryWriter(log_dir="runs")

# 创建weights文件夹
    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

# 实例化训练数据集
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

# 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    batch_size = args.batch_size
    nw = 0  # min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)
    """
    pin_memory是dataloader()的参数，默认值为False，其作用是是否把把数据存放在锁页内存中。
    主机的内存根据物理内存（内存条）与虚拟内存（硬盘）进行数据交换分为锁页内存和不锁页内存：
    锁页内存：数据存放在物理内存上（内存条）上；
    不锁页内存：当物理内存（内存条）满载时，把部分数据转换到虚拟内存上（硬盘）上。
    锁页内存（pin_memory)能够保持与GPU进行高速传输，在训练时加快数据的读取，从而加快训练速度。
    因此，如果主机/服务器的内存足够大，建议把pin_memory设为True
    """
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

# 如果存在预训练权重则载入
    model = shufflenet_v2_x1_0(num_classes=args.num_classes).to(device)

    # 如果预训练权重参数不是“”
    if args.weights != "":
        # 且如果预训练权重路径存在
        if os.path.exists(args.weights):
            # 则加载进来预训练权重 weights_dict
            weights_dict = torch.load(args.weights, map_location=device)
            # 将最后两两个线性层参数weight和bias置空  取出其它参数 load_weights_dict
            load_weights_dict = {k: v for k, v in weights_dict.items()
                                 if model.state_dict()[k].numel() == v.numel()}
            # 将处理好的预训练权重装入模型  当strict=True,要求预训练权重层数的键值与新构建的模型中的权重层数名称完全吻合；
            print(model.load_state_dict(load_weights_dict, strict=False))
        else:
            raise FileNotFoundError("not found weights file: {}".format(args.weights))

# 是否冻结权重
    # .named_parameters()生成器
    model_name_para = model.named_parameters()
    # model.named_parameters()，迭代打印model.named_parameters()将会打印每一次迭代元素的名字和param
    # model.parameters()，迭代打印model.parameters()将会打印每一次迭代元素的param而不会打印名字
    # 这是他和named_parameters的区别，两者都可以用来改变requires_grad的属性

    # 如果超参数--freeze_layers 为True
    if args.freeze_layers:
        for name, para in model_name_para:
            # 除最后的全连接层外，其他权重全部冻结
            # 如果“fc" 不在权重层name里 则不更新权重
            if "fc" not in name:
                para.requires_grad_(False)

    # 遍历处理过的para 如果p.requires_grad是True 则将当前遍历para放入pg[list]
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=4E-5)

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    """
    调整优化器学习率
    lr_scheduler.LambdaLR ():
    optimizer：被调整学习率的优化器
    lr_lambda：用户自定义的学习率调整规则。可以是lambda表达式，也可以是函数
    last_epoch：当前优化器的已迭代次数，后文我们将其称为epoch计数器。
                默认是-1，字面意思是第-1个epoch已完成，也就是当前epoch从0算起，从头开始训练。
                如果是加载checkpoint继续训练，那么这里要传入对应的已迭代次数
    verbose：是否在更新学习率时在控制台输出提醒
    """
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

#开始训练
    for epoch in range(args.epochs):
        # train
        mean_loss = train_one_epoch(model=model,
                                    optimizer=optimizer,
                                    data_loader=train_loader,
                                    device=device,
                                    epoch=epoch)
        # 一个epoch以后更新学习率
        scheduler.step()

        # validate
        acc = evaluate(model=model,
                       data_loader=val_loader,
                       device=device)

        print("[epoch {}] accuracy: {}".format(epoch, round(acc, 3))) # round() 返回四舍五入值（数值，保留几位）

        # tensorboard画图
        tags = ["loss", "accuracy", "learning_rate"] # 标题
        tb_writer.add_scalar(tags[0], mean_loss, epoch) # (标题，y值，x值)
        tb_writer.add_scalar(tags[1], acc, epoch)
        tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)

        # 保存权重 每一个epoch保存一次
        torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lrf', type=float, default=0.1)
    parser.add_argument('--data-path', type=str,
                        default="D:\Lzj_learning\Pycharm Program\PythonProject\Lzj_program\data_set\data_flower\images")
    # https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth
    parser.add_argument('--weights', type=str, default='./shufflenetv2_x1.pth',
                        help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0')

    opt = parser.parse_args()
    main(opt)
