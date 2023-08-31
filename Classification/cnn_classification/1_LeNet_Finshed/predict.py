import torch
print(torch.__version__)
import torchvision.transforms as transforms
from PIL import Image
from model import LeNet


def main():
    transform = transforms.Compose(
        [transforms.Resize((32, 32)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    net = LeNet()
    net.load_state_dict(torch.load('1_LeNet_Finshed/weight_LeNet.pth'))

    im = Image.open('1_LeNet_Finshed/4.jpg')
    im = transform(im)  # [C, H, W]
    im = torch.unsqueeze(im, dim=0)  # [N, C, H, W]

    # with torch.no_grad():
    #     outputs = net(im)
    #     predict = outputs.numpy()
    # print(predict)

    with torch.no_grad():
        outputs = net(im)
        print(outputs,"\n")

        predict = torch.max(outputs, dim=1)
        print(predict,"\n")

        predict = predict[1]
        print(predict)
        # 返回最大值的索引
        predict = predict.numpy()

    print(classes[int(predict)])
# torch.max(input, dim, max=None, max_indices=None) -> (Tensor, LongTensor) 返回输入张量给定维度上每行的最大值，并同时返回每个最大值的位置索引。


if __name__ == '__main__':
    main()
