import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import json
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from model import vgg


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # load image
    img_path = "./2.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)
    
    # create model
    model = vgg(model_name="vgg16", num_classes=5).to(device)

    # load model weights
    weights_path = "./vgg16Net.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)

    """
        模型保存有两种形式: 1、是保存模型的state_dict()，只是保存模型的参数。加载时需要先创建一个模型的实例model，之后通过torch.load()将保存
                        的模型参数加载进来，得到dict，再通过model.load_state_dict(dict)将模型的参数更新。
                        解释：https://blog.csdn.net/qq_52852138/article/details/123339337?spm=1001.2014.3001.5502
                        2、是将整个模型保存下来，之后加载的时候只需要通过torch.load()将模型加载，即可返回一个加载好的模型。
        map_location: 用于重定向，如此前模型的参数是在cpu中，我们将其加载到cuda:0中。或者有多张卡，可以将卡1中训练好的模型加载到卡2
                        解释：https://blog.csdn.net/qq_43219379/article/details/123675375   
    """
    model.load_state_dict(torch.load(weights_path, map_location=device))


    """
        model train（）、 model eval（）
        解释：https://zhuanlan.zhihu.com/p/357075502
    """
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    plt.title(print_res)
    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                  predict[i].numpy()))
    plt.show()


if __name__ == '__main__':
    main()
