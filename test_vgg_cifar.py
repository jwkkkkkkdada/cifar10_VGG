import torch
from torchvision import transforms
from cifar10_vgg import VGG
from PIL import Image
import os
transfrom = transforms.Compose([transforms.Resize((32,32)),transforms.ToTensor()])
para = torch.load('E:\pycharm\pycharm project\pytorch research\model.pth')
device = torch.device('cuda:0')
Mymodel = VGG()
Mymodel.load_state_dict(para)
Mymodel.to(device)

def main():
    idx = [0,1,2,3,4]
    for idx in idx:
        img_bath = os.path.join('E:\pycharm\pycharm project\pytorch research','{}.jpeg'.format(idx))
        img = Image.open(img_bath)
        img = img.convert('RGB')
        img = transfrom(img)
        img = torch.reshape(img,(1,3,32,32)).to(device)
        Mymodel.eval()
        with torch.no_grad():
            pred = Mymodel(img).argmax(dim = 1)
            if pred == 0:
                print("飞机")
            elif pred == 1:
                print("汽车")
            elif pred == 2:
                print("鸟")
            elif pred == 3:
                print("猫")
            elif pred == 4:
                print("鹿")
            elif pred == 5:
                print("狗")
            elif pred == 6:
                print("青蛙")
            elif pred == 7:
                print("马")
            elif pred == 8:
                print("船")
            elif pred == 9:
                print("卡车")




if __name__ == '__main__':
    main()
