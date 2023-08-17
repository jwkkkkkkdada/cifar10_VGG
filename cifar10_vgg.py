import torch
from torch import nn,optim
from torch.utils.data import DataLoader
from torchvision import datasets,transforms

class VGG(torch.nn.Module):
    def __init__(self):
        super(VGG,self).__init__()
        self.block1 = nn.Sequential(nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,stride=1,padding=1),
                                    nn.BatchNorm2d(num_features=64),nn.ReLU(),
                                    nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1),
                                    nn.BatchNorm2d(num_features=64),nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2,stride=2))

        self.block2 = nn.Sequential(nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1),
                                    nn.BatchNorm2d(num_features=128),nn.ReLU(),
                                    nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1),
                                    nn.BatchNorm2d(128),nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2,stride=2))

        self.block3 = nn.Sequential(nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=1),
                                    nn.BatchNorm2d(256),nn.ReLU(),
                                    nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1),
                                    nn.BatchNorm2d(256),nn.ReLU(),
                                    nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1),
                                    nn.BatchNorm2d(256),nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2,stride=2))

        self.block4 = nn.Sequential(nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,stride=1,padding=1),
                                    nn.BatchNorm2d(512),nn.ReLU(),
                                    nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1),
                                    nn.BatchNorm2d(512),nn.ReLU(),
                                    nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1),
                                    nn.BatchNorm2d(512),nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2,stride=2))

        self.block5 = nn.Sequential(nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1),
                                    nn.BatchNorm2d(512),nn.ReLU(),
                                    nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1),
                                    nn.BatchNorm2d(512),nn.ReLU(),
                                    nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1),
                                    nn.BatchNorm2d(512),nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2,stride=2))

        self.fc1 = nn.Sequential(nn.Flatten(),nn.Linear(in_features=512,out_features=512),nn.ReLU(),nn.Dropout())
        self.fc2 = nn.Sequential(nn.Linear(512,256),nn.ReLU(),nn.Dropout())
        self.fc3 = nn.Linear(256,10)

    def forward(self,input):
        x = self.block1(input)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.fc1(x)
        x = self.fc2(x)
        out = self.fc3(x)

        return out




def main():
    batch_size = 32
    cifar_train = datasets.CIFAR10('cifar',True,transform=transforms.Compose(
        [transforms.Resize((32,32)),transforms.ToTensor()]),download=True)

    #按照batch_size载入数据集
    cifar_train = DataLoader(cifar_train,batch_size = batch_size,shuffle=True)

    cifar_test = datasets.CIFAR10('cifar',False,transform=transforms.Compose(
        [transforms.Resize((32,32)),transforms.ToTensor()]),download=True)
    cifar_test = DataLoader(cifar_test,batch_size = batch_size)

    #input_x,label = iter(cifar_train).next()
    #print('x:',input_x.shape,'label:',label.shape)
    #在GPU上进行运算
    device = torch.device("cuda:0")
    network = VGG().to(device)
    criteon = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(network.parameters(),lr=1e-3)

    for epoch in range(50):
        network.train()
        for batchidx,(input_x,label) in enumerate(cifar_train):
            input_x,label = input_x.to(device),label.to(device)
            #logits:[b,10]
            #label:[b]
            logits = network(input_x)

            #loss,tensorscaler
            loss = criteon(logits,label)


            #梯度清0
            optimizer.zero_grad()
            #反向传播
            loss.backward()
            #梯度更新
            optimizer.step()

        print('epoch:',epoch,loss.item())

        network.eval()
        with torch.no_grad():
             #test
            total_correct = 0
            total_num = 0
            for x,label in cifar_test:
                x, label = x.to(device),label.to(device)
                #[b,10]
                logits = network(x)

                pred = logits.argmax(dim = 1)
                total_correct += torch.eq(pred,label).float().sum().item()
                total_num += x.size(0)
            acc = total_correct / total_num
            print(epoch,acc)

    torch.save(network.state_dict(),'E:\pycharm\pycharm project\pytorch research\model.pth')












if __name__ == '__main__':
    main()


