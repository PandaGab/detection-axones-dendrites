import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from deeplib.trainingq2 import train, validate
import math
from torch.autograd import Variable
from dataset import datasetDetection
import tifTransforms as tifT
from torch.utils.data import DataLoader
from utils import initialize_weights


# UNet desribe in pix2pix https://arxiv.org/abs/1611.07004
# à tester
class UNet(nn.Module):
    def __init__(self, inC, outC, nbf=64):
        super(UNet, self).__init__()
        
        self.down1 = nn.Conv2d(inC    , nbf    , 4, stride=2, padding=1)
        self.down2 = nn.Conv2d(nbf    , nbf * 2, 4, stride=2, padding=1, bias=False)
        self.d_bn2 = nn.BatchNorm2d(nbf * 2)
        self.down3 = nn.Conv2d(nbf * 2, nbf * 4, 4, stride=2, padding=1, bias=False)
        self.d_bn3 = nn.BatchNorm2d(nbf * 4)
        self.down4 = nn.Conv2d(nbf * 4, nbf * 8, 4, stride=2, padding=1, bias=False)
        self.d_bn4 = nn.BatchNorm2d(nbf * 8)
        self.down5 = nn.Conv2d(nbf * 8, nbf * 8, 4, stride=2, padding=1, bias=False)
        self.d_bn5 = nn.BatchNorm2d(nbf * 8)
        self.down6 = nn.Conv2d(nbf * 8, nbf * 8, 4, stride=2, padding=1, bias=False)
        self.d_bn6 = nn.BatchNorm2d(nbf * 8)
        self.down7 = nn.Conv2d(nbf * 8, nbf * 8, 4, stride=2, padding=1, bias=False)
        self.d_bn7 = nn.BatchNorm2d(nbf * 8)
        self.down8 = nn.Conv2d(nbf * 8, nbf * 8, 4, stride=2, padding=1, bias=False)
        self.d_bn8 = nn.BatchNorm2d(nbf * 8)
        self.up8 = nn.ConvTranspose2d(nbf * 8, nbf * 8, 4, stride=2, padding=1, bias=False)
        self.u_bn8 = nn.BatchNorm2d(nbf * 8)
        self.up7 = nn.ConvTranspose2d(nbf * 16, nbf * 8, 4, stride=2, padding=1, bias=False)
        self.u_bn7 = nn.BatchNorm2d(nbf * 8)
        self.up6 = nn.ConvTranspose2d(nbf * 16, nbf * 8, 4, stride=2, padding=1, bias=False)
        self.u_bn6 = nn.BatchNorm2d(nbf * 8)
        self.up5 = nn.ConvTranspose2d(nbf * 16, nbf * 8, 4, stride=2, padding=1, bias=False)
        self.u_bn5 = nn.BatchNorm2d(nbf * 8)
        self.up4 = nn.ConvTranspose2d(nbf * 16, nbf * 4, 4, stride=2, padding=1, bias=False)
        self.u_bn4 = nn.BatchNorm2d(nbf * 4)
        self.up3 = nn.ConvTranspose2d(nbf * 8, nbf * 2, 4, stride=2, padding=1, bias=False)
        self.u_bn3 = nn.BatchNorm2d(nbf * 2)
        self.up2 = nn.ConvTranspose2d(nbf * 4, nbf, 4, stride=2, padding=1, bias=False)
        self.u_bn2 = nn.BatchNorm2d(nbf)
        self.up1 = nn.ConvTranspose2d(nbf * 2, outC, 4, stride=2, padding=1)
        self.u7_drop = nn.Dropout2d()
        self.u6_drop = nn.Dropout2d()
        self.u5_drop = nn.Dropout2d()

        
        self.downReLU = nn.LeakyReLU(0.2, True)
        self.upReLU = nn.ReLU(True)
        
    
    def forward(self, x): # 1 channels
        d1 = self.downReLU(self.down1(x))  # 64  channels
        d2 = self.downReLU(self.d_bn2(self.down2(d1))) # 128 channels
        d3 = self.downReLU(self.d_bn3(self.down3(d2))) # 256 channels
        d4 = self.downReLU(self.d_bn4(self.down4(d3))) # 512 channels
        d5 = self.downReLU(self.d_bn5(self.down5(d4))) # 512 channels
        d6 = self.downReLU(self.d_bn6(self.down6(d5))) # 512 channels
        d7 = self.downReLU(self.d_bn7(self.down7(d6))) # 512 channels
        b8 = self.downReLU(self.d_bn8(self.down8(d7))) # 512 channels bottleneck
        u7 = self.u7_drop(self.upReLU(self.u_bn8(self.up8(b8))))  # 512 channels
        u6 = self.u6_drop(self.upReLU(self.u_bn7(self.up7(torch.cat([d7, u7], 1)))))
        u5 = self.u5_drop(self.upReLU(self.u_bn6(self.up6(torch.cat([d6, u6], 1)))))
        u4 = self.upReLU(self.u_bn5(self.up5(torch.cat([d5, u5], 1))))
        u3 = self.upReLU(self.u_bn4(self.up4(torch.cat([d4, u4], 1))))
        u2 = self.upReLU(self.u_bn3(self.up3(torch.cat([d3, u3], 1))))
        u1 = self.upReLU(self.u_bn2(self.up2(torch.cat([d2, u2], 1))))
        out = self.up1(torch.cat([d1, u1], 1))
        
        return out

class MyUNet(nn.Module):

    def __init__(self, n_layers, hidden_size=100):
        super().__init__()

        self.first_layer = nn.Linear(32 * 32 * 3, hidden_size)
        self.first_layer.weight.data.normal_(0.0, math.sqrt(2 / 32 * 32 * 3))
        self.first_layer.bias.data.fill_(0.005)

        self.layers = []
        for i in range(n_layers - 1):
            layer = nn.Linear(hidden_size, hidden_size)
            layer.weight.data.normal_(0.0, math.sqrt(2 / hidden_size))
            layer.bias.data.fill_(0.005)
            self.layers.append(layer)
            self.add_module('layer-%d' % i, layer)

        self.output_layer = nn.Linear(hidden_size, 10)
        self.output_layer.weight.data.normal_(0.0, math.sqrt(2 / hidden_size))
        self.output_layer.bias.data.fill_(0.005)

    def forward(self, x):
        x = x.view(-1, 32 * 32 * 3)
        out = F.relu(self.first_layer(x))
        for i, l in enumerate(self.layers):
            out = l.forward(out)
            out = F.relu(out)

            if i < len(self.layers) - 1:
                out = self.dropout(out, self.drop_probability, training=self.training)
        return self.output_layer.forward(out)


if __name__ == "__main__":
    csvFilePath = "/home/nani/Documents/data/2017-11-14 EXP 201b Drugs/transcriptionTable.txt"
    mean = [32772.82847326139]
    std = [8.01126226921115]
    transformations = transforms.Compose([tifT.RandomCrop(256),
                                          tifT.RandomHorizontalFlip(),                                         
#                                          tifT.Pad(100,mode='constant', constant_values=mean[0]),
                                          tifT.RandomVerticalFlip(),
                                          tifT.ToTensor(),
                                          tifT.Normalize(mean=mean,
                                                         std=std)])
    
    dataset = datasetDetection(csvFilePath, 
                               transforms=transformations)
    
    dataloader = DataLoader(dataset, batch_size=2,shuffle=False)
    for actine , mask in dataloader:
        actine = actine.type(torch.FloatTensor)
#        img = (actine[0,0].numpy(),mask[0,0].numpy(),mask[0,1].numpy(), mask[0,2].numpy())
#        plot(img)
#        img = (actine[1,0].numpy(),mask[1,0].numpy(),mask[1,1].numpy(),mask[1,2].numpy())
#        plot(img)
        break
    
    actine = Variable(actine, requires_grad=False) 
    unet = UNet(1, 3)
    unet.apply(initialize_weights)
    unet.eval()
    out = unet(actine)
    


    
    
    
#    net = MyUNet(3)
#    net.cpu()
#    #dataset_train = datasets.CIFAR10('../CIFAR10_data', train=True, download=True)
#    # optimizer = optim.SGD(net.parameters(), lr=0.01, nesterov=True, momentum=0.9)
#    optimizer = torch.optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
#    history = train(net, optimizer, dataset_train, 10, batch_size=32)
#
#    torch.save(net, './trainingModel.pt')
#    # the_model = torch.load('./trainingModel.pt')
#    #dataset_test = datasets.CIFAR10('../CIFAR10_data', train=False, download=True, transform=transforms.ToTensor())
#    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=32)
#    score, loss = validate(net, test_loader, use_gpu=False)
#    print(score)