import os
from shutil import copyfile
import numpy as np

from torchvision.datasets import ImageFolder
from torchvision import transforms as tf
from torchvision.models import resnet18

from sklearn.metrics import accuracy_score

import torch
from torch.autograd import Variable



def make_dir(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)

"""
Cette fonction sépare les images de CUB200 en un jeu d'entraînement et de test.

dataset_path: Path où se trouve les images de CUB200
train_path: path où sauvegarder le jeu d'entraînement
test_path: path où sauvegarder le jeu de test
"""


def separate_train_test(dataset_path, train_path, test_path):

    class_index = 1
    for classname in sorted(os.listdir(dataset_path)):
        if classname.startswith('.'):
            continue
        make_dir(os.path.join(train_path, classname))
        make_dir(os.path.join(test_path, classname))
        i = 0
        for file in sorted(os.listdir(os.path.join(dataset_path, classname))):
            if file.startswith('.'):
                continue
            file_path = os.path.join(dataset_path, classname, file)
            if i < 15:
                copyfile(file_path, os.path.join(test_path, classname, file))
            else:
                copyfile(file_path, os.path.join(train_path, classname, file))
            i += 1

        class_index += 1
        
def validate(model, val_loader, use_gpu=True):
    model.train(False)
    true = []
    pred = []
    val_loss = []

    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    for j, batch in enumerate(val_loader):

        inputs, targets = batch
        if use_gpu:
            inputs = inputs.cuda()
            targets = targets.cuda()

        inputs = Variable(inputs, volatile=True)
        targets = Variable(targets, volatile=True)
        output = model(inputs)

        predictions = output.max(dim=1)[1]

        val_loss.append(criterion(output, targets).data[0])
        true.extend(targets.data.cpu().numpy().tolist())
        pred.extend(predictions.data.cpu().numpy().tolist())
    model.train(True)
    return accuracy_score(true, pred) * 100, sum(val_loss) / len(val_loss)


def create_resnet18_and_sets(question, normalize, trainP, testP, mean, std):
    
    rgbMeanImageNet = (0.485, 0.456, 0.406)
    rgbStdImageNet = (0.229, 0.224, 0.225)
    
    if normalize:
        transformations = tf.Compose([tf.Resize((224,224)), 
                              tf.ToTensor(), 
                              tf.Normalize(mean=mean, std=std)])
    else:
        transformations = tf.Compose([tf.Resize((224,224)), 
                                      tf.ToTensor(),
                                      tf.Normalize(mean=rgbMeanImageNet, std=rgbStdImageNet)])

    train_set = ImageFolder(trainP, transform=transformations)
    test_set = ImageFolder(testP, transform=transformations)
    
    if question=='a':
        model = resnet18(pretrained=False)
        model.fc = torch.nn.Linear(512, 200)
    elif question=='b':
        model = resnet18(pretrained=True)
        model.fc = torch.nn.Linear(512, 200)
        for name, param in model.named_parameters():
            if not name[0:2]=='fc':
                param.requires_grad = False
    elif question=='c':
        model = resnet18(pretrained=True)
        model.fc = torch.nn.Linear(512,200)
        for name, param in model.named_parameters():
            if name[0:5]=='conv1' or name[0:3]=='bn1' or name[0:6]=='layer1':
                param.requires_grad = False
    elif question=='d':
        model = resnet18(pretrained=True)
        model.fc = torch.nn.Linear(512, 200)
    
    return train_set, test_set, model

def compute_mean_std(train_set):
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=32)
    nb = 0
    Rsum = 0
    Bsum = 0
    Gsum = 0
    for batch in train_loader:
        nb += len(batch[1])
        Rsum += torch.sum(batch[0][:,0,:,:])
        Bsum += torch.sum(batch[0][:,1,:,:])
        Gsum += torch.sum(batch[0][:,2,:,:])
    nb = nb * 224 * 224
    Rmean = Rsum / nb
    Bmean = Bsum / nb
    Gmean = Gsum / nb
    
    RstdSum = 0
    BstdSum = 0
    GstdSum = 0
    for batch in train_loader:
        RstdSum += torch.sum(torch.pow(batch[0][:,0,:,:] - Rmean,2))
        BstdSum += torch.sum(torch.pow(batch[0][:,1,:,:] - Bmean,2))
        GstdSum += torch.sum(torch.pow(batch[0][:,2,:,:] - Gmean,2))
    Rstd = np.sqrt(RstdSum / nb)
    Bstd = np.sqrt(BstdSum / nb)
    Gstd = np.sqrt(GstdSum / nb)
    
    RGBmean = (Rmean, Gmean, Bmean)
    RGBstd = (Rstd, Gstd, Bstd)
    return RGBmean, RGBstd
    

if __name__ == '__main__':
    dataset_path = './images'
    train_path = './q3Train'
    test_path = './q3Test'
    
    if not os.path.exists(train_path):
        separate_train_test(dataset_path, train_path, test_path)
    
    question = 'b'
    normalize = False
    
    batch_size = 32
    n_epoch = 10
    use_gpu = False
    
    transformations = tf.Compose([tf.Resize((224,224)), tf.ToTensor()])
    train_set = ImageFolder(train_path, transform=transformations)
    
    RGBmean, RGBstd = compute_mean_std(train_set)
    
    train_set, test_set, model = create_resnet18_and_sets(question, normalize,
                                                           train_path, test_path,
                                                           RGBmean, RGBstd)
    train_param = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(train_param)
    
    train_loader = torch.utils.data.DataLoader(train_set,batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set,batch_size)
    
    criterion = torch.nn.CrossEntropyLoss()
    
    if use_gpu:
        model.cuda()
        
    model.train()
    b = 0
    for i in range(n_epoch):
        print('epoch :',i)
        for batch in train_loader:
            b += 1
            inputs, targets = batch
            if use_gpu:
                inputs = inputs.cuda()
                targets = targets.cuda()
    
            inputs = Variable(inputs)
            targets = Variable(targets)
            
            optimizer.zero_grad()
            
            output = model(inputs)
            
            loss = criterion(output, targets)
            loss.backward()
            
            optimizer.step()
    
        score, loss = validate(model, test_loader, use_gpu=use_gpu)
        print('test acc:',score)



        




