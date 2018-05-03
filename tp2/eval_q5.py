import sys
import torch
import numpy as np
from torchvision import models
from torch.autograd import Variable
import torchvision
from torchvision.datasets import ImageFolder
from utils import plot_confusion_matrix

if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise RuntimeError('You must provide the path to the dataset as an argument')

    dataset_path = sys.argv[1]
    normalize = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    test_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224,224)),
            torchvision.transforms.ToTensor(),
            normalize])
    test_dataset = ImageFolder(dataset_path,test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset)


    model1 = models.resnet18(pretrained=False)
    in_features = model1.fc.in_features
    model1.fc = torch.nn.Linear(in_features,5)
    model2 = models.resnet34(pretrained=False)
    in_features = model2.fc.in_features
    model2.fc = torch.nn.Linear(in_features,5)
    model3 = models.resnet50(pretrained=False)
    in_features = model3.fc.in_features
    model3.fc = torch.nn.Linear(in_features,5)
    
    model1.load_state_dict(torch.load('./modelResnet18noFreeze.pth'))
    model2.load_state_dict(torch.load('./modelResnet34noFreeze.pth'))
    model3.load_state_dict(torch.load('./modelResnet50.pth'))
    
    model1.eval()
    model2.eval()
    model3.eval()

    cumsum = 0
    goodPred = 0
    y_true = []
    y_pred = []
    for batch in test_loader:
        inputs, targets = batch
        inputs = Variable(inputs, volatile=True)
        targets = Variable(targets, volatile=True).data.numpy()[0]
        output1 = model1(inputs)
        output2 = model2(inputs)
        output3 = model3(inputs)
        
        prediction1 = output1.max(dim=1)[1].data.numpy()[0]
        prediction2 = output2.max(dim=1)[1].data.numpy()[0]
        prediction3 = output3.max(dim=1)[1].data.numpy()[0]
        
        prediction = np.array([prediction1,
                               prediction2,
                               prediction3])
        idMajority = np.argmax(np.unique(prediction,return_counts=True)[1])
        
        y_pred.append(prediction[idMajority])
        y_true.append(targets)
        if targets == prediction[idMajority]:
            goodPred += 1
        
        cumsum += 1
    print('test acc:', goodPred / cumsum * 100,'%')
    plot_confusion_matrix(np.array(y_true),np.array(y_pred))
        

