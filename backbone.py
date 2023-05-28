import torch,pdb
import torchvision
import torch.nn as nn
import torch.nn.modules
import timm
import math
from model.resnet_cbam import resnet18_cbam,BasicBlock
import torch.utils.model_zoo as model_zoo
import pretrainedmodels
import model.resnet_cbam

class vgg16bn(torch.nn.Module):
    def __init__(self,pretrained = False):
        super(vgg16bn,self).__init__()
        model = list(torchvision.models.vgg16_bn(pretrained=pretrained).features.children())
        model = model[:33]+model[34:43]
        self.model = torch.nn.Sequential(*model)

    def forward(self,x):
        return self.model(x)


class resnet(torch.nn.Module):
    def __init__(self,layers,pretrained = False):
        super(resnet,self).__init__()
        if layers == '18':
            model = torchvision.models.resnet18(pretrained=pretrained)
        elif layers == '34':
            model = torchvision.models.resnet34(pretrained=pretrained)
        elif layers == '50':
            model = torchvision.models.resnet50(pretrained=pretrained)
        elif layers == '101':
            model = torchvision.models.resnet101(pretrained=pretrained)
        elif layers == '152':
            model = torchvision.models.resnet152(pretrained=pretrained)
        elif layers == '50next':
            model = torchvision.models.resnext50_32x4d(pretrained=pretrained)
        elif layers == '101next':
            model = torchvision.models.resnext101_32x8d(pretrained=pretrained)
        elif layers == '50wide':
            model = torchvision.models.wide_resnet50_2(pretrained=pretrained)
        elif layers == '101wide':
            model = torchvision.models.wide_resnet101_2(pretrained=pretrained)
        elif layers == '34fca':
            model = torch.hub.load('cfzd/FcaNet', 'fca34' ,pretrained=False)
        elif layers == 'resnest50':
            model = torch.hub.load('zhanghang1989/ResNeSt', 'resnest50', pretrained=True)
        elif layers == 'resnest101':
            model = torch.hub.load('zhanghang1989/ResNeSt', 'resnest101', pretrained=True)
        elif layers == 'resnest200':
            model = torch.hub.load('zhanghang1989/ResNeSt', 'resnest200', pretrained=True)
        elif layers == 'resnest269':
            model = torch.hub.load('zhanghang1989/ResNeSt', 'resnest269', pretrained=True)
        elif layers == 'seresnet50':
            model = timm.create_model('seresnet50', pretrained=True)
        elif layers == 'seresnet18':
            #model = pretrainedmodels.se_resnet18(pretrained='imagenet')
            model = torch.hub.load(
                'moskomule/senet.pytorch',
                'se_resnet20',
                num_classes=10)
        elif layers == 'resnet18_cbam':
            model = resnet18_cbam(block=BasicBlock, layers=[2,2,2,2], num_classes=1000)
        else:
            raise NotImplementedError

        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        #self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        #self.layer4 = model.layer4

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #x = self.maxpool(x)
        x = self.layer1(x)
        x2 = self.layer2(x)
        x3 = self.layer3(x2)
        #x4 = self.layer4(x3)
        return x2,x3





