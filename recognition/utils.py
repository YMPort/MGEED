import torch
import torchvision
from torchvision import models, transforms

class resnet_extend(models.resnet.ResNet):
    def __init__(self):
        super(resnet_extend, self).__init__(models.resnet.BasicBlock, [3, 4, 23, 3])
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        return x