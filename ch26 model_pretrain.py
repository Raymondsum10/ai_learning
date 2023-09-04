import torchvision

# train_data = torchvision.datasets.ImageNet("../data_image_net", split='train', download=True,
#                                            transform=torchvision.transforms.ToTensor())
from torch import nn

vgg16_false = torchvision.models.vgg16(pretrained=False)
vgg16_true = torchvision.models.vgg16(pretrained=True)
#設置為 false 是用默認的參數，未pass 過data >> 可能效果不太好
#設置為 true 是訓練好的，有好的效果 >> VGG 最後的linear 層是1000 >> 同樣是分類的model

#查看model 的結構
print(vgg16_true)

train_data = torchvision.datasets.CIFAR10('../data', train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
#CIFAR10 有10 類，但是VGG 有1000 類， 解決方法就是把VGG 最後的FC layer/ Linear layer改成1000 變成10


#這裏是加入一個新的module >> 一個新的linear layer
vgg16_true.classifier.add_module('add_linear', nn.Linear(1000, 10))
print(vgg16_true)

print(vgg16_false)
vgg16_false.classifier[6] = nn.Linear(4096, 10)
print(vgg16_false)

