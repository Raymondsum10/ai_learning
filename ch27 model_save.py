

#把改了的model save下來


import torch
import torchvision
from torch import nn

vgg16 = torchvision.models.vgg16(pretrained=False)
# 保存方式1,模型结构+模型参数
torch.save(vgg16, "vgg16_method1.pth")

# 保存方式2，只保存模型参数（官方推荐）
torch.save(vgg16.state_dict(), "vgg16_method2.pth")

