# 公众号：土堆碎念

import torch
import torch.nn.functional as F

input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]])

kernel = torch.tensor([[1, 2, 1],
                       [0, 1, 0],
                       [2, 1, 0]])

input = torch.reshape(input, (1, 1, 5, 5))
#mini batch = 1 , input channel = 1, H=5, w=5
kernel = torch.reshape(kernel, (1, 1, 3, 3))
#OUT channel = 1 , (input channel/groups)=1, H=3 ,W=3

print(input.shape)
print(kernel.shape)

#普通stride =1 ，只跳一格
output = F.conv2d(input, kernel, stride=1)
print(output)

#stride = 2 ， 跳二格
output2 = F.conv2d(input, kernel, stride=2)
print(output2)

#加入了padding
output3 = F.conv2d(input, kernel, stride=1, padding=1)
print(output3)