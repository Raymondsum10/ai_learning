
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter("logs")
image_path="data/val/ants/800px-Meat_eater_ant_qeen_excavating_hole.jpg"
img_PIL = Image.open(image_path)
img_array = np.array(img_PIL)
print(type(img_array))
print(img_array.shape)
#會print出(H,W,C) = (高度，寬度，通道channel)

writer.add_image("train", img_array,1,dataformats="HWC")

for i in range(100):
    writer.add_scalar("y=3x", 3*i, i)

writer.close()

#需要安裝pytorch環境的tensorboard

#在下面terminal 安裝
#pip install tensorboard
#pip install opencv-python

#輸入下面指令出tensorboard的圖表
#tensorboard --logdir=logs
#tensorboard --logdir=logs --port=6007
