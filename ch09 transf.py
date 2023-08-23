#P11
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from torchvision import transforms


img_path = "dataset/train/ants/0013035.jpg"
img = Image.open(img_path)

tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)

print(tensor_img)
