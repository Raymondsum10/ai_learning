#P12
from PIL import Image

from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs")
img_path = "dataset/train/ants/0013035.jpg"
img = Image.open(img_path)
print(img)

#to tensor
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image("totensor",img_tensor)

#normalize
print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([6,3,2],[9,3,5])
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image("Normalize" , img_norm,2)

writer.close()
