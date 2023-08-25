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

#resize
print(img.size)
trans_resize = transforms.Resize((512,512))
img_resize = trans_resize(img)
img_resize = trans_totensor(img_resize)
writer.add_image("resize" , img_resize,0)
print(img_resize)

#compose - resize - 2
trans_resize_2 = transforms.Resize(512)
#PIL > PIL > tensor
trans_compose = transforms.Compose([trans_resize_2, trans_totensor])
img_resize_2 = trans_compose(img)
writer.add_image("resize" , img_resize_2,1)

#RandomCrop
trans_random = transforms.RandomCrop((500, 700))
trans_compose_2 = transforms.Compose([trans_random, trans_totensor])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("RandomCrophw", img_crop, i)

writer.close()
