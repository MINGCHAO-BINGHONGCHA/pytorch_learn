from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import os

img_path = "dataset/train/ants_image/0013035.jpg"
writer = SummaryWriter("logs")
image = Image.open(img_path)
print(image)

# ToTensor
tensor_trans = transforms.ToTensor() #实例化ToTensor()
tensor_img = tensor_trans(image)
print(tensor_img)
writer.add_image("train_to_tensor", tensor_img)

# Normalize
print(tensor_img[0][0][0])
trans_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
img_norm = trans_norm(tensor_img)
print(img_norm[0][0][0])

writer.add_image("train_norm", img_norm)

writer.close()