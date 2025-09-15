from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import os

img_path = "dataset/train/ants_image/0013035.jpg"

image = Image.open(img_path)
print(image)

tensor_trans = transforms.ToTensor() #实例化ToTensor()
tensor_img = tensor_trans(image)

print(tensor_img)

writer = SummaryWriter("logs")

writer.add_image("train_to_tensor", tensor_img)

writer.close()