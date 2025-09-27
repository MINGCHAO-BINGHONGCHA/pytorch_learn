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
writer.add_image("train_to_tensor", tensor_img,0)

# Normalize
print(tensor_img[0][0][0])
trans_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
img_norm = trans_norm(tensor_img)
print(img_norm[0][0][0])

writer.add_image("train_norm", img_norm,0)

# Resize
'''
    改变图片大小
    format:PIL -> Resize -> PIL
'''
trans_resize  = transforms.Resize((1000, 1000))
img_resize = trans_resize(image)
print(img_resize.size)
tensor_img_2 = transforms.ToTensor()(img_resize)
writer.add_image("train_resize", tensor_img_2 ,1)

# Compose
'''
    组合多个transform操作
    [操作1,操作2,操作3...]
    1.input -> 操作1 -> 1.output == 2.input -> 操作2 -> 2.output== 3.input -> 操作3 -> 3.output ...
'''

trans_resize_2 = transforms.Resize(512)
print("reszie and toTensor before:",image)
trans_compose = transforms.Compose([trans_resize_2, tensor_trans]) #使用实例化后的对象
trans_resize_2 = trans_compose(image)
print("resize and toTensor after:",trans_resize_2)

writer.add_image("trans_compose",trans_resize_2,2)

# RadomCrop
trans_randomcrop = transforms.RandomCrop(512)
trans_randomcrop_compose = transforms.Compose([trans_randomcrop, tensor_trans])

for i in range(10):
    trans_randomcrop_img = trans_randomcrop_compose(image)
    writer.add_image("trans_randomcrop_img",trans_randomcrop_img,i)

writer.close()