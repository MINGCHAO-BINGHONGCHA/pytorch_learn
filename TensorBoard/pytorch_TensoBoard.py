from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np

writer = SummaryWriter("logs")

for i in range(100):
    writer.add_scalar("y=x^2",i*i,i) # 绘制一个二次函数图像

image_path = "..\\hymenoptera_data\\train\\ants\\0013035.jpg"
image_PIL = Image.open(image_path)
image_np = np.array(image_PIL) #转换图片格式

print(image_np.shape)#输出图片的形状信息(H,W,C)

writer.add_image("test_image",image_np,3,dataformats="HWC")# 将图像可视化

writer.close()