from torch.utils.data import Dataset
from PIL import Image
import os

class MyData(Dataset):
    
    def __init__(self, root_dir, train_dir):
        self.root_dir = root_dir
        self.train_dir = train_dir
        self.image_paths = os.listdir(os.path.join(root_dir, train_dir)) # 拼接路径并将路径中的图片名称存储到一个数组中
    def __getitem__(self, index):
        img_name = self.image_paths[index] # 获取第index张图片的名称
        img_item_path = os.path.join(self.root_dir, self.train_dir, img_name) #拼接路径
        img = Image.open(img_item_path)
        train = self.train_dir
        
        return img, train

    def __len__(self):
        return len(self.image_paths)
def testCode():
    img_path = "hymenoptera_data/train/ants/0013035.jpg"
    img = Image.open(img_path)
    img.show()
    print(img.size)
    
    dir_path = "hymenoptera_data/train/ants"
    img_path_list = os.listdir(dir_path)
    print(img_path_list[0])
    
    ants_datasets = MyData(root_dir="hymenoptera_data/train", train_dir="ants")
    bees_datasets = MyData(root_dir="hymenoptera_data/train", train_dir="bees")
    # 指定两个数据集位置
    
    
    img, train = ants_datasets[0] # 获取ants中的第1张图片
    img.show()
    print(train)
    
    img, train = bees_datasets[1] # 获取bees中的第2张图片
    img.show()
    print(train)
    
    train_datasets = ants_datasets + bees_datasets #  将两个数据集进行拼接
    
    print(len(train_datasets))
    
def main():
    testCode()
    
if __name__ == "__main__":
    main()