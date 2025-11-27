import os
from utils.transform import *
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import random
from PIL import ImageFilter
from PIL import Image
def blur(img, p=0.5):
    if random.random() < p:
        sigma = np.random.uniform(0.1, 2.0)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
    return img

# kvasir_SEG/ CVC-ClinicDB /kvasir_SEG + CVC-ClinicDB
class kvasir_SEG(Dataset):
    def __init__(self, root, data2_dir, mode='train', transform=None, cache=False):
        super(kvasir_SEG, self).__init__()
        self.data_path = os.path.join(root, data2_dir)

        self.id_list = []
        self.img_list = []
        self.gt_list = []
        self.mode = mode

        self.images_list = os.listdir(os.path.join(self.data_path, 'images'))  # images folder
        self.images_list = sorted(self.images_list)

        for img_id in self.images_list:
            self.id_list.append(img_id.split('.')[0])
            self.img_list.append(os.path.join(self.data_path, 'images', img_id))  # Image paths
            self.gt_list.append(os.path.join(self.data_path, 'masks', img_id))  # Mask paths

        if True:
            self.depth_list = []
            self.depth1_list = []
            for img_id in self.images_list:
                self.depth_list.append(os.path.join(self.data_path, 'depth_rgb', img_id))  # Depth paths for train mode
                self.depth1_list.append(os.path.join(self.data_path, 'depth', img_id))




        if transform is None:
            if mode == 'train':
                transform = transforms.Compose([
                    Resize((320, 320), train=True),
                    RandomHorizontalFlip(train=True),
                    RandomVerticalFlip(train=True),
                    RandomRotation(90,train=True),
                    RandomZoom((0.9, 1.1),train=True),
                ])
            elif mode == 'valid' or mode == 'test':
                transform = transforms.Compose([
                    Resize((320, 320),train=True),
                    #ToTensor(),
                ])
        self.transform = transform

        self.cache = cache
        if self.cache and mode == 'train':
            self.cache_img = list()
            self.cache_depth = list()
            self.cache_depth1 = list() # Cache for depth data
            self.cache_gt = list()
            for index in range(len(self.img_list)):
                img_path = self.img_list[index]
                depth_path = self.depth_list[index]
                depth1_path = self.depth1_list[index]
                gt_path = self.gt_list[index]

                self.cache_img.append(Image.open(img_path).convert('RGB'))
                self.cache_depth.append(Image.open(depth_path).convert('RGB'))
                self.cache_depth1.append(Image.open(depth1_path).convert('L'))
                self.cache_gt.append(Image.open(gt_path).convert('L'))
        elif self.cache and (mode == 'valid' or mode == 'test'):
            self.cache_img = list()
            self.cache_depth = list()
            self.cache_depth1 = list()  # Cache for depth data
            self.cache_gt = list()
            for index in range(len(self.img_list)):
                img_path = self.img_list[index]
                depth_path = self.depth_list[index]
                depth1_path = self.depth1_list[index]
                gt_path = self.gt_list[index]

                self.cache_img.append(Image.open(img_path).convert('RGB'))
                self.cache_depth.append(Image.open(depth_path).convert('RGB'))
                self.cache_depth1.append(Image.open(depth1_path).convert('L'))
                self.cache_gt.append(Image.open(gt_path).convert('L'))
    def __getitem__(self, index):
        if self.cache:
            img = self.cache_img[index]
            gt = self.cache_gt[index]
            depth = self.cache_depth[index]
            depth1 = self.cache_depth1[index]
        else:
            img_path = self.img_list[index]
            gt_path = self.gt_list[index]

            img = Image.open(img_path).convert('RGB')
            gt = Image.open(gt_path).convert('L')


            depth_path = self.depth_list[index]
            depth = Image.open(depth_path).convert('RGB')  # Load depth data for train mode
            depth1_path = self.depth1_list[index]
            depth1 = Image.open(depth1_path).convert('L')  # Load depth data for train mode


        #data = {'image': img, 'label': gt}
        if self.mode == 'train':
            data = {'image': img, 'label': gt, 'depth': depth, 'depth1': depth1} # Add depth data in train mode

            if self.transform:
                data = self.transform(data)
                to_tensor = transforms.ToTensor()
                data['label'] = to_tensor(data['label'])
                data['depth'] = to_tensor(data['depth'])  # Convert depth to tensor
                data['depth1'] = to_tensor(data['depth1'])
                img_s1 = Image.fromarray(np.array((data['image'])).astype(np.uint8))
                if random.random() < 0.8:
                    img_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s1)
                img_s1 = blur(img_s1, p=0.5)


                img_s1 = to_tensor(img_s1)
                data['image'] = to_tensor(data['image'])

                return {'id': self.id_list[index], 'image': data['image'], 'label': data['label'], 'image_s': img_s1, 'depth': data['depth'], 'depth1': data['depth1']}
        else:
            data = {'image': img, 'label': gt, 'depth': depth,'depth1':depth1}
            if self.transform:
                data = self.transform(data)
                to_tensor = transforms.ToTensor()
                data['label'] = to_tensor(data['label'])
                data['depth'] = to_tensor(data['depth'])
                data['depth1'] = to_tensor(data['depth1'])
                data['image'] = to_tensor(data['image'])
            return {'id': self.id_list[index], 'image': data['image'], 'label': data['label'],'depth':data['depth']}

    def __len__(self):
        return len(self.img_list)
