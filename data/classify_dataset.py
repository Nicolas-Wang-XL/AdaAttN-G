import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import torch.backends.cudnn as cudnn
import torch
from PIL import ImageFile

cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Disable OSError: image file is truncated


class ClassifyDataset(BaseDataset):

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.dir = opt.style_path
        self.paths = sorted(make_dataset(self.dir, opt.max_dataset_size))
        labels = sorted(os.listdir(self.dir))
        self.labels_dict = {}
        for i, lb in enumerate(labels):
            self.labels_dict[lb] = i
        print(self.labels_dict)
        self.size = len(self.paths)
        self.transform = get_transform(self.opt)

    def __getitem__(self, index):
        if self.opt.isTrain:
            index_img = index
        else:
            index_img = index // self.size
        img_path = self.paths[index_img]
        img = Image.open(img_path).convert('RGB')
        x = self.transform(img)

        label_name = os.path.basename(os.path.dirname(img_path))
        label = self.labels_dict[label_name]

        result = {'x': x, 'y': label}

        return result

    def __len__(self):
        if self.opt.isTrain:
            return self.size
        else:
            return min(self.size, self.opt.num_test)
