import os
import argparse
import torch
import torchvision
from torchvision import utils as vtuils
# from torch import nn
from PIL import Image


def apply(img_dir, dst_dir, aug, num_aug=8):
    img = Image.open(img_dir)
    for i in range(num_aug):
        aug(img).save(dst_dir+str(i)+'.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wikidir', type=str, default='../Dataset/wikiart', help="path of wiki dataset")
    parser.add_argument('--augdir', type=str, default='../Dataset/wikiartforclassify', help = 'path of new classify dataset')

    args = parser.parse_args()
    shape_aug = torchvision.transforms.RandomResizedCrop(
        (224, 224), scale=(0.6, 1), ratio=(0.5, 2))


    for base in os.listdir(args.wikidir):
        base_dir = os.path.join(args.wikidir, base)
        print(len(os.listdir(base_dir)))

    for base in os.listdir(args.wikidir):
        base_dir = os.path.join(args.wikidir, base)
        if len(os.listdir(base_dir)) < 300:
            continue

        if not os.path.exists(os.path.join(args.augdir, base)):
            os.makedirs(os.path.join(args.augdir, base))

        num = 2200//len(os.listdir(base_dir)) + 1
        cnt = 0
        for sub in os.listdir(base_dir):
            img_dir = os.path.join(base_dir, sub)
            # for i in os.listdir(sub_dir):
            #     img_dir = os.path.join(sub_dir, i)
            dst_dir = os.path.join(os.path.join(args.augdir, base), sub.split('.')[0])
            apply(img_dir, dst_dir, shape_aug, num)
            cnt += 1
            if cnt == 2200:
                break



