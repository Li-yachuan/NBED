from torch.utils import data
from os.path import join, abspath, splitext, split, isdir, isfile
from PIL import Image
from torchvision import transforms
import torch
import numpy as np
import random
import imageio
from pathlib import Path
from torch.nn.functional import interpolate


class BSDS_Loader(data.Dataset):
    """
    Dataloader BSDS500
    """

    def __init__(self, root='data/HED-BSDS', split='train', threshold=0.3, colorJitter=False):
        self.root = root
        self.split = split
        self.threshold = threshold
        print('Threshold for ground truth: %f on BSDS' % self.threshold)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        if colorJitter:
            self.transform = transforms.Compose([
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                normalize])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                normalize])
        if self.split == 'train':
            self.filelist = join(self.root, 'train_BSDS.lst')
        elif self.split == 'test':
            self.filelist = join(self.root, 'test.lst')
        else:
            raise ValueError("Invalid split type!")
        with open(self.filelist, 'r') as f:
            self.filelist = f.readlines()

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        if self.split == "train":
            img_lb_file = self.filelist[index].strip("\n").split(" ")
            img_file = img_lb_file[0]

            label_list = []
            for lb_file in img_lb_file[1:]:
                label_list.append(
                    transforms.ToTensor()(Image.open(join(self.root, lb_file))))
            lb = torch.cat(label_list, 0).mean(0, keepdim=True)

            lb[lb >= self.threshold] = 1
            lb[(lb > 0) & (lb < self.threshold)] = 2

        else:
            img_file = self.filelist[index].rstrip()

        with open(join(self.root, img_file), 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        img = self.transform(img)

        if self.split == "train":
            return img, lb
        else:
            img_name = Path(img_file).stem
            return img, img_name


class NYUD_Loader(data.Dataset):
    """
    Dataloader BSDS500
    """

    def __init__(self, root='data/HED-BSDS_PASCAL', split='train', mode="RGB"):
        self.root = root
        self.split = split
        #
        if mode == "RGB":
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
        else:  # calculate by Archer
            normalize = transforms.Normalize(mean=[0.519, 0.370, 0.465],
                                             std=[0.226, 0.246, 0.186])

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            normalize])

        if self.split == 'train':
            if mode == "RGB":
                self.filelist = join(root, "image-train.txt")
            else:
                self.filelist = join(root, "hha-train.txt")

        elif self.split == 'test':
            if mode == "RGB":
                self.filelist = join(root, "image-test.txt")
            else:
                self.filelist = join(root, "hha-test.txt")

        else:
            raise ValueError("Invalid split type!")

        with open(self.filelist, 'r') as f:
            self.filelist = f.readlines()

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        if self.split == "train":
            img_file, lb_file = self.filelist[index].strip("\n").split(" ")

        else:
            img_file = self.filelist[index].strip("\n").split(" ")[0]

        img = imageio.imread(join(self.root, img_file))
        img = self.transform(img)

        if self.split == "train":
            label = transforms.ToTensor()(imageio.imread(join(self.root, lb_file), as_gray=True)) / 255
            img, label = self.crop(img, label)
            return img, label

        else:
            img_name = Path(img_file).stem
            return img, img_name

    @staticmethod
    def crop(img, lb):
        _, h, w = img.size()
        crop_size = 400

        if h < crop_size or w < crop_size:
            resize_scale = round(max(crop_size / h, crop_size / w) + 0.1, 1)

            img = interpolate(img.unsqueeze(0), scale_factor=resize_scale, mode="bilinear").squeeze(0)
            lb = interpolate(lb.unsqueeze(0), scale_factor=resize_scale, mode="nearest").squeeze(0)
        _, h, w = img.size()
        i = random.randint(0, h - crop_size)
        j = random.randint(0, w - crop_size)
        img = img[:, i:i + crop_size, j:j + crop_size]
        lb = lb[:, i:i + crop_size, j:j + crop_size]

        return img, lb


class BIPED_Loader(data.Dataset):
    """
    Dataloader BSDS500
    """

    def __init__(self, root=' ', split='train', transform=False):
        self.root = root
        self.split = split
        self.transform = transform
        if self.split == 'train':
            self.filelist = join(root, "train_pair.lst")

        elif self.split == 'test':
            self.filelist = join(root, "test.lst")
        else:
            raise ValueError("Invalid split type!")
        with open(self.filelist, 'r') as f:
            self.filelist = f.readlines()
        # print(self.filelist)

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        if self.split == "train":
            img_file, lb_file = self.filelist[index].strip("\n").split(" ")

        else:
            img_file = self.filelist[index].rstrip()

        img = imageio.imread(join(self.root, img_file))
        img = transforms.ToTensor()(img)

        if self.split == "train":
            label = transforms.ToTensor()(imageio.imread(join(self.root, lb_file), as_gray=True)) / 255
            img, label = self.crop(img, label)
            return img, label

        else:
            img_name = Path(img_file).stem
            return img, img_name

    @staticmethod
    def crop(img, lb):
        _, h, w = img.size()
        assert (h > 400) and (w > 400)
        crop_size = 400
        i = random.randint(0, h - crop_size)
        j = random.randint(0, w - crop_size)
        img = img[:, i:i + crop_size, j:j + crop_size]
        lb = lb[:, i:i + crop_size, j:j + crop_size]

        return img, lb


class Multicue_Loader(data.Dataset):
    """
    Dataloader for Multicue
    """

    def __init__(self, root='data/', split='train', transform=False, threshold=0.3, setting=['boundary', '1']):
        """
        setting[0] should be 'boundary' or 'edge'
        setting[1] should be '1' or '2' or '3'
        """
        self.root = root
        self.split = split
        self.threshold = threshold
        print('Threshold for ground truth: %f on setting %s' % (self.threshold, str(setting)))
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            normalize])
        if self.split == 'train':
            self.filelist = join(self.root, 'train_pair_%s_set_%s.lst' % (setting[0], setting[1]))
        elif self.split == 'test':
            self.filelist = join(self.root, 'test_%s_set_%s.lst' % (setting[0], setting[1]))
        else:
            raise ValueError("Invalid split type!")
        with open(self.filelist, 'r') as f:
            self.filelist = f.readlines()

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        if self.split == "train":
            img_file, lb_file = self.filelist[index].split()
            img_file = img_file.strip()
            lb_file = lb_file.strip()

            lb = transforms.ToTensor()(Image.open(join(self.root, lb_file)).convert("L"))

            lb[lb > self.threshold] = 1
            lb[(lb > 0) & (lb < self.threshold)] = 2

        else:
            img_file = self.filelist[index].rstrip()

        with open(join(self.root, img_file), 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        img = self.transform(img)

        if self.split == "train":
            return self.crop(img, lb)
        else:
            img_name = Path(img_file).stem
            return img, img_name

    @staticmethod
    def crop(img, lb):
        _, h, w = img.size()
        crop_size = 400

        if (h < crop_size) or (w < crop_size):
            resize_scale = round(max(crop_size / h, crop_size / w) + 0.1, 1)

            img = interpolate(img.unsqueeze(0), scale_factor=resize_scale, mode="bilinear").squeeze(0)
            lb = interpolate(lb.unsqueeze(0), scale_factor=resize_scale, mode="nearest").squeeze(0)
        _, h, w = img.size()
        i = random.randint(0, h - crop_size)
        j = random.randint(0, w - crop_size)
        img = img[:, i:i + crop_size, j:j + crop_size]
        lb = lb[:, i:i + crop_size, j:j + crop_size]

        return img, lb
