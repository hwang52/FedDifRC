#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import torch
import torchvision
import numpy as np
from torchvision import datasets
from torchvision.transforms import transforms
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Subset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import ConcatDataset
from utils.data_auge import AutoAugment
from utils.dataset import ImageFolder_custom
from utils.options import args_parser
from utils.fgvc_cub import Cub2011
from utils.fgvc_flower import FlowersDataset
from utils.fgvc_car import Cars
from utils.fgvc_dog import Dogs
from utils.fgvc_aircraft import Aircraft
from utils.digits_mnistm import MNISTM
from utils.digits_syn import SyntheticDigits
from torchvision.transforms import Lambda


class GrayscaleToRGB(object):
    def __call__(self, img):
        if img.mode == 'L':
            img = img.convert("RGB")
        return img


def get_tiny_class_names():
    class_to_name = dict()
    fp = open('./data/tiny_imagenet/words.txt', 'r')
    data = fp.readlines()
    for line in data:
        words = line.strip('\n').split('\t')
        class_to_name[words[0]] = words[1].split(',')[0]
    fp.close()
    class_names = []
    fp = open('./data/tiny_imagenet/wnids.txt', 'r')
    data = fp.readlines()
    for line in data:
        ids = line.strip('\n').split('\t')
        class_names.append(class_to_name[ids[0]])
    fp.close()
    return class_names


class get_dataset(object):
    def __init__(self, args):
        super().__init__()
        self.args = args
    
    def get_cifar10_dataset(self): # Cifar10 10
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]], 
                                    std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]], 
                                    std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        ])
        transform_diffu = transforms.Compose([
            transforms.Resize((self.args.diffu_size, self.args.diffu_size)),
            transforms.ToTensor(),
            Lambda(lambda x: (x/255.0 - 0.5) * 2)
        ])
        train_set = datasets.CIFAR10('./data/cifar10/', train=True, download=True, transform=transform_train)
        test_set = datasets.CIFAR10('./data/cifar10/', train=False, transform=transform_test)
        diffu_set = datasets.CIFAR10('./data/cifar10/', train=True, download=True, transform=transform_diffu)
        class_names = train_set.classes
        return train_set, test_set, diffu_set, class_names
    
    def get_cifar100_dataset(self): # Cifar100 100
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            AutoAugment(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343], 
                                 std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343], 
                                 std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
        ])
        transform_diffu = transforms.Compose([
            transforms.Resize((self.args.diffu_size, self.args.diffu_size)),
            transforms.ToTensor(),
            Lambda(lambda x: (x/255.0 - 0.5) * 2)
        ])
        train_set = datasets.CIFAR100('./data/cifar100/', train=True, download=True, transform=transform_train)
        test_set = datasets.CIFAR100('./data/cifar100/', train=False, transform=transform_test)
        diffu_set = datasets.CIFAR100('./data/cifar100/', train=True, download=True, transform=transform_diffu)
        ori_names = train_set.classes
        class_names = []
        for e_name in ori_names:
            new_name = e_name.replace('_', ' ')
            class_names.append(new_name)
        return train_set, test_set, diffu_set, class_names
    
    def get_tiny_dataset(self): # Tiny-ImageNet 200
        transform_train = transforms.Compose([
            AutoAugment(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        transform_diffu = transforms.Compose([
            transforms.Resize((self.args.diffu_size, self.args.diffu_size)),
            transforms.ToTensor(),
            Lambda(lambda x: (x/255.0 - 0.5) * 2)
        ])
        dl_obj = ImageFolder_custom
        train_set = dl_obj('./data/tiny_imagenet/train/', transform=transform_train)
        test_set = dl_obj('./data/tiny_imagenet/val/', transform=transform_test)
        diffu_set = dl_obj('./data/tiny_imagenet/train/', transform=transform_diffu)
        class_names = get_tiny_class_names()
        return train_set, test_set, diffu_set, class_names
    
    def get_fgvc_cub_dataset(self): # CUB 200 
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            AutoAugment(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        transform_diffu = transforms.Compose([
            transforms.Resize((self.args.diffu_size, self.args.diffu_size)),
            transforms.ToTensor(),
            Lambda(lambda x: (x/255.0 - 0.5) * 2)
        ])
        train_set = Cub2011('./data/cub2011', train=True, download=True, transform=transform_train)
        test_set = Cub2011('./data/cub2011', train=False, download=True, transform=transform_test)
        diffu_set = Cub2011('./data/cub2011', train=True, download=True, transform=transform_diffu)
        class_names = train_set.class_names
        for idx in range(len(class_names)):
            temp_name = class_names[idx].split('.')[1]
            temp_name = temp_name.replace('_', ' ')
            if "bird" not in temp_name:
                class_names[idx] = temp_name + " bird"
            else:
                class_names[idx] = temp_name
        return train_set, test_set, diffu_set, class_names
    
    def get_fgvc_flower_dataset(self): # Flowers 102 
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            AutoAugment(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        transform_diffu = transforms.Compose([
            transforms.Resize((self.args.diffu_size, self.args.diffu_size)),
            transforms.ToTensor(),
            Lambda(lambda x: (x/255.0 - 0.5) * 2)
        ])
        train_set = FlowersDataset('./data/flower102/train', transform=transform_train)
        # val_dataset = FlowersDataset('./data/flower102/valid', transform=transform_train)
        # train_set = ConcatDataset([train_dataset, val_dataset])
        test_set = FlowersDataset('./data/flower102/valid', transform=transform_test)
        diffu_set = FlowersDataset('./data/flower102/train', transform=transform_diffu)
        ori_names = train_set.classes
        class_names = []
        for each_name in ori_names:
            new_name = each_name
            if "flower" not in new_name:
                new_name = new_name + " flower"
            class_names.append(new_name)
        return train_set, test_set, diffu_set, class_names
    
    def get_fgvc_car_dataset(self): # Cars 196 
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            AutoAugment(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        transform_diffu = transforms.Compose([
            transforms.Resize((self.args.diffu_size, self.args.diffu_size)),
            transforms.ToTensor(),
            Lambda(lambda x: (x/255.0 - 0.5) * 2)
        ])
        train_set = Cars('./data/cars', train=True, img_path='./data/cars/cars_train', transform=transform_train)
        test_set = Cars('./data/cars', train=False, img_path='./data/cars/cars_test', transform=transform_test)
        diffu_set = Cars('./data/cars', train=True, img_path='./data/cars/cars_train', transform=transform_diffu)
        ori_class_names = train_set.class_names
        class_names = []
        for car_name in ori_class_names:
            new_name = car_name
            if "car" not in new_name:
                new_name = new_name + " car"
            class_names.append(new_name)
        return train_set, test_set, diffu_set, class_names
    
    def get_fgvc_dog_dataset(self): # Dog 120 
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            AutoAugment(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        transform_diffu = transforms.Compose([
            transforms.Resize((self.args.diffu_size, self.args.diffu_size)),
            transforms.ToTensor(),
            Lambda(lambda x: (x/255.0 - 0.5) * 2)
        ])
        train_set = Dogs('./data/dogs', train=True, download=True, transform=transform_train)
        test_set = Dogs('./data/dogs', train=False, download=True, transform=transform_test)
        diffu_set = Dogs('./data/dogs', train=True, download=True, transform=transform_diffu)
        class_names = []
        ori_class_names = train_set.class_names
        for dog_name in ori_class_names:
            new_name = dog_name.replace('_', ' ')
            if "dog" not in new_name:
                new_name = new_name + " dog"
            class_names.append(new_name)
        return train_set, test_set, diffu_set, class_names
    
    def get_fgvc_aircraft_dataset(self): # Aircraft 100 
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            AutoAugment(),
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
        ])
        transform_diffu = transforms.Compose([
            transforms.Resize((self.args.diffu_size, self.args.diffu_size)),
            transforms.ToTensor(),
            Lambda(lambda x: (x/255.0 - 0.5) * 2)
        ])
        train_set = Aircraft('./data/aircraft', train=True, download=True, transform=transform_train)
        test_set = Aircraft('./data/aircraft', train=False, download=True, transform=transform_test)
        diffu_set = Aircraft('./data/aircraft', train=True, download=True, transform=transform_diffu)
        class_names = train_set.find_classes()[2]
        for idx in range(len(class_names)):
            temp_name = class_names[idx].replace('\n', '')
            class_names[idx] = temp_name + " aircraft"
        return train_set, test_set, diffu_set, class_names
    
    def get_digits_mnistm_dataset(self): # Mnist-M 10
        transform_train = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        transform_test = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        transform_diffu = transforms.Compose([
            transforms.Resize((self.args.diffu_size, self.args.diffu_size)),
            transforms.ToTensor(),
            Lambda(lambda x: (x/255.0 - 0.5) * 2)
        ])
        train_set = MNISTM('./data/mnistm/', train=True, download=True, transform=transform_train)
        test_set = MNISTM('./data/mnistm/', train=False, download=True, transform=transform_test)
        diffu_set = MNISTM('./data/mnistm/', train=True, download=True, transform=transform_diffu)
        class_names = ['handwritten number 0','handwritten number 1','handwritten number 2',
                       'handwritten number 3','handwritten number 4','handwritten number 5',
                       'handwritten number 6','handwritten number 7','handwritten number 8',
                       'handwritten number 9']
        # random get 10% dataset
        subset_size = int(len(train_set) * 0.1)
        indices = list(range(len(train_set)))
        np.random.shuffle(indices)
        train_indices = indices[:subset_size]
        train_set = Subset(train_set, train_indices)
        return train_set, test_set, diffu_set, class_names
    
    def get_digits_svhn_dataset(self): # SVHN 10
        transform_train = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        transform_test = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        transform_diffu = transforms.Compose([
            transforms.Resize((self.args.diffu_size, self.args.diffu_size)),
            transforms.ToTensor(),
            Lambda(lambda x: (x/255.0 - 0.5) * 2)
        ])
        train_set = datasets.SVHN('./data/svhn/', split='train', transform=transform_train)
        test_set = datasets.SVHN('./data/svhn/', split='test', transform=transform_test)
        diffu_set = datasets.SVHN('./data/svhn/', split='train', transform=transform_diffu)
        class_names = ['street view house number 10','street view house number 1','street view house number 2',
                       'street view house number 3','street view house number 4','street view house number 5',
                       'street view house number 6','street view house number 7','street view house number 8',
                       'street view house number 9']
        # random get 10% dataset
        subset_size = int(len(train_set) * 0.1)
        indices = list(range(len(train_set)))
        np.random.shuffle(indices)
        train_indices = indices[:subset_size]
        train_set = Subset(train_set, train_indices)
        return train_set, test_set, diffu_set, class_names
    
    def get_digits_usps_dataset(self): # USPS 10 
        transform_train = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        transform_test = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        transform_diffu = transforms.Compose([
            transforms.Resize((self.args.diffu_size, self.args.diffu_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            Lambda(lambda x: (x/255.0 - 0.5) * 2)
        ])
        train_set = datasets.USPS('./data/usps/', train=True, download=True, transform=transform_train)
        test_set = datasets.USPS('./data/usps/', train=False, download=True, transform=transform_test)
        diffu_set = datasets.USPS('./data/usps/', train=True, download=True, transform=transform_diffu)
        class_names = ['number 0','number 1','number 2', 'number 3','number 4','number 5',
                       'number 6','number 7','number 8', 'number 9']
        # random get 10% dataset
        subset_size = int(len(train_set) * 0.1)
        indices = list(range(len(train_set)))
        np.random.shuffle(indices)
        train_indices = indices[:subset_size]
        train_set = Subset(train_set, train_indices)
        return train_set, test_set, diffu_set, class_names
    
    def get_digits_syn_dataset(self): # SYN 10
        transform_train = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        transform_test = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        transform_diffu = transforms.Compose([
            transforms.Resize((self.args.diffu_size, self.args.diffu_size)),
            transforms.ToTensor(),
            Lambda(lambda x: (x/255.0 - 0.5) * 2)
        ])
        train_set = SyntheticDigits('./data/syn/', train=True, download=True, transform=transform_train)
        test_set = SyntheticDigits('./data/syn/', train=False, download=True, transform=transform_test)
        diffu_set = SyntheticDigits('./data/syn/', train=True, download=True, transform=transform_diffu)
        class_names = ['number 0','number 1','number 2', 'number 3','number 4','number 5',
                       'number 6','number 7','number 8', 'number 9']
        # random get 10% dataset
        subset_size = int(len(train_set) * 0.1)
        indices = list(range(len(train_set)))
        np.random.shuffle(indices)
        train_indices = indices[:subset_size]
        train_set = Subset(train_set, train_indices)
        return train_set, test_set, diffu_set, class_names

    def get_office_caltech_dataset(self): # caltech 10
        transform_train = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        transform_test = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        transform_diffu = transforms.Compose([
            transforms.Resize((self.args.diffu_size, self.args.diffu_size)),
            transforms.ToTensor(),
            Lambda(lambda x: (x/255.0 - 0.5) * 2)
        ])
        train_set = datasets.ImageFolder("./data/caltech10/", transform=transform_train)
        test_set = datasets.ImageFolder("./data/caltech10/", transform=transform_test)
        diffu_set = datasets.ImageFolder("./data/caltech10/", transform=transform_diffu)
        class_names = ['backpack', 'bike', 'calculator', 'headphone', 'keyboard', 
                        'laptop', 'monitor', 'mouse', 'mug', 'projector']
        return train_set, test_set, diffu_set, class_names
    
    def get_office_amazon_dataset(self): # amazon 10
        transform_train = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        transform_test = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        transform_diffu = transforms.Compose([
            transforms.Resize((self.args.diffu_size, self.args.diffu_size)),
            transforms.ToTensor(),
            Lambda(lambda x: (x/255.0 - 0.5) * 2)
        ])
        train_set = datasets.ImageFolder("./data/amazon/", transform=transform_train)
        test_set = datasets.ImageFolder("./data/amazon/", transform=transform_test)
        diffu_set = datasets.ImageFolder("./data/amazon/", transform=transform_diffu)
        class_names = ['backpack', 'bike', 'calculator', 'headphone', 'keyboard', 
                       'laptop', 'monitor', 'mouse', 'mug', 'projector']
        return train_set, test_set, diffu_set, class_names
    
    def get_office_webcam_dataset(self): # webcam 10
        transform_train = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        transform_test = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        transform_diffu = transforms.Compose([
            transforms.Resize((self.args.diffu_size, self.args.diffu_size)),
            transforms.ToTensor(),
            Lambda(lambda x: (x/255.0 - 0.5) * 2)
        ])
        train_set = datasets.ImageFolder("./data/webcam/", transform=transform_train)
        test_set = datasets.ImageFolder("./data/webcam/", transform=transform_test)
        diffu_set = datasets.ImageFolder("./data/webcam/", transform=transform_diffu)
        class_names = ['backpack', 'bike', 'calculator', 'headphone', 'keyboard', 
                       'laptop', 'monitor', 'mouse', 'mug', 'projector']
        return train_set, test_set, diffu_set, class_names
    
    def get_office_dslr_dataset(self): # dslr 10
        transform_train = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        transform_test = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        transform_diffu = transforms.Compose([
            transforms.Resize((self.args.diffu_size, self.args.diffu_size)),
            transforms.ToTensor(),
            Lambda(lambda x: (x/255.0 - 0.5) * 2)
        ])
        train_set = datasets.ImageFolder("./data/dslr/", transform=transform_train)
        test_set = datasets.ImageFolder("./data/dslr/", transform=transform_test)
        diffu_set = datasets.ImageFolder("./data/dslr/", transform=transform_diffu)
        class_names = ['backpack', 'bike', 'calculator', 'headphone', 'keyboard', 
                       'laptop', 'monitor', 'mouse', 'mug', 'projector']
        return train_set, test_set, diffu_set, class_names