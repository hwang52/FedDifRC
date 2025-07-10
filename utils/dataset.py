import numpy as np
from torch.utils.data.dataset import Dataset
import copy
from torchvision.datasets import ImageFolder, DatasetFolder


# def classify_label(dataset, num_classes: int):
#     list1 = [[] for _ in range(964)]
#     dataset_labels = [instance[1] for instance in dataset._flat_character_images]
#     # print(dataset_labels)
#     print(len(dataset))
#     for idx, _ in enumerate(dataset):
#         # print("idx=",idx,"  label=",dataset_labels[idx])
#         list1[dataset_labels[idx]].append(idx)
#     return list1

def classify_label(dataset, num_classes: int):
    list1 = [[] for _ in range(num_classes)]
    for idx, datum in enumerate(dataset):
        list1[datum[1]].append(idx)
    return list1




def show_clients_data_distribution(dataset, clients_indices: list, num_classes):
    dict_per_client = []
    for client, indices in enumerate(clients_indices):
        nums_data = [0 for _ in range(num_classes)]
        for idx in indices:
            label = dataset[idx][1]
            nums_data[label] += 1
        dict_per_client.append(nums_data)
        print(f'{client}: {nums_data}')
    return dict_per_client


def partition_train_teach(list_label2indices: list, ipc, seed=None):
    random_state = np.random.RandomState(0)
    list_label2indices_teach = []

    for indices in list_label2indices:
        random_state.shuffle(indices)
        list_label2indices_teach.append(indices[:ipc])

    return list_label2indices_teach


def partition_unlabel(list_label2indices: list, num_data_train: int):
    random_state = np.random.RandomState(0)
    list_label2indices_unlabel = []

    for indices in list_label2indices:
        random_state.shuffle(indices)
        list_label2indices_unlabel.append(indices[:num_data_train // 100])
    return list_label2indices_unlabel


def label_indices2indices(list_label2indices):
    indices_res = []
    for indices in list_label2indices:
        indices_res.extend(indices)

    return indices_res


class Indices2Dataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.indices = None
        self.targets = None

    def load(self, indices: list):
        self.targets = []
        self.indices = indices

    def __getitem__(self, idx):
        idx = self.indices[idx]
        image, label = self.dataset[idx]
        # print("image:",image,"  label:", label)
        return image, label

    def __len__(self):
        return len(self.indices)


class diffu_Indices2Dataset(Dataset):
    def __init__(self, dataset, diffu_feas, diffu_text_feas):
        self.dataset = dataset
        self.diffu_feas = diffu_feas
        self.diffu_text_feas = diffu_text_feas
        self.indices = None
        self.targets = None

    def load(self, indices: list):
        self.targets = []
        self.indices = indices

    def __getitem__(self, idx):
        idx = self.indices[idx]
        image, label = self.dataset[idx]
        dif_fea = self.diffu_feas[idx]
        dif_text_fea = self.diffu_text_feas[idx]
        # print(image[:,10:15])
        # print("image:",image.shape,"  dif_fea:", dif_fea.shape)
        return image, dif_fea, dif_text_fea, label

    def __len__(self):
        return len(self.indices)


class TensorDataset(Dataset):
    def __init__(self, images, labels): # images: n x c x h x w tensor
        self.images = images.detach().float()
        self.labels = labels.detach()

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.images.shape[0]


def get_class_num(class_list):
    index = []
    compose = []
    for class_index, j in enumerate(class_list):
        if j != 0:
            index.append(class_index)
            compose.append(j)
    return index, compose

class ImageFolder_custom(DatasetFolder):
    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        imagefolder_obj = ImageFolder(self.root, self.transform, self.target_transform)
        self.loader = imagefolder_obj.loader
        if self.dataidxs is not None:
            self.samples = np.array(imagefolder_obj.samples)[self.dataidxs]
        else:
            self.samples = np.array(imagefolder_obj.samples)

    def __getitem__(self, index):
        path = self.samples[index][0]
        target = self.samples[index][1]
        target = int(target)
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        if self.dataidxs is None:
            return len(self.samples)
        else:
            return len(self.dataidxs)