import os
import json
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, ConcatDataset


class FlowersDataset(ImageFolder):
    def __init__(self, root, transform):
        self.root = root
        super().__init__(self.root)
        self.files = [folder+"/"+fname for folder in os.listdir(root) for fname in os.listdir(root+"/"+folder) if fname.endswith('.jpg')]
        self.transform = transform
        with open('./data/flower102/cat_to_name.json', 'r') as f:
          self.classes_idx = json.load(f)
        category_map = sorted(self.classes_idx.items(), key=lambda x: int(x[0]))
        self.classes = [cat[1] for cat in category_map]
    
    def __len__(self):
        return len(self.files)


if __name__ == '__main__':
    train_dir = './dataset/flower102/train'
    train_dataset = FlowersDataset(train_dir, transform=None)
    val_dataset = FlowersDataset('./dataset/flower102/valid', transform=None)
    # print(len(train_dataset))
    # print(len(val_dataset))
    # com_dataset = ConcatDataset([train_dataset, val_dataset])
    # print(len(com_dataset))
    print(train_dataset.classes)