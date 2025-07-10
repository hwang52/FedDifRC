import os
import scipy.io as sio
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torchvision.datasets.utils import extract_archive


class Cars(VisionDataset):
    """`Stanford Cars <https://ai.stanford.edu/~jkrause/cars/car_dataset.html>`_ Dataset.

    Args:
        root (string): Root directory of the dataset.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    def __init__(self, root, img_path, train=True, transform=None, target_transform=None, download=False):
        super(Cars, self).__init__(root, transform=transform, target_transform=target_transform)

        self.loader = default_loader
        self.train = train
        self.img_path = img_path

        # if self._check_exists():
        #     print('Files already downloaded and verified.')
        # elif download:
        #     self._download()
        # else:
        #     raise RuntimeError(
        #         'Dataset not found. You can use download=True to download it.')
        
        cars_meta_mat = sio.loadmat(os.path.join(self.root, 'cars_meta.mat'))
        class_names = [arr[0] for arr in cars_meta_mat['class_names'][0]]
        self.class_names = class_names

        self.samples = []
        if self.train:
            train_loaded_mat = sio.loadmat(os.path.join(self.root, 'cars_train_annos.mat'))
            train_loaded_mat = train_loaded_mat['annotations'][0]
            for item in train_loaded_mat:
                path = str(item[-1][0])
                label = int(item[-2][0]) - 1
                self.samples.append((path, label))
        else:
            test_loaded_mat = sio.loadmat(os.path.join(self.root, 'cars_test_annos_withlabels.mat'))
            test_loaded_mat = test_loaded_mat['annotations'][0]
            for item in test_loaded_mat:
                path = str(item[-1][0])
                label = int(item[-2][0]) - 1
                self.samples.append((path, label))

    def __getitem__(self, index):
        path, target = self.samples[index]
        path = os.path.join(self.img_path, path)

        image = self.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target

    def __len__(self):
        return len(self.samples)


if __name__ == '__main__':
    train_dataset = Cars('../dataset/cars', train=True, img_path='../dataset/cars/cars_train')
    test_dataset = Cars('../dataset/cars', train=False, img_path='../dataset/cars/cars_test')
