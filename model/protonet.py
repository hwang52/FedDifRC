import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import Omniglot
from torchvision.models import resnet18
from tqdm import tqdm
from model.resnet_base import ResNet_cifar


image_size = 32
N_WAY = 5  # Number of classes in a task
N_SHOT = 5  # Number of images per class in the support set
N_QUERY = 10  # Number of images per class in the query set
N_EVALUATION_TASKS = 100
N_TRAINING_EPISODES = 40000
N_VALIDATION_TASKS = 100


def omni_trainloader():
    train_set = Omniglot(
        root="./data/",
        background=True,
        transform=transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=3),
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        ),
        download=True,
    )
    return train_set


def omni_testloader():
    test_set = Omniglot(
        root="./data/",
        background=False,
        transform=transforms.Compose(
            [
                # Omniglot images have 1 channel, but our model will expect 3-channel images
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize([int(image_size * 1.15), int(image_size * 1.15)]),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
            ]
        ),
        download=True,
    )
    return test_set



class ProtoNet(nn.Module):
    def __init__(self, backbone: nn.Module):
        super(ProtoNet, self).__init__()
        self.backbone = backbone

    def forward(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict query labels using labeled support images.
        """
        # Extract the features of support and query images
        z_support = self.backbone.forward(support_images)
        z_query = self.backbone.forward(query_images)
        # Infer the number of different classes from the labels of the support set
        n_way = len(torch.unique(support_labels))
        # Prototype i is the mean of all instances of features corresponding to labels == i
        z_proto = torch.cat(
            [
                z_support[torch.nonzero(support_labels == label)].mean(0)
                for label in range(n_way)
            ]
        )
        # Compute the euclidean distance from queries to prototypes
        dists = torch.cdist(z_query, z_proto)
        # And here is the super complicated operation to transform those distances into classification scores!
        scores = -dists
        # print("--------- scores.shape: ", scores.shape) [50,5]
        return scores

def protonet_model():
    backbone = ResNet_cifar(resnet_size=8, scaling=4, save_activations=False,
                            group_norm_num_groups=None, freeze_bn=False, freeze_bn_affine=False, num_classes=10)
    backbone.classifier = nn.Flatten()
    # print(convolutional_network)
    # model = ProtoNet(convolutional_network).cuda()
    model = ProtoNet(backbone)
    # print(model)
    return model


def evaluate_on_one_task(
    support_images: torch.Tensor,
    support_labels: torch.Tensor,
    query_images: torch.Tensor,
    query_labels: torch.Tensor,
    model):
    """
    Returns the number of correct predictions of query labels, and the total number of predictions.
    """
    return (
        torch.max(
            model(support_images, support_labels, query_images)
            .detach()
            .data,
            1,
        )[1]
        == query_labels
    ).sum().item(), len(query_labels)