#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
from torchvision import datasets
from torchvision.transforms import transforms
from utils.options import args_parser
from utils.long_tailed_cifar10 import train_long_tail
from utils.dataset import classify_label, show_clients_data_distribution, Indices2Dataset, diffu_Indices2Dataset
from utils.sample_dirichlet import clients_indices
import numpy as np
from torch import eq
import torch.nn.functional as F
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader
from model.resnet_base import ResNet_cifar
from model.resnet import ResNet18
from tqdm import tqdm
import copy, sys, os
import torch
import random
import faiss
from sklearn.decomposition import PCA
from pathlib import Path
from torchvision import models
from utils_dataset import get_dataset
from model.mobilenetv2 import mobilenetv2
from model.mobilenetv3 import mobilenetv3_small
from model.resnet_cnn import resnet10


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

model_dir = (Path(__file__).parent / "model").resolve()
if str(model_dir) not in sys.path: sys.path.insert(0, str(model_dir))
utils_dir = (Path(__file__).parent / "utils").resolve()
if str(utils_dir) not in sys.path: sys.path.insert(0, str(utils_dir))


class Global(object):
    def __init__(self, num_classes: int,device: str, args):
        self.device = device
        self.num_classes = num_classes
        self.args = args
        d_n = args.data_name
        if (d_n=='cifar10' or d_n=='cifar100'): # ResNet-10 (fea 512)
            self.global_model = resnet10(nclasses=args.num_classes).to(args.device)
        elif d_n=='tiny': # ResNet-10 (fea 512)
            self.global_model = resnet10(nclasses=args.num_classes).to(args.device)
        elif d_n.startswith('fgvc'): # MobileNet-V2 (fea 512)
            self.global_model = mobilenetv2(nclasses=args.num_classes).to(args.device)
        elif (d_n.startswith('digits') or d_n.startswith('office')): # ResNet-10 (fea 512)
            self.global_model = resnet10(nclasses=args.num_classes).to(args.device)
        else:
            exit('Load model error: unknown model!')

    def fedavg_parameters(self, list_dicts_local_params: list, list_nums_local_data: list):
        # average aggregation
        fedavg_global_params = copy.deepcopy(list_dicts_local_params[0])
        for name_param in list_dicts_local_params[0]:
            list_values_param = []
            for dict_local_params, num_local_data in zip(list_dicts_local_params, list_nums_local_data):
                list_values_param.append(dict_local_params[name_param] * num_local_data)
            value_global_param = sum(list_values_param) / sum(list_nums_local_data)
            fedavg_global_params[name_param] = value_global_param
        return fedavg_global_params

    def global_eval(self, fedavg_params, data_test, batch_size_test):
        self.global_model.load_state_dict(fedavg_params)
        self.global_model.eval()
        with torch.no_grad():
            num_corrects = 0
            test_loader = DataLoader(data_test, batch_size_test, 
                                pin_memory=True, num_workers=6, shuffle=False)
            for data_batch in test_loader:
                images, labels = data_batch
                feas, outputs = self.global_model(images.to(self.device))
                _, predicts = torch.max(outputs, -1)
                num_corrects += sum(eq(predicts.cpu(), labels.cpu())).item()
            accuracy = num_corrects / len(data_test)
        return accuracy

    def download_params(self):
        return self.global_model.state_dict()


class Local(object):
    def __init__(self, data_client, class_list: int, text_names):
        args = args_parser()
        self.data_client = data_client
        self.device = args.device
        self.class_compose = class_list
        self.text_names = text_names
        self.infonce_t = args.infonce_t
        d_n = args.data_name
        if (d_n=='cifar10' or d_n=='cifar100'): # ResNet-10 (fea 512)
            self.local_model = resnet10(nclasses=args.num_classes).to(args.device)
        elif d_n=='tiny': # ResNet-10 (fea 512)
            self.local_model = resnet10(nclasses=args.num_classes).to(args.device)
        elif d_n.startswith('fgvc'): # MobileNet-V2 (fea 512)
            self.local_model = mobilenetv2(nclasses=args.num_classes).to(args.device)
        elif (d_n.startswith('digits') or d_n.startswith('office')): # ResNet-10 (fea 512)
            self.local_model = resnet10(nclasses=args.num_classes).to(args.device)
        else:
            exit('Load model error: unknown model!')
        # loss function and optimizer
        self.loss_ce = torch.nn.CrossEntropyLoss(reduction='mean').to(args.device)
        self.loss_l1 = torch.nn.L1Loss(reduction='mean').to(args.device)
        self.loss_cos = torch.nn.CosineEmbeddingLoss(reduction='mean').to(args.device)
        self.loss_kl = torch.nn.KLDivLoss(reduction='batchmean').to(args.device)
        if d_n=='cifar10':
            self.optimizer = SGD(filter(lambda p: p.requires_grad, self.local_model.parameters()), 
                                    lr=0.01, momentum=0.9, weight_decay=1e-5) # cifar10
        else:
            self.optimizer = SGD(filter(lambda p: p.requires_grad, self.local_model.parameters()), 
                                    lr=0.01, momentum=0.9, weight_decay=1e-4) # others

    @torch.no_grad()
    def get_noise_fea(self, noise_sdfea_list):
        nsd_fea1, nsd_fea2, nsd_fea3 = noise_sdfea_list[1], noise_sdfea_list[2], noise_sdfea_list[3]
        target_dims = {'s1': 256, 's2': 128, 's3': 128}
        s1 = nsd_fea1.reshape(nsd_fea1.shape[0], nsd_fea1.shape[1], -1) # b,1280,h*w
        s2 = nsd_fea2.reshape(nsd_fea2.shape[0], nsd_fea2.shape[1], -1) # b,640,h*w
        s3 = nsd_fea2.reshape(nsd_fea3.shape[0], nsd_fea3.shape[1], -1) # b,320,h*w
        nsd_features = {'s1':s1, 's2':s2, 's3':s3}

        for name, tensor in zip(['s1', 's2', 's3'], [s1, s2, s3]):
            target_dim = target_dims[name]
            b_size = tensor.shape[0]
            tensor = tensor.permute(0, 2, 1) # b,h*w,c
            tensor = tensor.reshape(-1, tensor.shape[-1]) # b*h*w,c
            # Initialize a Faiss PCA object
            pca = faiss.PCAMatrix(tensor.shape[-1], target_dim)
            # Train the PCA object
            pca.train(tensor.cpu().numpy())
            # Apply PCA to the data
            transformed_tensor_np = pca.apply(tensor.cpu().numpy()) # b*h*w,256
            # Convert the transformed data back to a tensor
            transformed_tensor = torch.tensor(transformed_tensor_np, device=self.device) # b*h*w,256
            # Store the transformed tensor in the features dictionary
            nsd_features[name] = transformed_tensor.reshape(b_size, -1, target_dim) # b,h*w,256
        nsd_fea1 = nsd_features['s1'].permute(0, 2, 1).norm(dim=-1) # b,256,h*w->b,256
        nsd_fea2 = nsd_features['s2'].permute(0, 2, 1).norm(dim=-1) # b,128,h*w->b,128
        nsd_fea3 = nsd_features['s3'].permute(0, 2, 1).norm(dim=-1) # b,128,h*w->b,128
        nsd_fea = torch.cat((nsd_fea1, nsd_fea2, nsd_fea3), dim=1) # b,256+128+128=512
        return nsd_fea
    
    def calculate_infonce(self, f_now, f_pos, f_neg):
        f_proto = torch.cat((f_pos, f_neg), dim=0)
        l = torch.cosine_similarity(f_now, f_proto, dim=1)
        l = l / self.infonce_t 
        exp_l = torch.exp(l)
        exp_l = exp_l.view(1, -1)
        pos_mask = [1 for _ in range(f_pos.shape[0])] + [0 for _ in range(f_neg.shape[0])]
        pos_mask = torch.tensor(pos_mask, dtype=torch.float).to(self.device)
        pos_mask = pos_mask.view(1, -1)
        # pos_l = torch.einsum('nc,ck->nk', [exp_l, pos_mask])
        pos_l = exp_l * pos_mask
        sum_pos_l = pos_l.sum(1)
        sum_exp_l = exp_l.sum(1)
        infonce_loss = -torch.log(sum_pos_l / sum_exp_l)
        return infonce_loss

    def local_train(self, args, global_params):
        self.local_model.load_state_dict(global_params)
        self.local_model.train()
        for tr_idx in range(args.num_epochs_local_training):
            data_loader = DataLoader(
                dataset=self.data_client, 
                batch_size=args.batch_size_local_training, 
                pin_memory=True, num_workers=6, shuffle=True
            )
            for data_batch in data_loader:
                images, sd_feas, sd_text_feas, labels = data_batch
                images = images.to(self.device)
                # compute feas
                feas, outputs = self.local_model(images)
                # 1. ce loss
                loss_ce = self.loss_ce(outputs, labels.to(self.device))
                # 2. noise-driven diffu loss
                sd_feas = torch.tensor(sd_feas.clone().detach(), device=self.device) # [b,512]
                norm_feas = feas / feas.norm(dim=-1, keepdim=True) # [b,512]
                norm_sd_feas = sd_feas / sd_feas.norm(dim=-1, keepdim=True) # [b,512]
                loss_noise = self.loss_kl(F.log_softmax(norm_feas, dim=1), F.softmax(norm_sd_feas, dim=1))
                # 3. text_driven diffu loss
                sd_text_feas = torch.tensor(sd_text_feas.clone().detach(), device=self.device)
                f_now = feas
                f_pos, f_neg = [], []
                for sd_text_fea in sd_text_feas:
                    f_pos.append(sd_text_fea[-1])
                    f_neg.append(sd_text_fea[:-1])
                f_pos = torch.stack(f_pos)
                f_neg = torch.stack(f_neg).reshape(-1, f_now.size(-1))
                repeat_num = int((f_pos.size(0)+f_neg.size(0))/feas.size(0))
                f_now = f_now.repeat(repeat_num, 1)
                loss_text = self.calculate_infonce(f_now, f_pos, f_neg)
                # Backpropagation
                loss = loss_ce + loss_noise + loss_text 
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

        return self.local_model.state_dict()


def FedDifRC_param():
    args = args_parser()
    print(
        '=====> long-tail rate (imb_factor): {ib}\n'
        '=====> non-iid rate (non_iid): {non_iid}\n'
        '=====> activate clients (num_online_clients): {num_online_clients}\n'
        '=====> dataset classes (num_classes): {num_classes}\n'.format(
            ib=args.imb_factor,  # long-tail imbalance factor
            non_iid=args.non_iid_alpha,  # non-iid alpha based on Dirichlet-distribution
            num_online_clients=args.num_online_clients,  # activate clients in FL
            num_classes=args.num_classes)  # dataset classes
    )

    random_state = np.random.RandomState(args.seed)
    # get dataset
    all_dataset = get_dataset(args)
    data_local_training, data_global_test, diffu_set, class_names = getattr(all_dataset, 
            'get_{}_dataset'.format(args.data_name), 
            'dataset is none!')()
    print(class_names)
    # save latents diffusion features
    sd_feas = torch.load(f'./save/sdfeas_noise_{args.data_name}.pth')
    sd_text_feas = torch.load(f'./save/sdfeas_text_{args.data_name}.pth')
    sd_load_feas = {}
    sd_load_text_feas = {}
    for i in range(len(sd_feas)):
        sd_load_feas[i] = sd_feas[i].cpu().numpy()
        sd_load_text_feas[i] = sd_text_feas[i].cpu().numpy()
    # heterogeneous and long_tailed setting
    list_label2indices = classify_label(data_local_training, args.num_classes)
    _, list_label2indices_train_new = train_long_tail(copy.deepcopy(list_label2indices), args.num_classes,
                                                      args.imb_factor, args.imb_type)
    list_client2indices = clients_indices(copy.deepcopy(list_label2indices_train_new), args.num_classes,
                                          args.num_clients, args.non_iid_alpha, args.seed)
    original_dict_per_client = show_clients_data_distribution(data_local_training, list_client2indices,
                                                              args.num_classes)
    server_model = Global(num_classes=args.num_classes, device=args.device, args=args)
    total_clients = list(range(args.num_clients))
    # indices2data = Indices2Dataset(data_local_training)
    indices2data = diffu_Indices2Dataset(data_local_training, sd_load_feas, sd_load_text_feas)
    trained_acc = []

    for r in tqdm(range(1, args.num_rounds+1), desc='server-training'):
        global_params = server_model.download_params()
        online_clients = random_state.choice(total_clients, args.num_online_clients, replace=False)
        list_dicts_local_params = []
        list_nums_local_data = []

        # local training
        for client in online_clients:
            indices2data.load(list_client2indices[client])
            data_client = indices2data
            list_nums_local_data.append(len(data_client))
            local_model = Local(data_client=data_client, class_list=original_dict_per_client[client], 
                                text_names=class_names)
            # local update
            local_params = local_model.local_train(args, copy.deepcopy(global_params))
            list_dicts_local_params.append(copy.deepcopy(local_params))
        
        # aggregating local models with FedAvg use parameters
        round_params = server_model.fedavg_parameters(list_dicts_local_params, list_nums_local_data)
        # global eval
        one_re_train_acc = server_model.global_eval(round_params, data_global_test, args.batch_size_test)
        trained_acc.append(round(one_re_train_acc, 4))
        server_model.global_model.load_state_dict(copy.deepcopy(round_params))
        print("\n Acc: ", trained_acc, " Acc (MAX): ", max(trained_acc))
    print("\n Acc_List: ", trained_acc)
    print("\n Acc_Top1: ", max(trained_acc))
    print("\n Acc_Aveg: ", round(np.mean(trained_acc), 4))


if __name__ == '__main__':
    args = args_parser()
    torch.manual_seed(args.seed)  # cpu
    torch.cuda.manual_seed(args.seed)  # gpu
    np.random.seed(args.seed)  # numpy
    random.seed(args.seed)  # random and transforms
    torch.backends.cudnn.deterministic = True  # cudnn
    FedDifRC_param()
    # python3 main_FedDifRC.py --data_name cifar10 --num_classes 10 
    # python3 main_FedDifRC.py --data_name tiny --num_classes 200 