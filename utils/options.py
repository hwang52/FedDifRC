import argparse
import os
from utils.param_aug import ParamDiffAug


def args_parser():
    parser = argparse.ArgumentParser()
    path_dir = os.path.dirname(__file__)
    parser.add_argument('--data_name', type=str, default='cifar10')
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--in_channels', type=int, default=3)
    parser.add_argument('--num_clients', type=int, default=10) # all clients
    parser.add_argument('--num_online_clients', type=int, default=5) # clients
    parser.add_argument('--num_rounds', type=int, default=50)
    parser.add_argument('--server_lr', type=float, default=1.0)
    parser.add_argument('--num_epochs_local_training', type=int, default=10) # epochs
    parser.add_argument('--batch_size_local_training', type=int, default=64)
    parser.add_argument('--batch_size_test', type=int, default=100)
    parser.add_argument('--lr_local_training', type=float, default=0.01)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--non_iid_alpha', type=float, default=0.5) # noniid
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--imb_type', default="exp", type=str, help='imbalance type')
    parser.add_argument('--imb_factor', default=1.0, type=float, help='imbalance factor') # long-tailed
    parser.add_argument('--save_path', type=str, default=os.path.join(path_dir, 'result/'))
    parser.add_argument('--method', type=str, default='DSA', help='DC/DSA')
    parser.add_argument('--mu', type=float, default=0.01) # FedProx
    parser.add_argument('--init_belta', type=float, default=0.97) # FedAvgM
    parser.add_argument('--diffu_size', type=int, default=224)
    parser.add_argument('--neg_num', type=int, default=10)
    parser.add_argument('--infonce_t', type=float, default=0.04)
    parser.add_argument('--timestep', type=int, default=999)
    
    args = parser.parse_args()

    return args