import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.utils.data.distributed

from ...src.ring.node import node
from .resnet import *
import time
import argparse

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='RING ALL_REDUCE CIFAR10 Example')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--num_epochs', type=int, default=160)
    parser.add_argument('--no_cuda', type=bool, default=False)
    parser.add_argument('--test_batch_size', type=int, default=128)
    parser.add_argument('--bucket_comm', type=bool, default=True)
    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument('--start_epoch', type=int, default=0)
    args = parser.parse_args()
    
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.backends.cudnn.enabled = True

    dist.init_process_group('mpi')
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    #define net
    net = resnet20()
    
    if args.start_epoch != 0 and args.save_path != None:
        net.load_state_dict(torch.load(args.save_path))
    
    if args.cuda:
        net = net.cuda()

    #define loss function
    criterion = nn.CrossEntropyLoss

    #define optimize function
    def optim_fnc(learnrate, momentum, weight_decay):
        def fn(parameters):
            return optim.SGD(parameters, lr=learnrate, momentum=momentum, weight_decay=weight_decay)
        return fn
    opfn = optim_fnc(args.lr, args.momentum, args.weight_decay)
    adjust = [80,120]

    #define distributed dataset loader
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    loader_kwargs = {}
    data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    dist_sampler = torch.utils.data.distributed.DistributedSampler(data,
                    num_replicas = world_size,  rank = rank)
    dist_loader = torch.utils.data.DataLoader(data, sampler = dist_sampler, 
                    batch_size = args.batch_size//world_size, shuffle=False, **loader_kwargs)
        
    test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data', train=False, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                ])),
            batch_size = args.test_batch_size, shuffle=False, **loader_kwargs)
    #arguments
    node_kwargs = {'rank': rank, 'world_size': world_size, 'model': net, 
                'data_loader': dist_loader, 'num_epochs': args.num_epochs, 
                'criterion': criterion, 'cuda': args.cuda, 'optim_fn': opfn, 'bucket_comm': args.bucket_comm,
                'start_epoch': args.start_epoch, 'test_loader': test_loader ,'adjust': adjust}

    #build node

    if rank == 0:
        n = node(save_path=args.save_path, **node_kwargs)
    else:
        n = node(**node_kwargs) 
    n.train()
