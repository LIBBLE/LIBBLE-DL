import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.utils.data.distributed

from ...src.mapreduce.master import master
from ...src.mapreduce.worker import worker
import time
import argparse
from .resnet import *

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='MAP REDUCE CIFAR10 Example')
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
    torch.backends.cudnn.enabled= True

    dist.init_process_group('mpi')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    num_workers = world_size - 1

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


    #define distributed dataset loader
    loader_kwargs = {}
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    data = datasets.CIFAR10('./data', train=True, download=True, transform=transform_train)
    dist_sampler = torch.utils.data.distributed.DistributedSampler(data,
                    num_replicas = num_workers,  rank = max(0, rank-1))
    dist_loader = torch.utils.data.DataLoader(data, sampler = dist_sampler, 
                    batch_size = args.batch_size//num_workers, shuffle=True, **loader_kwargs)

    test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data', train=False, download=False,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                ])),
            batch_size = args.test_batch_size, shuffle=False, **loader_kwargs)
    #arguments
    node_kwargs = {'rank': rank, 'num_workers': num_workers, 'model': net, 
                'data_loader': dist_loader, 'num_epochs': args.num_epochs, 
                'criterion': criterion, 'cuda': args.cuda,
                'bucket_comm': args.bucket_comm, 'start_epoch': args.start_epoch}

    #build node
    if rank == 0:
        n = master(save_path=args.save_path, optim_fn=opfn, adjust =[80, 120],test_loader=test_loader, **node_kwargs)
    else:
        n = worker(save_path=args.save_path,test_loader=test_loader, **node_kwargs) 

    n.train()
