'''
        * Copyright (c) 2017 LIBBLE team supervised by Dr. Wu-Jun LI at Nanjing University.
        * All Rights Reserved.
        * Licensed under the Apache License, Version 2.0 (the "License");
        * you may not use this file except in compliance with the License.
        * You may obtain a copy of the License at
        *
        * http://www.apache.org/licenses/LICENSE-2.0
        *
        * Unless required by applicable law or agreed to in writing, software
        * distributed under the License is distributed on an "AS IS" BASIS,
        * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
        * See the License for the specific language governing permissions and
        * limitations under the License. '''

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from torchvision import datasets, transforms
from torch.autograd import Variable

from ...dispytorch.ps.coordinator import coordinator
from ...dispytorch.ps.server import server
from ...dispytorch.ps.worker import worker

import time
import argparse
from .resnet import *

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Parameter Server CIFAR10 Example')
    parser.add_argument('--num_servers', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--num_epochs', type=int, default=160)
    parser.add_argument('--test_batch_size', type=int, default=128)
    parser.add_argument('--time_window', type=int, default=50)
    parser.add_argument('--no_cuda', type=bool, default=False)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--save_path', type=str, default=None)

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.backends.cudnn.enabled = True

    dist.init_process_group('mpi')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert(args.num_servers + args.num_workers + 1 == world_size)

    #allocate rank of servers and workers
    servers = list(range(1, args.num_servers + 1))
    workers = list(range(args.num_servers+1, world_size))
    
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

    data = datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)
    dist_sampler = torch.utils.data.distributed.DistributedSampler(data,
                        num_replicas = args.num_workers,  
                        rank = max(0, rank - args.num_servers - 1))
    dist_loader = torch.utils.data.DataLoader(data, 
                        sampler = dist_sampler, batch_size = args.batch_size//args.num_workers, 
                        shuffle=False, **loader_kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=False, download=False,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                ])),
        batch_size = args.test_batch_size, shuffle=False)
    #arguments
    kwargs = {'rank': rank, 'servers': servers, 'model': net, 'workers': workers,
                'num_batches': len(dist_loader), 'num_epochs': args.num_epochs, 
                'cuda': args.cuda , 'criterion': criterion, 'start_epoch':args.start_epoch}

    #coordinator
    if rank == 0:
        n = coordinator(save_path=args.save_path,test_loader=test_loader, **kwargs)
    #server
    elif 0 < rank <= args.num_servers:
        n = server(optim_fn=opfn, time_window=args.time_window, adjust =[80, 120], **kwargs)
    #worker
    else:
        n = worker(save_path=args.save_path,data_loader=dist_loader, test_loader=None,  **kwargs) 
    
    n.train()
