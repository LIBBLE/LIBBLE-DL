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
from torch.distributed import reduce_op
from torch.autograd import Variable
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors, _take_tensors
import time
import queue
import threading
from .node import node

class worker(node):
    def __init__(self, rank, num_workers, model, data_loader,
                    num_epochs, criterion, cuda,
                    bucket_comm=True, start_epoch=0, 
                    test_loader=None, save_path=None):
        super(worker, self).__init__(rank, num_workers, model,
                                data_loader, num_epochs, criterion,
                                cuda, bucket_comm, start_epoch)
        self.test_loader = test_loader
        self.save_path = save_path
        self.epoch_time = 0

    def sync_params_bucket(self):
        params = [p.data for p in list(self.model.parameters())]
        for tensors in _take_tensors(params, self.mpi_size):
            flat_tensors = _flatten_dense_tensors(tensors)
            dist.broadcast(flat_tensors, src=0)
            for tensor, synced in zip(tensors,_unflatten_dense_tensors(flat_tensors, tensors)):
                tensor.copy_(synced)

    def sync_grads_bucket(self):
        grads = [p.grad.data for p in list(self.model.parameters()) if p.requires_grad]
        for tensors in _take_tensors(grads, self.mpi_size):
            flat_tensors = _flatten_dense_tensors(tensors)
            dist.reduce(flat_tensors, dst=0, op=reduce_op.SUM)

    def sync_buffers_bucket(self):
        buffers = [p.data for p in list(self.model._all_buffers())]
        for tensors in _take_tensors(buffers, self.mpi_size):
            flat_tensors = _flatten_dense_tensors(tensors)
            dist.reduce(flat_tensors, dst=0, op=reduce_op.SUM)

    def sync_params_layer(self):
        for p in self.model.parameters():
            dist.broadcast(p.data, src=0)

    def sync_grads_layer(self):
        for p in self.model.parameters():
            if p.requires_grad:
                dist.reduce(p.grad.data, dst=0, op=reduce_op.SUM)

    def sync_buffers_layer(self):
        for p in self.model._all_buffers():
            dist.reduce(p.data, dst=0, op=reduce_op.SUM)

    def sync_params(self):
        if self.bucket_comm:
            self.sync_params_bucket()
        else:
            self.sync_params_layer()

    def sync_grads(self):
        if self.bucket_comm:
            self.sync_grads_bucket()
        else:
            self.sync_grads_layer()

    def sync_buffers(self):
        if self.bucket_comm:
            self.sync_buffers_bucket()
        else:
            self.sync_buffers_layer()

    def train(self):
        self.sync_params()
        for epoch in range(self.num_epochs):
            dist.barrier()
            self.train_epoch(epoch)
            dist.barrier()
            self.experiment_epoch(epoch)
            self.sync_buffers()

    def train_epoch(self, epoch):
        self.model.train()
        self.data_loader.sampler.set_epoch(epoch)
        start_time = time.time()
        total = 0
        for batch_idx, (inputs, targets) in enumerate(self.data_loader):
            st = time.time()
            if self.cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            self.model.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            total += len(inputs)
            loss.backward()
            comm_st = time.time()
            self.sync_grads()
            self.sync_params()
        self.epoch_time = time.time() - start_time


    def experiment_epoch(self,epoch):
        train_loss = 0.0
        total = 0
        self.model.eval()
        for batch_idx, (inputs, targets) in enumerate(self.data_loader):
            if self.cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            with torch.no_grad():
                inputs, targets = Variable(inputs), Variable(targets)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            train_loss += (loss.item() * len(inputs))
            total += len(inputs)
        epoch_time = torch.FloatTensor([self.epoch_time])
        loss_and_num = torch.FloatTensor([train_loss, total])
        dist.reduce(epoch_time, dst=0, op=reduce_op.MAX)
        dist.reduce(loss_and_num, dst=0, op=reduce_op.SUM)

    def test_epoch(self):
        self.model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        for inputs, targets in self.test_loader:
            if self.cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            with torch.no_grad():
                inputs, targets = Variable(inputs), Variable(targets)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            test_loss += (loss.item() * len(inputs))
            total += len(inputs)
            pred = outputs.data.max(1, keepdim=True)[1]
            correct += pred.eq(targets.data.view_as(pred)).long().cpu().sum().item()
        
        test_loss /= total
        test_error = 1 - correct / total
        
        return(test_loss, test_error)

