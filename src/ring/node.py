import torch
import torch.distributed as dist
from torch.distributed import reduce_op
from .ring_allreduce import new_all_reduce
from torch.autograd import Variable
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors, _take_tensors
import time
import queue
import threading

class node:
    def __init__(self, rank, world_size, model, data_loader,
                    num_epochs, criterion, optim_fn, cuda,
                    test_loader=None, bucket_comm=False, save_path=None,
                    start_epoch=0, adjust = []):
        self.rank = rank
        self.world_size = world_size
        assert dist.get_world_size() == self.world_size
        self.model = model
        self.data_loader = data_loader
        self.test_loader = test_loader
        self.num_epochs = num_epochs
        self.criterion = criterion(size_average=True)
        self.optim_fn = optim_fn
        self.save_path = save_path
        MB = 1024 * 1024
        self.mpi_size = 50 * MB
        self.cuda = cuda
        self.optimizer = self.optim_fn(self.model.parameters())
        self.epoch_time = 0
        self.bucket_comm = bucket_comm
        self.start_epoch = start_epoch
        self.adjust = adjust

    def sync_grads_bucket(self):
        grads = [p.grad.data for p in list(self.model.parameters()) if p.requires_grad]
        for tensors in _take_tensors(grads, self.mpi_size):
            flat_tensors = _flatten_dense_tensors(tensors)
            new_all_reduce(flat_tensors, cuda=self.cuda)
            flat_tensors.div_(self.world_size)
            for tensor, synced in zip(tensors,_unflatten_dense_tensors(flat_tensors, tensors)):
                tensor.copy_(synced)

    def sync_grads_layer(self):
        for p in self.model.parameters():
            if p.requires_grad:
                new_all_reduce(p.grad.data, cuda=self.cuda)
        for p in self.model.parameters():
            if p.requires_grad:
                p.grad.div_(self.world_size)
    
    def sync_params_bucket(self):
        params = [p.data for p in list(self.model.parameters())]
        for tensors in _take_tensors(params, self.mpi_size):
            flat_tensors = _flatten_dense_tensors(tensors)
            dist.broadcast(flat_tensors, src=0)
            if self.rank != 0:
                for tensor, synced in zip(tensors,_unflatten_dense_tensors(flat_tensors, tensors)):
                    tensor.copy_(synced)
    
    def sync_params_layer(self):
        for p in self.model.parameters():
            dist.broadcast(p.data, src=0)

    def sync_grads(self):
        if self.bucket_comm:
            self.sync_grads_bucket()
        else:
            self.sync_grads_layer()

    def sync_params(self):
        if self.bucket_comm:
            self.sync_params_bucket()
        else:
            self.sync_params_layer()

    def train(self):
        self.sync_params()
        for epoch in range(self.start_epoch, self.num_epochs):
            if epoch in self.adjust:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= 0.1
            dist.barrier()
            self.train_epoch(epoch)
            train_loss, epoch_time = self.experiment_epoch()
            print_str = 'Epoch:{}\tTime:{}\tTrain Loss:{}'.format(epoch, epoch_time, train_loss)
            if self.test_loader != None:
                test_loss, test_error = self.test_epoch()
                print_str += '\tTest Loss:{}\tTest Error:{}'.format(test_loss, test_error)
            if self.rank == 0:
                print(print_str)
            if self.save_path != None:
                torch.save(self.model.state_dict(), self.save_path)
                print('model saved')

    def train_epoch(self, epoch):
        self.model.train()
        self.data_loader.sampler.set_epoch(epoch)
        start_time = time.time()
        total = 0
        for batch_idx, (inputs, targets) in enumerate(self.data_loader):
            cal_st = time.time()
            if self.cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            total += len(inputs)
            loss.backward()
            cal_t = time.time() - cal_st
            com_st = time.time()
            dist.barrier()
            self.sync_grads()
            comm_t = time.time() - com_st
            self.optimizer.step()
            batch_t =  time.time() - cal_st
            #print(cal_t, comm_t, batch_t)
            del inputs, targets
        self.epoch_time = time.time() - start_time


    def experiment_epoch(self):
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
            del inputs, targets
        epoch_time = torch.FloatTensor([self.epoch_time])
        train_loss = torch.FloatTensor([train_loss])
        total = torch.FloatTensor([total])
        dist.reduce(epoch_time, dst=0, op=reduce_op.MAX)
        dist.reduce(train_loss, dst=0, op=reduce_op.SUM)
        dist.reduce(total, dst=0, op=reduce_op.SUM)

        train_loss = (train_loss.item() / total.item())
        epoch_time = epoch_time.item()

        return (train_loss, epoch_time)

    def test_epoch(self):
        self.model.eval()
        test_loss = 0.0
        correct = 0
        for inputs, targets in self.test_loader:
            if self.cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            with torch.no_grad():
                inputs, targets = Variable(inputs), Variable(targets)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            test_loss += (loss.item() * len(inputs))
            pred = outputs.data.max(1, keepdim=True)[1]
            correct += pred.eq(targets.data.view_as(pred)).long().cpu().sum().item()
            del inputs, targets
        test_loss /= len(self.test_loader.dataset)
        test_error = 1 - correct / len(self.test_loader.dataset)

        return (test_loss, test_error)
