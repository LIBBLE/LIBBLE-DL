import torch
import torch.distributed as dist
from torch.distributed import reduce_op
from torch.autograd import Variable
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors, _take_tensors
import time
import queue
import threading
from .node import node

class master(node):
    def __init__(self, rank, num_workers, model, data_loader,
                num_epochs, criterion, cuda, optim_fn,
                bucket_comm=True, test_loader=None, 
                save_path=None, start_epoch=0, adjust = []):
        super(master, self).__init__(rank, num_workers, model,
                                data_loader, num_epochs, criterion,
                                cuda, bucket_comm, start_epoch)
        self.test_loader = test_loader
        self.optimizer = optim_fn(self.model.parameters())
        self.epoch_time = 0
        self.num_batches = len(self.data_loader)
        self.save_path = save_path
        self.adjust = adjust
        for i,p in enumerate(self.model.parameters()):
            if p.requires_grad:
                if self.cuda:
                    p.grad = Variable(torch.cuda.FloatTensor(p.data.size()))
                else:
                    p.grad = Variable(torch.FloatTensor(p.data.size()))
                p.retain_grad()

    def sync_params_bucket(self):
        params = [p.data for p in list(self.model.parameters())]
        for tensors in _take_tensors(params, self.mpi_size):
            flat_tensors = _flatten_dense_tensors(tensors)
            dist.broadcast(flat_tensors, src=0)

    def sync_grads_bucket(self):
        grads = [p.grad.data for p in list(self.model.parameters()) if p.requires_grad]
        for tensors in _take_tensors(grads, self.mpi_size):
            flat_tensors = _flatten_dense_tensors(tensors)
            dist.reduce(flat_tensors, dst=0, op=reduce_op.SUM)
            flat_tensors.div_(self.num_workers)
            for tensor, synced in zip(tensors,_unflatten_dense_tensors(flat_tensors, tensors)):
                tensor.copy_(synced)

    def sync_buffers_bucket(self):
        buffers = [p.data for p in list(self.model._all_buffers())]
        for tensors in _take_tensors(buffers, self.mpi_size):
            flat_tensors = _flatten_dense_tensors(tensors)
            flat_tensors.zero_()
            dist.reduce(flat_tensors, dst=0, op=reduce_op.SUM)
            flat_tensors.div_(self.num_workers)
            for tensor, synced in zip(tensors,_unflatten_dense_tensors(flat_tensors, tensors)):
                tensor.copy_(synced)

    def sync_params(self):
        if self.bucket_comm:
            self.sync_params_bucket()
        else:
            self.sync_params_layer()

    def sync_params_layer(self):
        for p in self.model.parameters():
            dist.broadcast(p.data, src=0)

    def sync_grads_layer(self):
        for p in self.model.parameters():
            if p.requires_grad:
                dist.reduce(p.grad.data, dst=0, op=reduce_op.SUM)
                p.grad.data.div_(self.num_workers)

    def sync_buffers_layer(self):
        for p in self.model._all_buffers():
            p.data.zero_()
            dist.reduce(p.data, dst=0, op=reduce_op.SUM)
            p.data.div_(self.num_workers)

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
        for epoch in range(self.start_epoch, self.num_epochs):
            if epoch in self.adjust:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= 0.1
            dist.barrier()
#           train
            for batch_idx, _ in enumerate(self.data_loader):
                self.optimizer.zero_grad()
                self.sync_grads()
                self.optimizer.step()
                self.sync_params()
            dist.barrier()
#           train loss
            for _ in self.data_loader:
                pass
            epoch_time = torch.zeros(1)
            loss_and_num = torch.zeros(2)
            dist.reduce(epoch_time, dst=0, op=reduce_op.MAX)    
            dist.reduce(loss_and_num, dst=0, op=reduce_op.SUM)
            train_loss = loss_and_num[0].item()
            total = loss_and_num[1].item()

            train_loss /= total
            epoch_time = epoch_time.item()

            print_str = 'Epoch:{}\tTime:{}\tTrain Loss:{}'.format(epoch, epoch_time, train_loss)
            self.sync_buffers()
            if self.test_loader != None:
                test_loss, test_error = self.test_epoch()
                print_str += '\tTest Loss:{}\tTest Error:{}'.format(test_loss, test_error)
            print(print_str)
            if self.save_path != None:
                torch.save(self.model.state_dict(), self.save_path)
                print('model saved')


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
        test_loss /= len(self.test_loader.dataset)
        test_error = 1 - correct / len(self.test_loader.dataset)

        return(test_loss, test_error)
