from .node import node

import torch
import torch.distributed as dist
from collections import OrderedDict
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors, _take_tensors
import threading
import queue
from torch.autograd import Variable
import time
from torch.distributed import reduce_op

class worker(node):
    def __init__(self, rank, servers, model, workers,
                    data_loader, criterion, num_batches,
                    num_epochs, cuda,  start_epoch=0,
                    test_loader=None, save_path=None):
        super(worker, self).__init__(rank=rank, servers=servers, model=model,
                                    num_batches=num_batches, num_epochs=num_epochs,
                                    cuda=cuda, workers=workers, criterion=criterion,
                                    start_epoch=start_epoch)
        self.data_loader = data_loader
        self.test_loader = test_loader
        self.save_path = save_path
        self.param_bars = None 
        self.param_to_server = {}
        self.params_dict = OrderedDict([(i, p) for i, p in enumerate(self.model.parameters())])
        self.params_by_server_rank = None
        self.grad_queues = {s: queue.Queue() for s in self.servers}
        self.server_threads = {s: None for s in self.servers}
        self.numgrads_by_server = {s: 0 for s in self.servers}

        self.sync_splits()
        self.start_daemon()

    def sync_splits(self):
        temp = torch.IntTensor(self.num_servers + 1)
        dist.broadcast(temp, src=0)
        self.param_bars = temp.tolist()

        self.params_by_server_rank = {self.servers[i]: sorted(range(self.param_bars[i], self.param_bars[i+1])) for i in range(self.num_servers)}

        for server_rank, indices in self.params_by_server_rank.items():
            for index in indices:
                self.param_to_server[index] = server_rank
        
        for i, p in enumerate(self.model.parameters()):
            if p.requires_grad:
                self.numgrads_by_server[self.param_to_server[i]] += 1

    def sync_buffers(self):
        for p in self.model._all_buffers():
            dist.send(p.data, dst=0)
   
    #regist grad's hook function
    def start_daemon(self):
        #push grad to server when backward to this parameter
        def grad_hook_fn(index):
            def put_(grad):
                dst = self.param_to_server[index]
                self.grad_queues[dst].put(grad.data)
            return put_

        for i, p in enumerate(self.model.parameters()):
            if p.requires_grad:
                p.register_hook(grad_hook_fn(i))

    def push_grads_pre(self):
        def server_thread_fn(s):
            for _ in range(self.numgrads_by_server[s]):
                grad_data = self.grad_queues[s].get()
                dist.send(grad_data, dst=s)
        for s in self.servers:
            t = threading.Thread(target = server_thread_fn, args=(s,))
            t.daemon = True
            t.start()
            self.server_threads[s] = t

    def push_grads_post(self):
        for t in self.server_threads.values():
            t.join()
            
    def pull_params(self):
        def pull_from_server(s):
            params_indices = self.params_by_server_rank[s]
            for index in params_indices:
                param = self.params_dict[index].data
                dist.recv(param, src=s)
        s_threads = []
        for s in self.servers:
            t = threading.Thread(target=pull_from_server, args=(s,))
            t.daemon=True
            t.start()
            s_threads.append(t)

        for t in s_threads:
            t.join()
    
    def train(self):
        self.pull_params()
        for epoch in range(self.start_epoch, self.num_epochs):
            dist.barrier()
            epoch_time = self.train_epoch(epoch)
            dist.barrier()
            self.pull_params()
            total, train_loss = self.experiment_epoch()

            dist.send(torch.FloatTensor([epoch_time]), dst=0)
            dist.send(torch.FloatTensor([train_loss]), dst=0)
            dist.send(torch.IntTensor([total]), dst=0)
            self.sync_buffers()


    def train_epoch(self, epoch):
        self.model.train()
        self.data_loader.sampler.set_epoch(epoch)
        total = 0
        st = time.time()
        for batch_idx, (inputs, targets) in enumerate(self.data_loader):
            cal_st = time.time()
            if self.cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            self.model.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            total += len(inputs)
            #push grads to servers
            self.push_grads_pre()
            loss.backward()
            cal_t = time.time() - cal_st
            comm_st = time.time()
            self.push_grads_post()
            #pull params from servers
            self.pull_params()
            comm_t = time.time() - comm_st
            batch_t =  time.time() - cal_st
        return time.time() - st

    def experiment_epoch(self):
        self.model.eval()
        total = 0
        train_loss = 0.0
        
        for batch_idx, (inputs, targets) in enumerate(self.data_loader):
            if self.cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            total += len(inputs)
            train_loss += (len(inputs) * loss.item())

        return (total, train_loss)

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

