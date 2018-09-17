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
        
from .node import node

import torch
import torch.distributed as dist
from torch.autograd import Variable
from torch._utils import _flatten_dense_tensors, \
            _unflatten_dense_tensors,_take_tensors

import time
import threading
import queue
from collections import OrderedDict
from torch.distributed import reduce_op

class server(node):
    def __init__(self, rank, servers, workers, model, 
                num_batches, num_epochs, cuda, 
                optim_fn, time_window, criterion, start_epoch=0,adjust=[]):
        super(server, self).__init__(rank=rank, servers=servers, model=model,
                                    num_batches=num_batches, num_epochs=num_epochs,
                                    cuda=cuda, workers=workers, criterion=criterion,
                                    start_epoch=start_epoch)
        self.optim_fn = optim_fn
        self.time_window = time_window

        self.params_dict = None

        self.grad_queue = {}
        self.optims = {}

        self.grad_thread = {}

        self.worker_thread = []
        self.worker_time = {w: 0 for w in self.workers}
        self.processed_grads = {w: 0 for w in self.workers}
    
        self.sync_splits()

        self.start_daemon()

        self.needed_processed_grads = sum([1 for i,p in self.params_dict.items() if p.requires_grad])
        self.adjust = adjust

    def wait_time_window(self, w, period=0.01):
        while True:
            if self.worker_time[w] - min(self.worker_time.values()) <= self.time_window:
                return True
            time.sleep(period)

    def wait_process_grad(self, w, period = 0.01):
        while True:
            if self.processed_grads[w] == self.needed_processed_grads:
                return True
            time.sleep(period)

    def sync_splits(self):
        temp = torch.IntTensor(self.num_servers + 1)
        dist.broadcast(temp, src=0)
        param_bars = temp.tolist()

        params_by_server_rank = {self.servers[i]: sorted(range(param_bars[i], param_bars[i+1])) for i in range(self.num_servers)}
        self.local_param_indices = params_by_server_rank[self.rank]
        self.params_dict = OrderedDict([(i, p) for i, p in enumerate(self.model.parameters()) if i in self.local_param_indices])

        for i, p in enumerate(self.model.parameters()):
            if i in self.local_param_indices and p.requires_grad:
                self.optims[i] = self.optim_fn([self.params_dict[i]])



    
    def start_daemon(self):
        def update_param(i):
            for _ in range((self.num_epochs-self.start_epoch) * self.num_batches * self.num_workers):
                w, tmp = self.grad_queue[i].get(True)
                self.params_dict[i].grad = Variable(tmp)
                self.optims[i].step()
                self.processed_grads[w] += 1
                del tmp
                
        for i, p in self.params_dict.items():
            if p.requires_grad:
                q = queue.Queue()
                self.grad_queue[i] = q
                t = threading.Thread(target=update_param, args=(i,))
                t.daemon = True
                t.start()
                self.grad_thread[i] = t
    

    def train(self):
        #thread functions with workers
        def respond_pull(w):
            for param in self.params_dict.values():
                dist.send(param.data, dst=w)
        
        def respond_push(w):
            for i, p in reversed(self.params_dict.items()):
                if p.requires_grad:
                    if self.cuda:
                        buf = torch.FloatTensor(p.data.size()).cuda()
                    else:
                        buf = torch.FloatTensor(p.data.size())
                    dist.recv(buf, w)
                    buf.div_(self.num_workers)
                    self.grad_queue[i].put((w, buf))

        def respond_fn(index):
            w = self.workers[index]
            for batch_idx in range(self.num_batches):
                self.wait_time_window(w)
                respond_push(w)
                self.wait_process_grad(w)
                self.processed_grads[w] = 0
                respond_pull(w)
                self.worker_time[w] += 1
                self.wait_time_window(w)
        
        #threads with worker
        wts = []
        for w in self.workers:
            t = threading.Thread(target=respond_pull, args=(w,))
            t.daemon = True
            t.start()
            wts.append(t)
        for t in wts:
            t.join()
        for epoch in range(self.start_epoch, self.num_epochs):
            if epoch in self.adjust:
                for i, p in enumerate(self.model.parameters()):
                    if i in self.local_param_indices and p.requires_grad:
                        for param_group in self.optims[i].param_groups:
                            param_group['lr'] *= 0.1
            dist.barrier()
            for index in range(self.num_workers):
                t = threading.Thread(target=respond_fn, args=(index,))
                t.daemon = True
                t.start()
                self.worker_thread.append(t)
            for t in self.worker_thread:
                t.join()

            dist.barrier()
            wts = []
            for w in self.workers:
                t = threading.Thread(target=respond_pull, args=(w,))
                t.daemon = True
                t.start()
                wts.append(t)
            for t in wts:
                t.join()

            #thread function with coordinator:
                

            coordinator_thread = threading.Thread(target=respond_pull, args=(0,))
            coordinator_thread.daemon = True
            coordinator_thread.start()
            coordinator_thread.join()
