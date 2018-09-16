from .node import node

import threading
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors, _take_tensors

import torch
import torch.distributed as dist
from torch.autograd import Variable
import time
from collections import OrderedDict
from torch.distributed import reduce_op

class coordinator(node):
    def __init__(self, rank, servers, workers, criterion,
                    model, num_batches, num_epochs, 
                    cuda, data_loader=None, test_loader=None,
                    start_epoch=0, save_path=None):
        super(coordinator, self).__init__(rank=rank, servers=servers,
                    model=model, num_batches=num_batches, num_epochs=num_epochs,
                    cuda=cuda, workers=workers, criterion=criterion,
                    start_epoch=start_epoch)
    
        self.test_loader = test_loader
        self.data_loader = data_loader
        self.worker_thread = []
        self.save_path = save_path
        self.params_dict = OrderedDict([(i, p) for i, p in enumerate(self.model.parameters())])

        self.has_update = True
        self.no_changes = {s: False for s in self.servers}
        self.sync_splits()

    def sync_splits(self):
        #split the paramters to servers by the size
        params_sizes = []
        for p in self.model.parameters():
            if p.is_sparse:
                indices = p._indices()
                values = p._values()
                size = indices.numel() * indices.element_size() + values.numel() * values.element_size()
            else:
                size = p.numel() * p.element_size()
            params_sizes.append(size)
        self.param_bars = [0] + self.partition_list(params_sizes, self.num_servers) + [self.num_params]
        self.params_by_server_rank = {self.servers[i]: sorted(range(self.param_bars[i], self.param_bars[i+1])) for i in range(self.num_servers)}
        
        dist.broadcast(torch.IntTensor(self.param_bars), src = 0)
    
    def sync_params(self):
        def pull_from_server(s):
            dist.send(torch.rand(1), dst=s)
            params_indices = self.params_by_server_rank[s]
            for index in params_indices:
                param = self.params_dict[index].data
                dist.recv(param, src=s)

        server_threads = {}
        for s in self.servers:
            t = threading.Thread(target=pull_from_server, args=(s,))
            t.daemon=True
            t.start()
            server_threads[s] = t

        for t in server_threads.values():
            t.join()

    def sync_buffers(self):
        for p in self.model._all_buffers():
            p.data.zero_()
            recv_buff = torch.FloatTensor(p.data.size()).cuda()
            for w in self.workers:
                dist.recv(recv_buff, src=w)
                p.data.add_(recv_buff)
            p.data.div_(self.num_workers)    

    def train(self):
        for epoch in range(self.start_epoch, self.num_epochs):
            dist.barrier()
            dist.barrier()

            epoch_time = {w : torch.FloatTensor([0]) for w in self.workers}
            train_loss = {w : torch.FloatTensor([0]) for w in self.workers}
            total = {w: torch.IntTensor([0]) for w in self.workers}
            
            for w in self.workers:
                dist.recv(epoch_time[w], src=w)
                dist.recv(train_loss[w], src=w)
                dist.recv(total[w], src=w)

            epoch_time = max([t.item() for t in epoch_time.values()])
            train_loss = sum([l.item() for l in train_loss.values()])
            total = sum([t.item() for t in total.values()])

            print_str = 'Epoch:{}\tTime:{}\tTrain Loss:{}'.format(epoch, epoch_time, train_loss / total)
            
            self.sync_params()
            self.sync_buffers()
            if self.test_loader != None:
                test_loss, test_error = self.test_epoch()
                print_str += '\tTest Loss:{}\tTest Error:{}'.format(test_loss, test_error)
            print(print_str)
            if self.save_path != None:
                torch.save(self.model.state_dict(), self.save_path)
                print('#model saved')

    def test_epoch(self):
        correct = 0
        test_loss = 0.0 
        self.model.eval()
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

        return (test_loss, test_error)

    def partition_list(self, a, k):
        assert(1 <= k <= len(a))
        if k == 1: 
            return []
        if k == len(a): 
            return list(range(1,len(a)))
        partition_between = []
        for i in range(k-1):
            partition_between.append(int((i+1)*len(a)/k))
        average_height = float(sum(a))/k
        best_score = None
        best_partitions = None
        count = 0
        no_improvements_count = 0
        while True:
            partitions = []
            index = 0
            for div in partition_between:
                partitions.append(a[index:div])
                index = div
            partitions.append(a[index:])
            worst_height_diff = 0
            worst_partition_index = -1
            for p in partitions:
                height_diff = average_height - sum(p)
                if abs(height_diff) > abs(worst_height_diff):
                    worst_height_diff = height_diff
                    worst_partition_index = partitions.index(p)
            if best_score is None or abs(worst_height_diff) < best_score:
                best_score = abs(worst_height_diff)
                best_partitions = partitions
                no_improvements_count = 0
            else:
                no_improvements_count += 1
            if worst_height_diff == 0 or no_improvements_count > 5 or count > 100:
                return partition_between
            count += 1
            if worst_partition_index == 0:   
                if worst_height_diff < 0: partition_between[0] -= 1 
                else: partition_between[0] += 1 
            elif worst_partition_index == len(partitions)-1: 
                if worst_height_diff < 0: partition_between[-1] += 1 
                else: partition_between[-1] -= 1 
            else: 
                left_bound = worst_partition_index - 1 
                right_bound = worst_partition_index 
                if worst_height_diff < 0: 
                    if sum(partitions[worst_partition_index-1]) > sum(partitions[worst_partition_index+1]):
                        partition_between[right_bound] -= 1
                    else:
                        partition_between[left_bound] += 1
                else:
                    if sum(partitions[worst_partition_index-1]) > sum(partitions[worst_partition_index+1]):
                        partition_between[left_bound] -= 1
                    else:
                        partition_between[right_bound] += 1
