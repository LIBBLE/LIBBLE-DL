import torch
import torch.distributed as dist
import threading
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors, _take_tensors


def new_all_reduce(data, cuda=False, threshold=3):
    ts = 1
    for s in data.size():
        ts *= s
    if ts < threshold * dist.get_world_size():
        dist.all_reduce(data)
    else:
        if data.dim() > 1:
            flat_data = data.contiguous().view(-1)
            ring_all_reduce(flat_data, cuda=cuda)
        else:
            ring_all_reduce(data, cuda=cuda)

def ring_all_reduce(data, cuda=False):
    #distributed environment
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    #define neighborhood
    left = ((rank - 1) + world_size) % world_size
    right = (rank + 1) % world_size

    #seg_sizes: split data to segments
    length = len(data)
    assert(length >= world_size)
    seg_size = int(length / world_size)
    residual = length % world_size
    seg_sizes = [seg_size] * world_size
    for i in range(residual):
        seg_sizes[i] += 1
    #seg_ends: ith segment's ending index
    seg_ends = [0] * world_size
    seg_ends[0] = seg_sizes[0]
    for i in range(1, world_size):
        seg_ends[i] = seg_sizes[i] + seg_ends[i-1]
    assert(seg_ends[-1] == length)
    
    #define communication buffer
    if not cuda:
        recv_buff = torch.FloatTensor(length)
    else:
        recv_buff = torch.FloatTensor(length).cuda()
    
    def send_fn(d, r):
        dist.send(d, dst=r)

    #first loop
    dist.barrier()
    for i in range(world_size - 1):
        send_index = (rank - i + world_size) % world_size
        recv_index = (rank - i - 1 + world_size) % world_size
        send_start = seg_ends[send_index] - seg_sizes[send_index]
        recv_start = seg_ends[recv_index] - seg_sizes[recv_index]

        send_req = dist.isend(data[send_start:seg_ends[send_index]], right)
        #send_thread = threading.Thread(target=send_fn, args=(data[send_start:seg_ends[send_index]], right))
        #send_thread.daemon = True
        #send_thread.start()

        dist.recv(recv_buff[recv_start:seg_ends[recv_index]], src=left)        
        data[recv_start:seg_ends[recv_index]].add_(recv_buff[recv_start:seg_ends[recv_index]])
        send_req.wait()
        dist.barrier()

    #second loop
    for i in range(world_size - 1):
        send_index = (rank - i + 1 + world_size) % world_size
        recv_index = (rank - i + world_size) % world_size
        send_start = seg_ends[send_index] - seg_sizes[send_index]
        recv_start = seg_ends[recv_index] - seg_sizes[recv_index]

        send_req = dist.isend(data[send_start:seg_ends[send_index]], right)
        #send_thread = threading.Thread(target=send_fn, args=(data[send_start:seg_ends[send_index]], right))
        #send_thread.daemon = True
        #send_thread.start()

        dist.recv(data[recv_start:seg_ends[recv_index]], src=left)
        send_req.wait()
        dist.barrier()
    
    del recv_buff

