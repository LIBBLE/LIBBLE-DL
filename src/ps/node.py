import torch.distributed as dist
class node:
    def __init__(self, rank, servers, model, num_batches, num_epochs, workers, criterion, cuda=False, start_epoch=0):
        self.rank = rank
        self.servers = servers
        self.workers = workers
        self.model = model
        self.num_params = sum(1 for _ in self.model.parameters())
        self.num_servers = len(self.servers)
        self.num_workers = len(self.workers)
        assert(dist.get_world_size() == (1 + self.num_servers + self.num_workers))
        self.num_batches = num_batches
        self.num_epochs = num_epochs
        MB = 1024 * 1024
        self.mpi_size = 10 * MB
        self.criterion = criterion(size_average = True)
        self.cuda = cuda
        self.start_epoch = start_epoch
