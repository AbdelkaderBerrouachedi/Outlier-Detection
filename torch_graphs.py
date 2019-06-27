import torch



class TorchGraph:

    def __init__(self, num_attrib, cuda=False):
        self.cuda = cuda
        self.num_attrib = num_attrib
        self.adj_mat = None
        self.nodes = None


    def add_node(self, attrib):
        if self.nodes is None:# or node_id not in self.nodes[:, 1]:
            self.nodes = torch.tensor(attrib)

