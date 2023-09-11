import dgl
import torch

from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset

class CitationDataset(torch.utils.data.Dataset):
    
    def __init__(self, name):
        super().__init__()
        start = time.time()
        print("[I] Loading dataset %s..." % (name))
        self.name = name
        dataset = None
        if name == 'cora':
            dataset = CoraGraphDataset()
        elif name == 'citeseer':
            dataset = CiteseerGraphDataset()
        elif name == 'pubmed':
            dataset = PubmedGraphDataset()
        self.graph = dataset[0]
        self.num_Class = dataset.num_classes
        self.labels = self.graph.ndata['label']
        self.train_mask = self.graph.ndata['train_mask']
        self.val_mask = self.graph.ndata['val_mask']
        self.test_mask = self.graph.ndata['test_mask']

        print("[I] Finished loading.")
        print("[I] Data load time: {:.4f}s".format(time.time() - start))