from data.citation_data import CitationDataset

def load_data(dataset):
    if dataset == 'cora' or dataset == 'citeseer' or dataset == 'pubmed':
        data = CitationDataset(dataset)
        return data