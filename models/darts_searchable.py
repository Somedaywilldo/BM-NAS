import torch
import torch.optim as op
import numpy as np
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataset import Subset
from IPython import embed

# ntu
import models.search.ntu_darts_searchable as ntu
from datasets import ntu as ntu_data

# mm_imdb
import models.search.mmimdb_darts_searchable as mmimdb
from datasets import mmimdb as mmimdb_data

# # egogesture
# import models.search.ego_darts_searchable as ego
# from datasets import ego as ego_data

from models.utils import parse_opts
import models.search.tools as tools

class MMIMDB_Searcher():
    def __init__(self, args, device, logger):
        self.args = args
        self.device = device
        self.logger = logger

        transformer_val = transforms.Compose([mmimdb_data.ToTensor()])
        transformer_tra = transforms.Compose([mmimdb_data.ToTensor()])

        dataset_training = mmimdb_data.MM_IMDB(args.datadir, transform=transformer_tra, stage='train', feat_dim=300, args=args)
        dataset_dev = mmimdb_data.MM_IMDB(args.datadir, transform=transformer_val, stage='dev', feat_dim=300, args=args)
        dataset_test = mmimdb_data.MM_IMDB(args.datadir, transform=transformer_val, stage='test', feat_dim=300, args=args)

        datasets = {'train': dataset_training, 'dev': dataset_dev, 'test': dataset_test}
        self.dataloaders = {
            x: DataLoader(datasets[x], batch_size=args.batchsize, shuffle=True, num_workers=args.num_workers,
                          drop_last=False) for x in ['train', 'dev', 'test']}
    def search(self):
        best_f1, best_genotype = mmimdb.train_darts_model(self.dataloaders, self.args, self.device, self.logger)
        return best_f1, best_genotype

class NTUSearcher():
    def __init__(self, args, device, logger):
        self.args = args
        self.device = device
        self.logger = logger

        # Handle data
        transformer_val = transforms.Compose([ntu_data.NormalizeLen(), ntu_data.ToTensor()])
        transformer_tra = transforms.Compose(
            [ntu_data.AugCrop(), ntu_data.NormalizeLen(), ntu_data.ToTensor()])

        dataset_training = ntu_data.NTU(args.datadir, transform=transformer_tra, stage='train_exp', args=args)
        dataset_dev = ntu_data.NTU(args.datadir, transform=transformer_val, stage='dev', args=args)
        dataset_test = ntu_data.NTU(args.datadir, transform=transformer_val, stage='test', args=args)

        datasets = {'train': dataset_training, 'dev': dataset_dev, 'test': dataset_test}
        self.dataloaders = {
            x: DataLoader(datasets[x], batch_size=args.batchsize, shuffle=True, num_workers=args.num_workers,
                          drop_last=False) for x in ['train', 'dev', 'test']}
    def search(self):
        # search functions that are specific to the dataset
        best_genotype = ntu.train_darts_model(self.dataloaders, self.args, self.device, self.logger)
        return best_genotype

class Ego_Searcher():
    def __init__(self, args, device, logger):
        self.args = args
        self.device = device
        self.logger = logger

        self.opt = parse_opts(self.args)

        self.dataloaders = {
            'train': ego_data.get_train_loader(self.opt, self.args),
            'dev': ego_data.get_dev_loader(self.opt, self.args),
            'test': ego_data.get_test_loader(self.opt, self.args)
        }

    def search(self):
        best_genotype = ego.train_darts_model(self.dataloaders, 
                                                    self.args, 
                                                    self.opt,
                                                    self.device, 
                                                    self.logger)
        return best_genotype
