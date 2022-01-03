import numpy as np
import torch
import argparse
import time

import re
import models.search.darts.utils as utils
import torch.backends.cudnn as cudnn
import glob 
import logging
import sys
import os

import torch.optim as op

import models.auxiliary.scheduler as sc
import models.search.train_searchable.mmimdb as tr
import models.search.mmimdb_darts_searchable as mmimdb
import torchvision.transforms as transforms
from datasets import mmimdb as mmimdb_data
from torch.utils.data import DataLoader

from models.search.plot_genotype import Plotter
from models.search.darts.genotypes import *
from IPython import embed


def parse_args():
    parser = argparse.ArgumentParser(description='BM-NAS Configuration')
    
    parser.add_argument('--seed', type=int, default=2, help='random seed')
    # experiment directory
    parser.add_argument('--save', type=str, default='EXP', help='where to save the experiment')

    # loading searching experiment, if not None, perform evalution
    parser.add_argument('--search_exp_dir', type=str, help='evaluate which search exp', default=None)
    # loading evaluation experiment, if not None, perform test
    parser.add_argument('--eval_exp_dir', type=str, help='test which eval exp', default=None)

    # dataset and data parallel
    parser.add_argument('--datadir', type=str, help='data directory',
                        default='/mnt/scratch/xiaoxiang/yihang/mmimdb/')
    parser.add_argument('--small_dataset', action='store_true', default=False, help='use mini dataset for debugging')
    parser.add_argument('--num_workers', type=int, help='Dataloader CPUS', default=32)
    parser.add_argument('--use_dataparallel', help='Use several GPUs', action='store_true', default=False)

    # basic learning settings
    parser.add_argument('--batchsize', type=int, help='batch size', default=8)
    parser.add_argument('--epochs', type=int, help='training epochs', default=30)
    parser.add_argument("--drpt", action="store", default=0.1, dest="drpt", type=float, help="dropout")

    # number of input features
    parser.add_argument('--num_input_nodes', type=int, help='cell input', default=6)
    parser.add_argument('--num_keep_edges', type=int, help='cell step connect', default=2)

    # for cells and steps and inner representation size
    parser.add_argument('--C', type=int, help='channels for conv layer', default=192)
    parser.add_argument('--L', type=int, help='length after conv and pool', default=16)
    parser.add_argument('--multiplier', type=int, help='cell output concat', default=2)
    parser.add_argument('--steps', type=int, help='cell steps', default=2)
    parser.add_argument('--node_steps', type=int, help='inner node steps', default=1)
    parser.add_argument('--node_multiplier', type=int, help='inner node output concat', default=1)
    
    # number of classes
    parser.add_argument('--num_outputs', type=int, help='output dimension', default=23)
    parser.add_argument('--f1_type', type=str, help="use 'weighted' or 'macro' F1 Score", default='weighted')

    # archtecture optimizer
    parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
    parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
    
    # network optimizer and scheduler
    parser.add_argument('--eta_max', type=float, help='max learning rate', default=0.001)
    parser.add_argument('--eta_min', type=float, help='min laerning rate', default=0.000001)
    parser.add_argument('--Ti', type=int, help='for cosine annealing scheduler, epochs Ti', default=1)
    parser.add_argument('--Tm', type=int, help='for cosine annealing scheduler, epochs multiplier Tm', default=2)
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')

    return parser.parse_args()

def get_dataloaders(args):
    transformer_val = transforms.Compose([mmimdb_data.ToTensor()])
    transformer_tra = transforms.Compose([mmimdb_data.ToTensor()])

    dataset_training = mmimdb_data.MM_IMDB(args.datadir, transform=transformer_tra, stage='train', feat_dim=300,args=args)
    dataset_dev = mmimdb_data.MM_IMDB(args.datadir, transform=transformer_val, stage='dev', feat_dim=300, args=args)
    dataset_test = mmimdb_data.MM_IMDB(args.datadir, transform=transformer_val, stage='test', feat_dim=300, args=args)

    datasets = {'train': dataset_training, 'dev': dataset_dev, 'test': dataset_test}
    dataloaders = {x: DataLoader(datasets[x], batch_size=args.batchsize, shuffle=True, num_workers=args.num_workers,
                    drop_last=False) for x in ['train', 'dev', 'test']}

    return dataloaders


def train_model(model, dataloaders, args, device, logger):
    dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'dev', 'test']}

    num_batches_per_epoch = dataset_sizes['train'] / args.batchsize
    criterion = torch.nn.BCEWithLogitsLoss()

    # optimizer and scheduler
    params = model.central_params()
    optimizer = op.Adam(params, lr=args.eta_max / 10, weight_decay=1e-4)
    scheduler = sc.LRCosineAnnealingScheduler(args.eta_max, args.eta_min, args.Ti, args.Tm, num_batches_per_epoch)

    # hardware tuning
    if torch.cuda.device_count() > 1 and args.use_dataparallel:
        model = torch.nn.DataParallel(model)
    model.to(device)

    plotter = Plotter(args)

    if torch.cuda.device_count() > 1 and args.use_dataparallel:
        params = model.module.parameters()
        # params = model.module.central_params()
    else:
        params = model.parameters()
        # params = model.central_params()

    optimizer = op.Adam(params, lr=args.eta_max, weight_decay=1e-4)
    scheduler = sc.LRCosineAnnealingScheduler(args.eta_max, args.eta_min, args.Ti, args.Tm, num_batches_per_epoch)

    status = 'eval'
    test_f1, test_genotype = tr.train_mmimdb_track_f1(model, None, criterion, optimizer, 
                                            scheduler, dataloaders, dataset_sizes,
                                            device, args.epochs, 
                                            args.use_dataparallel, logger, plotter, args,
                                            args.f1_type, 0.0, 0.3, 
                                            status)

    # logger.info("*" * 10)
    # logger.info('Final test F1: ' + str(test_f1) )
    return test_f1

def test_model(model, dataloaders, args, device, logger, test_model_path):
    
    model.load_state_dict(torch.load(test_model_path))
    model.eval()
    criterion = torch.nn.BCEWithLogitsLoss()
    dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['test']}
    # hardware tuning
    if torch.cuda.device_count() > 1 and args.use_dataparallel:
        model = torch.nn.DataParallel(model)
    model.to(device)
    # status 
    status = 'eval'
    test_f1 = tr.test_mmimdb_track_f1(model, criterion, dataloaders, 
                                    dataset_sizes, device, 
                                    args.use_dataparallel, logger, args,
                                    args.f1_type, init_f1=0.0, th_fscore=0.3)
    # logger.info('Final test F1: {}'.format(test_f1))
    return test_f1

if __name__ == "__main__":
    args = parse_args()
    test_only = False

    # test only
    test_model_path = None

    if args.eval_exp_dir != None:
        test_only = True
        
        args.save = 'test-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
        args.save = os.path.join(args.eval_exp_dir, args.save)
        
        best_test_model_path = os.path.join(args.eval_exp_dir, 'best', 'best_test_model.pt')
        best_genotype_path = os.path.join(args.eval_exp_dir, 'best', 'best_test_genotype.pkl')

    elif args.search_exp_dir != None:
        best_genotype_path = os.path.join(args.search_exp_dir, 'best', 'best_genotype.pkl')
        args.save = 'eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
        args.save = os.path.join(args.search_exp_dir, args.save)

    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled=True
    torch.cuda.manual_seed(args.seed)

    utils.create_exp_dir(args.save, scripts_to_save=None)

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
            format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logger = logging.getLogger()
    logger.addHandler(fh)

    logging.info("args = %s", args)

    # %% hardware
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_gpu else "cpu")

    criterion = torch.nn.BCEWithLogitsLoss()
    genotype = utils.load_pickle(best_genotype_path)
    
    model = mmimdb.Found_Image_Text_Net(args, criterion, genotype)

    dataloaders = get_dataloaders(args)
    start_time = time.time()
    
    model_f1 = None
    if test_only:
        model_f1 = test_model(model, dataloaders, args, device, logger, best_test_model_path)
    else:
        model_f1 = train_model(model, dataloaders, args, device, logger)

    time_elapsed = time.time() - start_time

    logger.info("*" * 50)
    logger.info('Total duration {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    logger.info('Final model {} F1: {}'.format(args.f1_type, model_f1))