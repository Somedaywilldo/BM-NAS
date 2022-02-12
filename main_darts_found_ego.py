import argparse
import time
import re
import glob 
import logging
import sys
import os
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.optim as op
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from datasets import ego as ego_data
import models.search.darts.utils as utils
import models.auxiliary.scheduler as sc
import models.search.train_searchable.ego as tr
import models.search.ego_darts_searchable as ego
from models.search.plot_genotype import Plotter
from models.search.darts.genotypes import *
from models.utils import parse_opts

from IPython import embed

def parse_args():
    parser = argparse.ArgumentParser(description='Modality optimization.')
    parser.add_argument('--seed', type=int, default=2, help='random seed')
    # experiment directory
    parser.add_argument('--search_exp_dir', type=str, help='evaluate which search exp', default=None)
    parser.add_argument('--eval_exp_dir', type=str, help='evaluate which eval exp', default=None)
    parser.add_argument('--save', type=str, default='EXP', help='where to save the experiment')

    # pretrained backbone checkpoints and annotations
    parser.add_argument('--checkpointdir', type=str, help='pretrained checkpoints and annotations dir',
                        default='/mnt/data/xiaoxiang/yihang/Baidu_MM/BM-NAS/checkpoints/ego')
    parser.add_argument('--annotation', default='/mnt/data/xiaoxiang/yihang/Baidu_MM/BM-NAS/checkpoints/ego/egogestureall_but_None.json', type=str, help='Annotation file path')
    parser.add_argument('--rgb_cp', type=str, help='rgb video model pth path',
                        default='egogesture_resnext_1.0x_RGB_32_acc_94.01245.checkpoint')
    parser.add_argument('--depth_cp', type=str, help='depth video model pth path',
                        default='egogesture_resnext_1.0x_Depth_32_acc_93.61060.checkpoint')
    
    # dataset and data parallel
    parser.add_argument('--datadir', type=str, help='data directory',
                        default='/mnt/scratch/xiaoxiang/yihang/EgoGesture')
    parser.add_argument('--small_dataset', action='store_true', default=False, help='use mini dataset for debugging')
    parser.add_argument('--use_dataparallel', help='Use several GPUs', action='store_true', dest='use_dataparallel',
                        default=False)
    parser.add_argument('--j', dest='num_workers', type=int, help='Dataloader CPUS', default=32)
    
    # basic learning settings
    parser.add_argument('--batchsize', type=int, help='batch size', default=48)
    parser.add_argument('--epochs', type=int, help='training epochs', default=50)
    parser.add_argument("--drpt", action="store", default=0.2, dest="drpt", type=float, help="dropout")

    # number of input features
    parser.add_argument('--num_input_nodes', type=int, help='total number of modality features', default=8)
    parser.add_argument('--num_keep_edges', type=int, help='cells and steps will have 2 input edges', default=2)
    
    # for cells and steps and inner representation size
    parser.add_argument('--C', type=int, help='channels', default=256)
    parser.add_argument('--L', type=int, help='length after pool', default=8)
    parser.add_argument('--multiplier', type=int, help='cell output concat', default=4)
    parser.add_argument('--steps', type=int, help='cell steps', default=4)
    parser.add_argument('--node_multiplier', type=int, help='inner node output concat', default=1)
    parser.add_argument('--node_steps', type=int, help='inner node steps', default=2)
    
    # number of classes    
    parser.add_argument('--num_outputs', type=int, help='output dimension', default=83)

    # archtecture optimizer
    parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
    parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
    
    # network optimizer and scheduler
    parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
    parser.add_argument('--eta_max', type=float, help='for cosine annealing scheduler, max learning rate', default=0.003)
    parser.add_argument('--eta_min', type=float, help='for cosine annealing scheduler, max learning rate', default=0.000001)
    parser.add_argument('--Ti', type=int, help='for cosine annealing scheduler, epochs Ti', default=5)
    parser.add_argument('--Tm', type=int, help='for cosine annealing scheduler, epochs multiplier Tm', default=2)
    
    return parser.parse_args()

def get_dataloaders(opt, args):
    opt.modality = 'RGB-D'
    dataloaders = {
        # 'train': ego_data.get_train_loader(opt, args),
        'train': ego_data.get_train_dev_loader(opt, args),
        'test': ego_data.get_test_loader(opt, args)
    }
    return dataloaders

def train_model(model, criterion, dataloaders, args, device, logger):
    dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'test']}
    num_batches_per_epoch = dataset_sizes['train'] / args.batchsize

    # loading pretrained weights
    rgb_model_path = os.path.join(args.checkpointdir, args.rgb_cp)
    depth_model_path = os.path.join(args.checkpointdir, args.depth_cp)

    model.rgb_net.load_state_dict(torch.load(rgb_model_path))
    model.depth_net.load_state_dict(torch.load(depth_model_path))

    # optimizer and scheduler
    # hardware tuning
    if torch.cuda.device_count() > 1 and args.use_dataparallel:
        model = torch.nn.DataParallel(model)
    model.to(device)

    plotter = Plotter(args)
    params = None

    if torch.cuda.device_count() > 1 and args.use_dataparallel:
        params = model.module.parameters()
        # params = model.module.central_params()
    else:
        params = model.parameters()
        # params = model.central_params()

    optimizer = op.Adam(params, lr=args.eta_max, weight_decay=1e-4)
    scheduler = sc.LRCosineAnnealingScheduler(args.eta_max, args.eta_min, args.Ti, args.Tm, num_batches_per_epoch)

    # status 
    status = 'eval'
    test_acc, test_genotype = tr.train_ego_track_acc(model, None, criterion, optimizer, 
                                            scheduler, dataloaders, dataset_sizes,
                                            device, args.epochs,
                                            args.use_dataparallel, logger, plotter, args,
                                            status)

    # logger.info('Final test acc: ' + str(val_acc) )
    return test_acc

def test_model(model, criterion, dataloaders, args, device, 
                logger, test_model_path, genotype):
    # criterion = torch.nn.CrossEntropyLoss()
    model.load_state_dict(torch.load(test_model_path))
    model.eval()
    dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['test']}
    # hardware tuning
    if torch.cuda.device_count() > 1 and args.use_dataparallel:
        # print("using parallel")
        model = torch.nn.DataParallel(model)
    model.to(device)
    # status 
    status = 'eval'
    test_acc = tr.test_ego_track_acc(model, dataloaders, criterion, genotype, 
                                    dataset_sizes, device, logger, args)
    test_acc = test_acc.item()
    # logger.info('Final test accuracy: {}'.format(test_acc))
    return test_acc

if __name__ == "__main__":
    args = parse_args()
    # args.save = 'eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))

    test_only = False

    # test only
    test_model_path = None

    if args.eval_exp_dir != None:
        test_only = True
        eval_exp_dir = args.eval_exp_dir
        args.search_exp_dir = args.eval_exp_dir.split('/')[0]

        batchsize = args.batchsize
        epochs = args.epochs

        search_exp_dir = args.search_exp_dir
        
        args.batchsize = batchsize
        args.epochs = epochs
        
        args.save = 'test-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
        args.save = os.path.join(eval_exp_dir, args.save)
        
        best_test_model_path = os.path.join(eval_exp_dir, 'best', 'best_test_model.pt')
        best_genotype_path = os.path.join(eval_exp_dir, 'best', 'best_test_genotype.pkl')

    elif args.search_exp_dir != None:
        best_genotype_path = os.path.join(args.search_exp_dir, 'best', 'best_genotype.pkl')

        batchsize = args.batchsize
        epochs = args.epochs

        search_exp_dir = args.search_exp_dir
        # new_args = utils.load_pickle(os.path.join(args.search_exp_dir, 'args.pkl'))
        # args = new_args
        
        args.batchsize = batchsize
        args.epochs = epochs
        
        args.save = 'eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
        args.save = os.path.join(search_exp_dir, args.save)

    args.no_bad_skel = True

    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled=True
    torch.cuda.manual_seed(args.seed)


    utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
            format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logger = logging.getLogger()
    logger.addHandler(fh)

    logging.info("args = %s", args)
    opt = parse_opts(args)
    logging.info("opt = %s", opt)
    # embed()

    # %% hardware
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_gpu else "cpu")

    criterion = torch.nn.CrossEntropyLoss()
    genotype = utils.load_pickle(best_genotype_path)
    # genotype = Genotype(edges=[('skip', 3), ('skip', 7)], steps=[StepGenotype(inner_edges=[('skip', 1), ('skip', 0)], inner_steps=['cat_conv_relu'], inner_concat=[2])], concat=[8])
    
    model = ego.Found_RGB_Depth_Net(args, opt, criterion, genotype)

    dataloaders = get_dataloaders(opt, args)
    start_time = time.time()
     
    model_acc = None
    if test_only:
        model_acc = test_model(model, criterion, dataloaders, args, device, logger, best_test_model_path, genotype)
    else:
        model_acc = train_model(model, criterion, dataloaders, args, device, logger)

    time_elapsed = time.time() - start_time
    logger.info("*" * 50)
    logger.info('Training in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    logger.info('Model acc: {}'.format(model_acc))


    # filename = args.checkpointdir+"/final_conf_" + confstr + "_" + str(modelacc.item())+'.checkpoint'
    # torch.save(rmode.state_dict(), filename)
