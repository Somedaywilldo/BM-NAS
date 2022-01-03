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
import models.search.train_searchable.ntu as tr
import models.search.ntu_darts_searchable as ntu

from models.search.plot_genotype import Plotter
from models.search.darts.genotypes import *
from IPython import embed

def parse_args():
    parser = argparse.ArgumentParser(description='Modality optimization.')

    parser.add_argument('--search_exp_dir', type=str, help='evaluate which search exp', default=None)
    parser.add_argument('--eval_exp_dir', type=str, help='evaluate which eval exp', default=None)

    parser.add_argument('--checkpointdir', type=str, help='output base dir',
                        default='/mnt/data/xiaoxiang/yihang/Baidu_MM/BM-NAS/checkpoints/ntu')
    parser.add_argument('--datadir', type=str, help='data directory',
                        default='/mnt/scratch/xiaoxiang/yihang/NTU/')
    parser.add_argument('--ske_cp', type=str, help='Skeleton net checkpoint (assuming is contained in checkpointdir)',
                        default='skeleton_32frames_85.24.checkpoint')
    parser.add_argument('--rgb_cp', type=str, help='RGB net checkpoint (assuming is contained in checkpointdir)',
                        default='rgb_8frames_83.91.checkpoint')
    
    # args for darts
    parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
    parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
    parser.add_argument('--num_input_nodes', type=int, help='cell input', default=8)
    parser.add_argument('--num_keep_edges', type=int, help='cell step connect', default=2)
    parser.add_argument('--multiplier', type=int, help='cell output concat', default=4)
    parser.add_argument('--steps', type=int, help='cell steps', default=4)
    parser.add_argument('--unrolled', action="store_true", default=False, help='unrolled gradient of darts')
    parser.add_argument('--seed', type=int, default=2, help='random seed')
    # parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
    parser.add_argument('--save', type=str, default='EXP', help='load dir')
    # search-EXP-20200824-012150
    # for darts operations and inner representation size
    parser.add_argument('--C', type=int, help='channels', default=256)
    parser.add_argument('--L', type=int, help='length after pool', default=8)
    # parser.add_argument('--num_heads', type=int, help='attention heads number', default=2)
    parser.add_argument('--node_multiplier', type=int, help='inner node output concat', default=1)
    parser.add_argument('--node_steps', type=int, help='inner node steps', default=2)
    
    parser.add_argument('--small_dataset', action='store_true', default=False, help='dataset scale')

    parser.add_argument('--num_outputs', type=int, help='output dimension', default=60)
    parser.add_argument('--batchsize', type=int, help='batch size', default=96)

    parser.add_argument('--epochs', type=int, help='training epochs', default=30)
    parser.add_argument('--eta_max', type=float, help='eta max', default=0.001)
    parser.add_argument('--eta_min', type=float, help='eta min', default=0.000001)
    parser.add_argument('--Ti', type=int, help='epochs Ti', default=5)
    parser.add_argument('--Tm', type=int, help='epochs multiplier Tm', default=2)
    parser.add_argument('--use_dataparallel', help='Use several GPUs', action='store_true', dest='use_dataparallel',
                        default=False)
    parser.add_argument('--j', dest='num_workers', type=int, help='Dataloader CPUS', default=16)
    parser.add_argument('--modality', type=str, help='', default='both')
    parser.add_argument('--no-verbose', help='verbose', action='store_false', dest='verbose', default=True)

    parser.add_argument("--vid_dim", action="store", default=256, dest="vid_dim",
                        help="frame side dimension (square image assumed) ")
    parser.add_argument("--vid_fr", action="store", default=30, dest="vi_fr", help="video frame rate")
    parser.add_argument("--vid_len", action="store", default=(8, 32), dest="vid_len", type=int, nargs='+',
                        help="length of video, as a tuple of two lengths, (rgb len, skel len)")
    parser.add_argument("--drpt", action="store", default=0.2, dest="drpt", type=float, help="dropout")

    parser.add_argument('--no_bad_skel', action="store_true",
                        help='Remove the 300 bad samples, espec. useful to evaluate', default=True)
    parser.add_argument("--no_norm", action="store_true", default=False, dest="no_norm",
                        help="Not normalizing the skeleton")

    return parser.parse_args()

def get_dataloaders(args):
    import torchvision.transforms as transforms
    from datasets import ntu as d
    from torch.utils.data import DataLoader

    # Handle data
    transformer_val = transforms.Compose([d.NormalizeLen(args.vid_len), d.ToTensor()])
    transformer_tra = transforms.Compose([d.AugCrop(), d.NormalizeLen(args.vid_len), d.ToTensor()])

    dataset_training = d.NTU(args.datadir, transform=transformer_tra, stage='train_val', args=args)
    
    dataset_testing = d.NTU(args.datadir, transform=transformer_val, stage='test', args=args)

    datasets = {'train': dataset_training, 'test': dataset_testing}

    dataloaders = {x: DataLoader(datasets[x], batch_size=args.batchsize, shuffle=True, num_workers=args.num_workers,
                                 drop_last=False, pin_memory=True) for x in ['train', 'test']}

    return dataloaders

def train_model(model, dataloaders, args, device, logger):
    dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'test']}

    num_batches_per_epoch = dataset_sizes['train'] / args.batchsize
    criterion = torch.nn.CrossEntropyLoss()
    # loading pretrained weights
    skemodel_filename = os.path.join(args.checkpointdir, args.ske_cp)
    rgbmodel_filename = os.path.join(args.checkpointdir, args.rgb_cp)
    model.skenet.load_state_dict(torch.load(skemodel_filename))
    model.rgbnet.load_state_dict(torch.load(rgbmodel_filename))

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
    else:
        params = model.parameters()

    optimizer = op.Adam(params, lr=args.eta_max, weight_decay=1e-4)
    scheduler = sc.LRCosineAnnealingScheduler(args.eta_max, args.eta_min, args.Ti, args.Tm, num_batches_per_epoch)

    # status 
    status = 'eval'
    test_acc, test_genotype = tr.train_ntu_track_acc(model, None, criterion, optimizer, 
                                            scheduler, dataloaders, dataset_sizes,
                                            device, args.epochs, args.verbose,
                                            args.use_dataparallel, logger, plotter, args,
                                            status)

    test_acc = test_acc.item()
    logger.info('Final test accuracy: ' + str(test_acc) )
    return test_acc

def test_model(model, dataloaders, args, device, 
                logger, test_model_path, genotype):
    criterion = torch.nn.CrossEntropyLoss()
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
    test_acc = tr.test_ntu_track_acc(model, dataloaders, criterion, genotype, 
                                    dataset_sizes, device, logger, args)
    test_acc = test_acc.item()
    # logger.info('Final test accuracy: {}'.format(test_acc))
    return test_acc

if __name__ == "__main__":
    args = parse_args()
    test_only = False
    # test only
    test_model_path = None

    if args.eval_exp_dir != None:
        test_only = True
        eval_exp_dir = args.eval_exp_dir
        args.search_exp_dir = args.eval_exp_dir.split('/')[0]
        best_genotype_path = os.path.join(args.eval_exp_dir, 'best', 'best_test_genotype.pkl')

        batchsize = args.batchsize
        epochs = args.epochs

        search_exp_dir = args.search_exp_dir
        # new_args = utils.load_pickle(os.path.join(args.search_exp_dir, 'args.pkl'))
        # args = new_args
        
        args.batchsize = batchsize
        args.epochs = epochs
        
        args.save = 'test-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
        args.save = os.path.join(eval_exp_dir, args.save)
        
        test_model_path = os.path.join(eval_exp_dir, 'best', 'best_test_model.pt')

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

    # %% hardware
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_gpu else "cpu")

    criterion = torch.nn.CrossEntropyLoss()

    # print(best_genotype_path)
    genotype = utils.load_pickle(best_genotype_path)
    # genotype = Genotype(edges=[('skip', 3), ('skip', 7)], steps=[StepGenotype(inner_edges=[('skip', 1), ('skip', 0)], inner_steps=['cat_conv_relu'], inner_concat=[2])], concat=[8])
    
    model = ntu.Found_Skeleton_Image_Net(args, criterion, genotype)
    # model = ntu.Searchable_Skeleton_Image_Net(args, criterion, genotype)

    dataloaders = get_dataloaders(args)
    start_time = time.time()
    
    model_acc = None
    if test_only:
        model_acc = test_model(model, dataloaders, args, device, logger, test_model_path, genotype)
    else:
        model_acc = train_model(model, dataloaders, args, device, logger)

    time_elapsed = time.time() - start_time

    logger.info("*" * 50)
    logger.info('Training in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    logger.info('Model Acc: {}'.format(model_acc))


    # filename = args.checkpointdir+"/final_conf_" + confstr + "_" + str(modelacc.item())+'.checkpoint'
    # torch.save(rmode.state_dict(), filename)
