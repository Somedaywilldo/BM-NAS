import models.darts_searchable as S
import torch
import argparse
import time
import models.search.darts.utils as utils
import torch.backends.cudnn as cudnn
import numpy as np

import glob 
import logging
import sys
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Modality optimization.')

    parser.add_argument('--seed', type=int, default=2, help='random seed')

    parser.add_argument('--checkpointdir', type=str, help='output base dir',
                        default='checkpoints/ntu')
    parser.add_argument('--datadir', type=str, help='data directory',
                        default='BM-NAS_dataset/NTU/')

    parser.add_argument('--ske_cp', type=str, help='Skeleton net checkpoint (assuming is contained in checkpointdir)',
                        default='skeleton_32frames_85.24.checkpoint')
    parser.add_argument('--rgb_cp', type=str, help='RGB net checkpoint (assuming is contained in checkpointdir)',
                        default='rgb_8frames_83.91.checkpoint')

    # args for darts
    parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
    parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
    

    parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
    parser.add_argument('--num_input_nodes', type=int, help='cell input', default=8)
    parser.add_argument('--num_keep_edges', type=int, help='cell step connect', default=2)
    parser.add_argument('--multiplier', type=int, help='cell output concat', default=2)
    parser.add_argument('--steps', type=int, help='cell steps', default=2)
    
    parser.add_argument('--node_multiplier', type=int, help='inner node output concat', default=2)
    parser.add_argument('--node_steps', type=int, help='inner node steps', default=2)
    
    # for darts operations and inner representation size
    parser.add_argument('--C', type=int, help='channels for conv layer', default=128)
    parser.add_argument('--L', type=int, help='length after conv and pool', default=8)
    # parser.add_argument('--num_heads', type=int, help='attention heads number', default=2)
    parser.add_argument('--batchsize', type=int, help='batch size', default=96)
    parser.add_argument('--use_dataparallel', help='Use several GPUs', action='store_true', default=False)
    parser.add_argument('--modality', type=str, help='', default='both')


    parser.add_argument('--small_dataset', action='store_true', default=False, help='dataset scale')

    parser.add_argument('--num_outputs', type=int, help='output dimension', default=60)
    parser.add_argument('--epochs', type=int, help='training epochs', default=30)
    parser.add_argument('--eta_max', type=float, help='eta max', default=1e-3)
    parser.add_argument('--eta_min', type=float, help='eta min', default=1e-6)
    parser.add_argument('--Ti', type=int, help='epochs Ti', default=1)
    parser.add_argument('--Tm', type=int, help='epochs multiplier Tm', default=2)
    parser.add_argument('--num_workers', type=int, help='Dataloader CPUS', default=16)

    parser.add_argument("--drpt", action="store", default=0.2, dest="drpt", type=float, help="dropout")
    parser.add_argument('--save', type=str, default='EXP', help='experiment name')

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled=True
    torch.cuda.manual_seed(args.seed)

    args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
    args.save = os.path.join('final_exp/ntu', args.save)
    utils.create_exp_dir(args.save, scripts_to_save=None)

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
            format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logger = logging.getLogger()
    logger.addHandler(fh)

    logging.info("args = %s", args)

    # utils.save_pickle(args, os.path.join(args.save, 'args.pkl'))

    # %% hardware
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_gpu else "cpu")

    # %% Searcher
    ntu_searcher = S.NTUSearcher(args, device, logger)

    # %% Do the search
    logger.info("BM-NAS for NTU Started.")
    start_time = time.time()

    best_acc, best_genotype = ntu_searcher.search()

    time_elapsed = time.time() - start_time

    logger.info("*" * 50)
    logger.info('Search complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    # %% Get best K=5
    logger.info('Now listing best fusion_net genotype:')
    logger.info(best_genotype)
