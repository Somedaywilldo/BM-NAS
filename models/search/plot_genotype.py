import os
import sys
import time
import glob
import numpy as np
import torch
import logging
import argparse
from IPython import embed

from .darts.visualize import plot

class Plotter():
    def __init__(self, args):
        self.args = args
    
    def plot(self, genotype, file_name, task=None):
        plot(   genotype, 
                file_name, 
                self.args,
                task)


if __name__ == '__main__':
    pass



# parser = argparse.ArgumentParser("")
# parser.add_argument('--inner_representation_size', type=int, default=128)
# args = parser.parse_args()
# CIFAR_CLASSES = 10

# # np.random.seed(args.seed)
# # torch.manual_seed(args.seed)

# model = FusionNetwork(steps=7, multiplier=1, num_input_nodes=8, num_keep_edges=2, args=args)
# # embed()
# # architect = Architect(model, args)
# genotype = model.genotype()
# plot(genotype.normal, "normal", model._multiplier, model._num_input_nodes, model._num_keep_edges)
