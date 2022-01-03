import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable

def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])

class Architect(object):
    def __init__(self, model, args, criterion, optimizer):
        self.network_weight_decay = args.weight_decay
        self.criterion = criterion
        self.model = model
        self.optimizer = optimizer
    
    def log_learning_rate(self, logger):
        for param_group in self.optimizer.param_groups:
            logger.info("Architecture Learning Rate: {}".format(param_group['lr']))
            break
    
    def step(self, input_valid, target_valid, logger):
        self.optimizer.zero_grad()
        self._backward_step(input_valid, target_valid)
        self.optimizer.step()

    def _backward_step(self, input_valid, target_valid):
        logits = self.model(input_valid)
        loss = self.criterion(logits, target_valid)
        loss.backward()
    