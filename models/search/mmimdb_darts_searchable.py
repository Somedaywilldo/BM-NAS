import torch
import torch.nn as nn
import torch.optim as op
import os

import models.auxiliary.scheduler as sc
import models.auxiliary.aux_models as aux
import models.central.mmimdb as mmimdb
import models.search.train_searchable.mmimdb as tr

from IPython import embed
import numpy as np

from .darts.model_search import FusionNetwork
from .darts.model import Found_FusionNetwork
from models.search.plot_genotype import Plotter
from .darts.architect import Architect

def train_darts_model(dataloaders, args, device, logger):
    dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'dev', 'test']}
    num_batches_per_epoch = dataset_sizes['train'] / args.batchsize
    criterion = torch.nn.BCEWithLogitsLoss()
    # model to train
    model = Searchable_Image_Text_Net(args, criterion)
    params = model.central_params()

    # optimizer and scheduler
    optimizer = op.Adam(params, lr=args.eta_max, weight_decay=args.weight_decay)
    scheduler = sc.LRCosineAnnealingScheduler(args.eta_max, args.eta_min, args.Ti, args.Tm,
                                              num_batches_per_epoch)

    arch_optimizer = op.Adam(model.arch_parameters(),
            lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)

    # hardware tuning
    if torch.cuda.device_count() > 1 and args.parallel:
        model = torch.nn.DataParallel(model)

    model.to(device)
    architect = Architect(model, args, criterion, arch_optimizer)

    plotter = Plotter(args)
    best_f1, best_genotype = tr.train_mmimdb_track_f1(model, architect,
                                            criterion, optimizer, scheduler, dataloaders,
                                            dataset_sizes,
                                            device=device, 
                                            num_epochs=args.epochs, 
                                            parallel=args.parallel,
                                            logger=logger,
                                            plotter=plotter,
                                            args=args,
                                            f1_type=args.f1_type,
                                            init_f1=0.0, th_fscore=0.3)

    return best_f1, best_genotype

class Searchable_Image_Text_Net(nn.Module):
    def __init__(self, args, criterion):
        super().__init__()

        self.args = args
        self.criterion = criterion

        self.imagenet = mmimdb.GP_VGG(args)
        self.textnet = mmimdb.MaxOut_MLP(args)

        self.reshape_layers = self.create_reshape_layers(args)
        self.multiplier = args.multiplier
        self.steps = args.steps
        self.parallel = args.parallel

        self.num_input_nodes = args.num_input_nodes
        self.num_keep_edges = args.num_keep_edges
        
        self._criterion = criterion

        self.fusion_net = FusionNetwork( steps=self.steps, multiplier=self.multiplier, 
                                         num_input_nodes=self.num_input_nodes, num_keep_edges=self.num_keep_edges,
                                         args=self.args,
                                         criterion=self.criterion)
        
        self.central_classifier = nn.Linear(self.args.C * self.args.L * self.multiplier,
                                            args.num_outputs)

    def create_reshape_layers(self, args):
        C_ins = [512, 512, 512, 512, 64, 128]
        reshape_layers = nn.ModuleList()
        for i in range(len(C_ins)):
            reshape_layers.append(aux.ReshapeInputLayer_MMIMDB(C_ins[i], args.C, args.L, args))
        return reshape_layers

    def reshape_input_features(self, input_features):
        ret = []
        for i, input_feature in enumerate(input_features):
            reshaped_feature = self.reshape_layers[i](input_feature)
            ret.append(reshaped_feature)
        return ret

    def forward(self, tensor_tuple):
        text, image = tensor_tuple

        # apply net on input image        
        image_features = self.imagenet(image)
        image_features = image_features[0:-1]

        # apply net on input skeleton
        text_features = self.textnet(text)
        text_features = text_features[0:-1]

        input_features = list(image_features) + list(text_features)
        input_features = self.reshape_input_features(input_features)

        out = self.fusion_net(input_features)
        out = self.central_classifier(out)
        return out

    def genotype(self):
        return self.fusion_net.genotype()
    
    def central_params(self):
        central_parameters = [
            {'params': self.reshape_layers.parameters()},
            {'params': self.fusion_net.parameters()},
            {'params': self.central_classifier.parameters()}
        ]
        return central_parameters
    
    def _loss(self, input_features, labels):
        logits = self(input_features)
        return self._criterion(logits, labels) 

    def arch_parameters(self):
        return self.fusion_net.arch_parameters() 

class Found_Image_Text_Net(nn.Module):
    def __init__(self, args, criterion, genotype):
        super().__init__()

        self.args = args
        self.imagenet = mmimdb.GP_VGG(args)
        self.textnet = mmimdb.MaxOut_MLP(args)
        self._genotype = genotype

        self.reshape_layers = self.create_reshape_layers(args)

        self.multiplier = args.multiplier
        self.steps = args.steps
        self.parallel = args.parallel

        self.num_input_nodes = args.num_input_nodes
        self.num_keep_edges = args.num_keep_edges
        
        self.criterion = criterion

        self.fusion_net = Found_FusionNetwork( steps=self.steps, multiplier=self.multiplier, 
                                         num_input_nodes=self.num_input_nodes, num_keep_edges=self.num_keep_edges,
                                         args=self.args,
                                         criterion=self.criterion,
                                         genotype=self._genotype)
        
        self.central_classifier = nn.Linear(self.args.C * self.args.L * self.multiplier,
                                            args.num_outputs)

    def create_reshape_layers(self, args):
        C_ins = [512, 512, 512, 512, 64, 128]
        reshape_layers = nn.ModuleList()

        input_nodes = []
        for edge in self._genotype.edges:
            input_nodes.append(edge[1])
        input_nodes = list(set(input_nodes))

        for i in range(len(C_ins)):
            if i in input_nodes:
                reshape_layers.append(aux.ReshapeInputLayer_MMIMDB(C_ins[i], args.C, args.L, args))
            else:
                # here the reshape layers is not used, so we set it to ReLU to make it have no parameters
                reshape_layers.append(nn.ReLU())
        
        return reshape_layers

    def reshape_input_features(self, input_features):
        ret = []
        for i, input_feature in enumerate(input_features):
            reshaped_feature = self.reshape_layers[i](input_feature)
            ret.append(reshaped_feature)
        return ret

    def forward(self, tensor_tuple):
        text, image = tensor_tuple

        # apply net on input image        
        image_features = self.imagenet(image)
        image_features = image_features[0:-1]

        # apply net on input skeleton
        text_features = self.textnet(text)
        text_features = text_features[0:-1]

        input_features = list(image_features) + list(text_features)
        input_features = self.reshape_input_features(input_features)

        out = self.fusion_net(input_features)
        out = self.central_classifier(out)

        return out

    def genotype(self):
        return self._genotype
    
    def central_params(self):
        central_parameters = [
            {'params': self.reshape_layers.parameters()},
            {'params': self.fusion_net.parameters()},
            {'params': self.central_classifier.parameters()}
        ]
        return central_parameters
    
    def _loss(self, input_features, labels):
        logits = self(input_features)
        return self._criterion(logits, labels) 
