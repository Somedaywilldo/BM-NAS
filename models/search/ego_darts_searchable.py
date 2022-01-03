import torch
import torch.nn as nn
import torch.optim as op
import os

import models.auxiliary.scheduler as sc
import models.auxiliary.aux_models as aux
import models.central.ego as ego
import models.search.train_searchable.ego as tr
import torch.nn.functional as F

from IPython import embed
import numpy as np

from .darts.model_search import FusionNetwork
from .darts.model import Found_FusionNetwork
from models.search.plot_genotype import Plotter
from .darts.architect import Architect

def train_darts_model(dataloaders, args, opt, device, logger):
    dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'dev', 'test']}
    num_batches_per_epoch = dataset_sizes['train'] / args.batchsize

    criterion = torch.nn.CrossEntropyLoss()
    # criterion.to(device)

    # model to train
    model = Searchable_RGB_Depth_Net(args, opt, criterion)
    params = model.central_params()

    # loading pretrained weights
    rgb_model_path = os.path.join(args.checkpointdir, args.rgb_cp)
    # rgb_model_path = os.path.join(args.rgb_cp)
    depth_model_path = os.path.join(args.checkpointdir, args.depth_cp)

    model.rgb_net.load_state_dict(torch.load(rgb_model_path))
    logger.info("Loading rgb checkpoint: " + rgb_model_path)
    model.depth_net.load_state_dict(torch.load(depth_model_path))
    logger.info("Loading depth checkpoint: " + depth_model_path)

    # optimizer and scheduler
    optimizer = op.Adam(params, lr=args.eta_max, weight_decay=1e-4)
    # optimizer = op.Adam(None, lr=args.eta_max, weight_decay=1e-4)
    scheduler = sc.LRCosineAnnealingScheduler(args.eta_max, args.eta_min, args.Ti, args.Tm,
                                              num_batches_per_epoch)

    arch_optimizer = op.Adam(model.arch_parameters(),
            lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)

    # hardware tuning
    if torch.cuda.device_count() > 1 and args.parallel:
        model = torch.nn.DataParallel(model)
        # model = torch.nn.parallel.DistributedDataParallel(model)
    model.to(device)
    architect = Architect(model, args, criterion, arch_optimizer)

    plotter = Plotter(args)

    best_acc, best_genotype = tr.train_ego_track_acc(model, architect,
                                            criterion, optimizer, scheduler, dataloaders,
                                            dataset_sizes,
                                            device=device, 
                                            num_epochs=args.epochs, 
                                            parallel=args.parallel,
                                            logger=logger,
                                            plotter=plotter,
                                            args=args)

    return best_acc, best_genotype

class Searchable_RGB_Depth_Net(nn.Module):
    def __init__(self, args, opt, criterion):
        super().__init__()

        self.args = args
        self.opt = opt
        self.criterion = criterion

        self.rgb_net = ego.get_rgb_model(opt)
        self.depth_net = ego.get_depth_model(opt)

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
        # self.bn = nn.BatchNorm1d(args.num_outputs * 2)
        # self.central_classifier = nn.Linear(args.num_outputs * 2,
        #                                     args.num_outputs)

    def create_reshape_layers(self, args):
        C_ins = [512, 1024, 2048, 2048, 512, 1024, 2048, 2048]
        reshape_layers = nn.ModuleList()
        for i in range(len(C_ins)):
            reshape_layers.append(aux.ReshapeInputLayer(C_ins[i], args.C, args.L, args))
        return reshape_layers

    def reshape_input_features(self, input_features):
        ret = []
        for i, input_feature in enumerate(input_features):
            reshaped_feature = self.reshape_layers[i](input_feature)
            ret.append(reshaped_feature)
        return ret

    def forward(self, inputs):
        # rgb = inputs[:,0:3,:,:,:]
        # depth = inputs[:,3::,:,:,:]
        rgb, depth = inputs
        # apply net on input rgb videos 
        self.rgb_net.eval()       
        rgb_features = self.rgb_net(rgb)
        rgb_features = rgb_features[0:-1]

        # # apply net on input depth videos   
        self.depth_net.eval()     
        depth_features = self.depth_net(depth)
        depth_features = depth_features[0:-1]
        # embed()
        # exit(0)
        input_features = list(rgb_features) + list(depth_features)
        input_features = self.reshape_input_features(input_features)

        out = self.fusion_net(input_features)
        out = self.central_classifier(out)
        # exit(0)
        
        # out = rgb_features[-1]
        # out = depth_features[-1]
        # out = depth_features[-1]
        # out = torch.cat([rgb_features[-1], depth_features[-1]], dim=1)

        # out = torch.cat([input_features[3], input_features[7]], dim=1)

        # out = self.bn(out)
        # out = F.relu(out)
        # print(out.shape)
        # out = out.view(out.size(0), -1)
        # out = self.central_classifier(out)
        # , rgb_features[-1]), dim=1)
        # out = F.softmax(out, dim=1)
        # out = self.bn(out)
        # out = F.relu(out)
        # out = self.central_classifier(out)

        # embed()
        # out = F.softmax(out, dim=1)
        # print(out.shape)
        return out

    def genotype(self):
        return self.fusion_net.genotype()
    
    def central_params(self):
        central_parameters = [
            {'params': self.fusion_net.parameters()},
            {'params': self.central_classifier.parameters()}
            ,{'params': self.reshape_layers.parameters()}
        ]
        return central_parameters
    
    def _loss(self, input_features, labels):
        logits = self(input_features)
        return self._criterion(logits, labels) 

    def arch_parameters(self):
        return self.fusion_net.arch_parameters() 


class Found_RGB_Depth_Net(nn.Module):
    def __init__(self, args, opt, criterion, genotype):
        super().__init__()

        self.args = args
        self.opt = opt
        self.criterion = criterion

        self.rgb_net = ego.get_rgb_model(opt)
        self.depth_net = ego.get_depth_model(opt)

        self._genotype = genotype

        for p in self.rgb_net.parameters():
            p.requires_grad = False
        
        for p in self.depth_net.parameters():
            p.requires_grad = False

        self.reshape_layers = self.create_reshape_layers(args)
        self.multiplier = args.multiplier
        self.steps = args.steps
        self.parallel = args.parallel

        self.num_input_nodes = args.num_input_nodes
        self.num_keep_edges = args.num_keep_edges
        
        self._criterion = criterion

        self.fusion_net = Found_FusionNetwork( steps=self.steps, multiplier=self.multiplier, 
                                         num_input_nodes=self.num_input_nodes, num_keep_edges=self.num_keep_edges,
                                         args=self.args,
                                         criterion=self.criterion,
                                         genotype=self._genotype)
        
        self.central_classifier = nn.Linear(self.args.C * self.args.L * self.multiplier,
                                            args.num_outputs)
        # self.bn = nn.BatchNorm1d(args.num_outputs * 2)
        # self.central_classifier = nn.Linear(args.num_outputs * 2,
        #                                     args.num_outputs)

    def create_reshape_layers(self, args):
        C_ins = [512, 1024, 2048, 2048, 512, 1024, 2048, 2048]
        reshape_layers = nn.ModuleList()

        input_nodes = []
        for edge in self._genotype.edges:
            input_nodes.append(edge[1])
        input_nodes = list(set(input_nodes))

        for i in range(len(C_ins)):
            if i in input_nodes:
                reshape_layers.append(aux.ReshapeInputLayer(C_ins[i], args.C, args.L, args))
            else:
                reshape_layers.append(nn.ReLU())
        
        return reshape_layers

    def reshape_input_features(self, input_features):
        ret = []
        for i, input_feature in enumerate(input_features):
            reshaped_feature = self.reshape_layers[i](input_feature)
            ret.append(reshaped_feature)
        return ret

    def forward(self, inputs):
        # rgb = inputs[:,0:3,:,:,:]
        # depth = inputs[:,3::,:,:,:]
        rgb, depth = inputs
        # apply net on input rgb videos 
        self.rgb_net.eval()       
        rgb_features = self.rgb_net(rgb)
        rgb_features = rgb_features[0:-1]

        # # apply net on input depth videos   
        self.depth_net.eval()     
        depth_features = self.depth_net(depth)
        depth_features = depth_features[0:-1]
        # embed()
        # exit(0)
        input_features = list(rgb_features) + list(depth_features)
        input_features = self.reshape_input_features(input_features)

        out = self.fusion_net(input_features)
        out = self.central_classifier(out)
        # exit(0)
        return out
    
    def central_params(self):
        central_parameters = [
            {'params': self.fusion_net.parameters()},
            {'params': self.central_classifier.parameters()}
            ,{'params': self.reshape_layers.parameters()}
        ]
        return central_parameters
    
    def genotype(self):
        return self._genotype
    
    def _loss(self, input_features, labels):
        logits = self(input_features)
        return self._criterion(logits, labels) 

    def arch_parameters(self):
        return self.fusion_net.arch_parameters() 
