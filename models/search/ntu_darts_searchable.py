import torch
import torch.nn as nn
import torch.optim as op
import os

import models.auxiliary.scheduler as sc
import models.auxiliary.aux_models as aux
import models.central.ntu as ntu
import models.search.train_searchable.ntu as tr

from IPython import embed
import numpy as np

from .darts.model_search import FusionNetwork
from .darts.model import Found_FusionNetwork
from models.search.plot_genotype import Plotter
from .darts.architect import Architect
import torch.nn.functional as F

from .darts.node_operations import *

def train_darts_model(dataloaders, args, device, logger):
    dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'dev', 'test']}
    num_batches_per_epoch = dataset_sizes['train'] / args.batchsize
    criterion = torch.nn.CrossEntropyLoss()

    # model to train
    model = Searchable_Skeleton_Image_Net(args, criterion, logger)


    # loading pretrained weights
    skemodel_filename = os.path.join(args.checkpointdir, args.ske_cp)
    rgbmodel_filename = os.path.join(args.checkpointdir, args.rgb_cp)

    model.skenet.load_state_dict(torch.load(skemodel_filename))
    model.rgbnet.load_state_dict(torch.load(rgbmodel_filename))
    # parameters to update during training

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
        # model = torch.nn.parallel.DistributedDataParallel(model)
    model.to(device)
    architect = Architect(model, args, criterion, arch_optimizer)

    plotter = Plotter(args)

    best_genotype = tr.train_ntu_track_acc(model, architect,
                                            criterion, optimizer, scheduler, dataloaders,
                                            dataset_sizes,
                                            device=device,
                                            num_epochs=args.epochs,
                                            parallel=args.parallel,
                                            logger=logger,
                                            plotter=plotter,
                                            args=args)


    return best_genotype

class Searchable_Skeleton_Image_Net(nn.Module):
    def __init__(self, args, criterion, logger):
        super(Searchable_Skeleton_Image_Net, self).__init__()

        self.args = args
        self.criterion = criterion
        self.logger = logger

        self.rgbnet = ntu.Visual(args)
        self.skenet = ntu.Skeleton(args)

        # self.gp_v, self.gp_s = self._create_global_poolings()
        # self.gp_v, self.gp_s = self._create_global_poolings(args)
        self.reshape_layers = self.create_reshape_layers(args)
        # self.fusion_layers = self._create_fc_layers()
        self.multiplier = args.multiplier
        self.steps = args.steps
        self.parallel = args.parallel

        self.num_input_nodes = 8
        self.num_keep_edges = 2
        self._criterion = criterion

        self.fusion_net = FusionNetwork( steps=self.steps, multiplier=self.multiplier,
                                         num_input_nodes=self.num_input_nodes, num_keep_edges=self.num_keep_edges,
                                         args=self.args,
                                         criterion=self.criterion,
                                         logger=self.logger)

        self.central_classifier = nn.Linear(self.args.C * self.args.L * self.multiplier,
                                            args.num_outputs)

    def create_reshape_layers(self, args):
        C_ins = [512, 1024, 2048, 2048, 128, 256, 1024, 512]
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

    def forward(self, tensor_tuple):

        skeleton, image = tensor_tuple[1], tensor_tuple[0]

        # apply net on input image
        visual_features = self.rgbnet(image)
        # visual_classifier = visual_features[-1]
        visual_features = visual_features[-5:-1]

        # apply net on input skeleton
        skel_features, _ = self.skenet(skeleton)
        skel_features = skel_features[-4:]

        # apply global pooling and batchnorms on visual features
        # embed()

        # print(skel_features[3][0][0])

        # visual_features = [self.gp_v[idx](feat) for idx, feat in enumerate(visual_features)]
        # skel_features = [self.gp_s[idx](feat) for idx, feat in enumerate(skel_features)]

        # embed()
        # exit(0)
        input_features = list(visual_features) + list(skel_features)
        input_features = self.reshape_input_features(input_features)


        # for input_feature in input_features:
        #     print(input_feature.shape)
        # print(input_features[7][0,0,:])
        # exit(0)

        out = self.fusion_net(input_features)
        out = self.central_classifier(out)

        return out

    def genotype(self):
        return self.fusion_net.genotype()

    def central_params(self):
        central_parameters = [
            {'params': self.fusion_net.parameters()},
            {'params': self.central_classifier.parameters()}
        ]
        return central_parameters

    def _loss(self, input_features, labels):
        logits = self(input_features)
        return self._criterion(logits, labels)

    def arch_parameters(self):
        return self.fusion_net.arch_parameters()

    # def _create_global_poolings(self, args):
    #     gp_list_v = [aux.GlobalPadPooling2D(args) for i in range(4)]
    #     gp_list_s = [aux.GlobalPadPooling2D(args) for i in range(4)]
    #     return nn.ModuleList(gp_list_v), nn.ModuleList(gp_list_s)

    def _create_global_poolings(self, args):
        gp_list_v = [aux.Global_Pool_FC() for i in range(4)]
        gp_list_s = [aux.Global_Pool_FC() for i in range(4)]
        return nn.ModuleList(gp_list_v), nn.ModuleList(gp_list_s)

class Found_Skeleton_Image_Net(nn.Module):
    def __init__(self, args, criterion, genotype):
        super().__init__()

        self.args = args
        self.criterion = criterion

        self.rgbnet = ntu.Visual(args)
        self.skenet = ntu.Skeleton(args)
        self._genotype = genotype

        self.reshape_layers = self.create_reshape_layers(args)
        # self.fusion_layers = self._create_fc_layers()
        self.multiplier = args.multiplier
        self.steps = args.steps
        self.parallel = args.parallel

        self.num_input_nodes = 8
        self.num_keep_edges = 2
        self._criterion = criterion

        self.fusion_net = Found_FusionNetwork( steps=self.steps, multiplier=self.multiplier,
                                         num_input_nodes=self.num_input_nodes, num_keep_edges=self.num_keep_edges,
                                         args=self.args,
                                         criterion=self.criterion,
                                         genotype=self._genotype)

        self.central_classifier = nn.Linear(self.args.C * self.args.L * self.multiplier,
                                            args.num_outputs)

    def create_reshape_layers(self, args):
        C_ins = [512, 1024, 2048, 2048, 128, 256, 1024, 512]
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

    def forward(self, tensor_tuple):

        skeleton, image = tensor_tuple[1], tensor_tuple[0]

        # apply net on input image
        visual_features = self.rgbnet(image)
        # visual_classifier = visual_features[-1]
        visual_features = visual_features[-5:-1]

        # apply net on input skeleton
        skel_features, _ = self.skenet(skeleton)
        skel_features = skel_features[-4:]

        input_features = list(visual_features) + list(skel_features)
        input_features = self.reshape_input_features(input_features)

        out = self.fusion_net(input_features)

        # print("out shape:", out.shape)
        # embed()

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

class Found_Simple_Concat_Skeleton_Image_Net(nn.Module):
    def __init__(self, args, criterion, genotype):
        super().__init__()

        self.args = args
        self.criterion = criterion

        self.rgbnet = ntu.Visual(args)
        self.skenet = ntu.Skeleton(args)

        self.reshape_layers = self.create_reshape_layers(args)
        # self.fusion_layers = self._create_fc_layers()
        self.multiplier = args.multiplier
        self.steps = args.steps
        self.parallel = args.parallel

        self.num_input_nodes = 8
        self.num_keep_edges = 2
        self._criterion = criterion

        self._genotype = None
        # self.fusion_net = Found_FusionNetwork( steps=self.steps, multiplier=self.multiplier,
        #                                  num_input_nodes=self.num_input_nodes, num_keep_edges=self.num_keep_edges,
        #                                  args=self.args,
        #                                  criterion=self.criterion,
        #                                  genotype=self._genotype)

        self.fusion_net = nn.ReLU()

        self.central_classifier = nn.Sequential(
            nn.Linear(self.args.C * self.args.L * 2, self.args.C),
            nn.ReLU(),
            nn.BatchNorm1d(self.args.C),
            nn.Linear(self.args.C, args.num_outputs)
        )

    def create_reshape_layers(self, args):
        C_ins = [512, 1024, 2048, 2048, 128, 256, 1024, 512]
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

    def forward(self, tensor_tuple):

        skeleton, image = tensor_tuple[1], tensor_tuple[0]

        # apply net on input image
        visual_features = self.rgbnet(image)
        # visual_classifier = visual_features[-1]
        visual_features = visual_features[-5:-1]

        # apply net on input skeleton
        skel_features, _ = self.skenet(skeleton)
        skel_features = skel_features[-4:]

        input_features = list(visual_features) + list(skel_features)
        input_features = self.reshape_input_features(input_features)

        # concat most important features v2, v3, s3, aka: 2, 3, 7
        # v2 = input_features[2]
        v3 = input_features[3]
        s3 = input_features[7]

        out = torch.cat([v3, s3], dim=1)
        # out = self.fusion_net(input_features)
        # print("out shape:", out.shape)
        # embed()
        # exit()
        out = out.view(out.size(0), -1)
        out = self.central_classifier(out)
        return out

    def genotype(self):
        return self._genotype

    def central_params(self):
        central_parameters = [
            # {'params': self.fusion_net.parameters()},
            {'params': self.central_classifier.parameters()}
        ]
        return central_parameters

    def _loss(self, input_features, labels):
        logits = self(input_features)
        return self._criterion(logits, labels)

class Found_Ensemble_Concat_Skeleton_Image_Net(nn.Module):
    def __init__(self, args, criterion, genotype):
        super().__init__()

        self.args = args
        self.criterion = criterion

        self.rgbnet = ntu.Visual(args)
        self.skenet = ntu.Skeleton(args)

        self.reshape_layers = self.create_reshape_layers(args)
        # self.fusion_layers = self._create_fc_layers()
        self.multiplier = args.multiplier
        self.steps = args.steps
        self.parallel = args.parallel

        self.num_input_nodes = 8
        self.num_keep_edges = 2
        self._criterion = criterion

        self._genotype = None
        # self.fusion_net = Found_FusionNetwork( steps=self.steps, multiplier=self.multiplier,
        #                                  num_input_nodes=self.num_input_nodes, num_keep_edges=self.num_keep_edges,
        #                                  args=self.args,
        #                                  criterion=self.criterion,
        #                                  genotype=self._genotype)

        self.fusion_net = nn.ReLU()

        # self.central_classifier = nn.Linear(self.args.C * self.args.L * self.multiplier,
        #                                     args.num_outputs)
        self.central_classifier = nn.Sequential(
            nn.Linear(self.args.C * self.args.L * 5, self.args.C),
            nn.ReLU(),
            nn.BatchNorm1d(self.args.C),
            nn.Linear(self.args.C, args.num_outputs)
        )

    def create_reshape_layers(self, args):
        C_ins = [512, 1024, 2048, 2048, 128, 256, 1024, 512, 60, 60]
        reshape_layers = nn.ModuleList()
        for i in range(len(C_ins)):
            reshape_layers.append(aux.ReshapeInputLayer(C_ins[i], args.C, args.L))
        return reshape_layers

    def reshape_input_features(self, input_features):
        ret = []
        for i, input_feature in enumerate(input_features):
            reshaped_feature = self.reshape_layers[i](input_feature)
            ret.append(reshaped_feature)
        return ret

    def forward(self, tensor_tuple):

        skeleton, image = tensor_tuple[1], tensor_tuple[0]

        # apply net on input image
        visual_features = self.rgbnet(image)
        # visual_classifier = visual_features[-1]
        # visual_features = visual_features[-5:-1]
        v_logits = visual_features[-1]
        visual_features = visual_features[-5:-1]

        # apply net on input skeleton
        skel_features, s_logits = self.skenet(skeleton)
        skel_features = skel_features[-4:]

        input_features = list(visual_features) + list(skel_features) + [v_logits, s_logits]
        input_features = self.reshape_input_features(input_features)

        # concat most important features v2, v3, s3, aka: 2, 3, 7
        v2 = input_features[2]
        v3 = input_features[3]
        s3 = input_features[7]

        v_logits = input_features[8]
        s_logits = input_features[9]

        # embed()
        # exit(0)

        out = torch.cat([v2, v3, s3, v_logits, s_logits], dim=1)
        # out = self.fusion_net(input_features)
        # print("out shape:", out.shape)
        # embed()
        # exit()
        out = out.view(out.size(0), -1)

        out = self.central_classifier(out)

        return out

    def genotype(self):
        return self._genotype

    def central_params(self):
        central_parameters = [
            # {'params': self.fusion_net.parameters()},
            {'params': self.central_classifier.parameters()}
        ]
        return central_parameters

    def _loss(self, input_features, labels):
        logits = self(input_features)
        return self._criterion(logits, labels)

class Found_Ensemble_Skeleton_Image_Net(nn.Module):
    def __init__(self, args, criterion, genotype):
        super().__init__()

        self.args = args
        self.criterion = criterion

        self.rgbnet = ntu.Visual(args)
        self.skenet = ntu.Skeleton(args)

        self.reshape_layers = self.create_reshape_layers(args)
        # self.fusion_layers = self._create_fc_layers()
        self.multiplier = args.multiplier
        self.steps = args.steps
        self.parallel = args.parallel

        self.num_input_nodes = 8
        self.num_keep_edges = 2
        self._criterion = criterion

        self._genotype = None
        # self.fusion_net = Found_FusionNetwork( steps=self.steps, multiplier=self.multiplier,
        #                                  num_input_nodes=self.num_input_nodes, num_keep_edges=self.num_keep_edges,
        #                                  args=self.args,
        #                                  criterion=self.criterion,
        #                                  genotype=self._genotype)

        self.fusion_net = nn.ReLU()

        # self.central_classifier = nn.Linear(self.args.C * self.args.L * 2,
        #                                     args.num_outputs)

        self.central_classifier = nn.Sequential(
            nn.Linear(self.args.C * self.args.L * 2, self.args.C),
            nn.ReLU(),
            nn.BatchNorm1d(self.args.C),
            nn.Linear(self.args.C, args.num_outputs)
        )

    def create_reshape_layers(self, args):
        C_ins = [512, 1024, 2048, 2048, 128, 256, 1024, 512, 60, 60]
        reshape_layers = nn.ModuleList()
        for i in range(len(C_ins)):
            reshape_layers.append(aux.ReshapeInputLayer(C_ins[i], args.C, args.L))
        return reshape_layers

    def reshape_input_features(self, input_features):
        ret = []
        for i, input_feature in enumerate(input_features):
            reshaped_feature = self.reshape_layers[i](input_feature)
            ret.append(reshaped_feature)
        return ret

    def forward(self, tensor_tuple):

        skeleton, image = tensor_tuple[1], tensor_tuple[0]

        # apply net on input image
        visual_features = self.rgbnet(image)
        # visual_classifier = visual_features[-1]
        # visual_features = visual_features[-5:-1]
        v_logits = visual_features[-1]
        visual_features = visual_features[-5:-1]

        # apply net on input skeleton
        skel_features, s_logits = self.skenet(skeleton)
        skel_features = skel_features[-4:]

        input_features = list(visual_features) + list(skel_features) + [v_logits, s_logits]
        input_features = self.reshape_input_features(input_features)

        # concat most important features v2, v3, s3, aka: 2, 3, 7
        # v2 = input_features[2]
        # v3 = input_features[3]
        # s3 = input_features[7]

        v_logits = input_features[8]
        s_logits = input_features[9]

        # embed()
        # exit(0)

        out = torch.cat([v_logits, s_logits], dim=1)
        # out = self.fusion_net(input_features)
        # print("out shape:", out.shape)
        # embed()
        # exit()
        out = out.view(out.size(0), -1)

        out = self.central_classifier(out)

        return out

    def genotype(self):
        return self._genotype

    def central_params(self):
        central_parameters = [
            # {'params': self.fusion_net.parameters()},
            {'params': self.central_classifier.parameters()}
        ]
        return central_parameters

    def _loss(self, input_features, labels):
        logits = self(input_features)
        return self._criterion(logits, labels)

class Found_Simple_Concat_Attn_Skeleton_Image_Net(nn.Module):
    def __init__(self, args, criterion, genotype):
        super().__init__()

        self.args = args
        self.criterion = criterion

        self.rgbnet = ntu.Visual(args)
        self.skenet = ntu.Skeleton(args)

        self.reshape_layers = self.create_reshape_layers(args)
        # self.fusion_layers = self._create_fc_layers()
        self.multiplier = args.multiplier
        self.steps = args.steps
        self.parallel = args.parallel

        self.num_input_nodes = 8
        self.num_keep_edges = 2
        self._criterion = criterion

        self._genotype = None
        # self.fusion_net = Found_FusionNetwork( steps=self.steps, multiplier=self.multiplier,
        #                                  num_input_nodes=self.num_input_nodes, num_keep_edges=self.num_keep_edges,
        #                                  args=self.args,
        #                                  criterion=self.criterion,
        #                                  genotype=self._genotype)

        self.fusion_net = nn.ReLU()

        self.attn1 = ScaledDotAttn()
        self.attn2 = ScaledDotAttn()

        self.central_classifier = nn.Sequential(
            nn.Linear(self.args.C * self.args.L * 2, self.args.C),
            nn.ReLU(),
            nn.BatchNorm1d(self.args.C),
            nn.Linear(self.args.C, args.num_outputs)
        )

    def create_reshape_layers(self, args):
        C_ins = [512, 1024, 2048, 2048, 128, 256, 1024, 512]
        reshape_layers = nn.ModuleList()
        for i in range(len(C_ins)):
            reshape_layers.append(aux.ReshapeInputLayer(C_ins[i], args.C, args.L))
        return reshape_layers

    def reshape_input_features(self, input_features):
        ret = []
        for i, input_feature in enumerate(input_features):
            reshaped_feature = self.reshape_layers[i](input_feature)
            ret.append(reshaped_feature)
        return ret

    def forward(self, tensor_tuple):

        skeleton, image = tensor_tuple[1], tensor_tuple[0]

        # apply net on input image
        visual_features = self.rgbnet(image)
        # visual_classifier = visual_features[-1]
        visual_features = visual_features[-5:-1]

        # apply net on input skeleton
        skel_features, _ = self.skenet(skeleton)
        skel_features = skel_features[-4:]

        input_features = list(visual_features) + list(skel_features)
        input_features = self.reshape_input_features(input_features)

        # concat most important features v2, v3, s3, aka: 2, 3, 7
        # v2 = input_features[2]
        v3 = input_features[3]
        s3 = input_features[7]

        out1 = self.attn1(v3, s3)
        out2 = self.attn2(s3, v3)

        out = torch.cat([out1, out2], dim=1)
        # out = self.fusion_net(input_features)
        # print("out shape:", out.shape)
        # embed()
        # exit()
        out = out.view(out.size(0), -1)
        out = self.central_classifier(out)
        return out

    def genotype(self):
        return self._genotype

    def central_params(self):
        central_parameters = [
            # {'params': self.fusion_net.parameters()},
            {'params': self.central_classifier.parameters()}
        ]
        return central_parameters

    def _loss(self, input_features, labels):
        logits = self(input_features)
        return self._criterion(logits, labels)
