import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from .operations import *
from .utils import drop_path
from .genotypes import *
from random import sample

import argparse
from IPython import embed

from .node import *

class Found_FusionCell(nn.Module):
    def __init__(self, steps, args, genotype):
        super().__init__()

        self.C = args.C
        self.L = args.L

        op_names, indices = zip(*genotype.edges)
        concat = genotype.concat
        step_nodes = genotype.steps
        self.args = args

        self._compile(self.C, self.L, op_names, indices, concat, step_nodes, args)
        self._steps = steps
        # self.bn = nn.BatchNorm1d(self.C * self._multiplier)
        self.ln = nn.LayerNorm([self.C * self._multiplier, self.L])


    def _compile(self, C, L, op_names, indices, concat, gene_step_nodes, args):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self._multiplier = len(concat)
        # self.step_node_ops = []

        self._ops = nn.ModuleList()
        self._step_nodes = nn.ModuleList()

        for name, index in zip(op_names, indices):
            op = OPS[name](C, L, self.args)
            self._ops += [op]
        
        self._indices = indices

        for gene_step_node in gene_step_nodes:
            step_node = Found_FusionNode(args.node_steps, args.node_multiplier, args, gene_step_node)
            # try darts found node cell
            # step_node = Found_DARTS_FusionNode(args.node_steps, args.node_multiplier, args, gene_step_node)
            # try mfas fusion step node
            # step_node = Found_MFAS_FusionNode(args.node_steps, args.node_multiplier, args, gene_step_node)
            # try aoa
            # step_node = Found_AOA_FusionNode(args.node_steps, args.node_multiplier, args, gene_step_node)
            # try two head attn
            # step_node = Found_TwoHeadAttn_FusionNode(args.node_steps, args.node_multiplier, args, gene_step_node)
            self._step_nodes.append(step_node)

    def forward(self, input_features):

        states = []
        for input_feature in input_features:
            states.append(input_feature)

        # states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2*i]]
            h2 = states[self._indices[2*i+1]]

            op1 = self._ops[2*i]
            op2 = self._ops[2*i+1]
            
            h1 = op1(h1)
            h2 = op2(h2)
            
            s = self._step_nodes[i](h1, h2)
            states += [s]
        
        out = torch.cat(states[-self._multiplier:], dim=1)
        # out = self.bn(out)
        out = self.ln(out)
        out = F.relu(out)
        out = out.view(out.size(0), -1)

        # print("cell out shape:". out.shape)
        return out


class Found_Random_FusionCell(nn.Module):
    def __init__(self, steps, args, genotype):
        super().__init__()

        self.C = args.C
        self.L = args.L

        op_names, indices = zip(*genotype.edges)
        concat = genotype.concat
        step_nodes = genotype.steps
        self.args = args

        self._compile(self.C, self.L, op_names, indices, concat, step_nodes, args)
        self._steps = steps
        # self.bn = nn.BatchNorm1d(self.C * self._multiplier)
        self.ln = nn.LayerNorm([self.C * self._multiplier, self.L])

    def _compile(self, C, L, op_names, indices, concat, gene_step_nodes, args):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self._multiplier = len(concat)
        # self.step_node_ops = []

        self._ops = nn.ModuleList()
        self._step_nodes = nn.ModuleList()

        for name, index in zip(op_names, indices):
            op = OPS[name](C, L, self.args)
            self._ops += [op]

        # print(self._indices)
        # exit(0)
        self._indices = indices
        # embed()
        # exit(0)

        for gene_step_node in gene_step_nodes:
            step_node = Found_FusionNode(args.node_steps, args.node_multiplier, args, gene_step_node)
            self._step_nodes.append(step_node)

    def forward(self, input_features):

        states = []
        for input_feature in input_features:
            states.append(input_feature)

        # states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2*i]]
            h2 = states[self._indices[2*i+1]]

            op1 = self._ops[2*i]
            op2 = self._ops[2*i+1]
            
            h1 = op1(h1)
            h2 = op2(h2)
            
            s = self._step_nodes[i](h1, h2)
            states += [s]
        
        out = torch.cat(states[-self._multiplier:], dim=1)
        # out = self.bn(out)
        out = self.ln(out)
        out = F.relu(out)
        out = out.view(out.size(0), -1)

        # print("cell out shape:". out.shape)
        return out

class Found_FusionNetwork(nn.Module):

    def __init__(self, steps, multiplier, num_input_nodes, num_keep_edges, args, criterion, genotype):
        super().__init__()
        
        self._steps = steps
        self._multiplier = multiplier
        self._criterion = criterion
        self._genotype = genotype

        # input node number in a cell
        self._num_input_nodes = num_input_nodes
        self._num_keep_edges = num_keep_edges
        # self.drop_prob = args.drop_path_prob

        # self.cell = Found_FusionCell(steps, args, self._genotype)
        self.cell = Found_Random_FusionCell(steps, args, self._genotype)

    def forward(self, input_features):
        assert self._num_input_nodes == len(input_features)
        out = self.cell(input_features)
        return out

    def _loss(self, input_features, labels):
        logits = self(input_features)
        return self._criterion(logits, labels) 

    def get_genotype(self):
        return self._genotype
