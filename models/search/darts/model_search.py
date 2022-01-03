import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from .operations import *
from .genotypes import PRIMITIVES, Genotype
from .node_search import FusionNode

from IPython import embed

class FusionCell(nn.Module):
    def __init__(self, steps, multiplier, args):
        super(FusionCell, self).__init__()

        self._steps = steps
        self._multiplier = multiplier

        self._ops = nn.ModuleList()
        self.args = args

        self._step_nodes = nn.ModuleList()
        self.num_input_nodes = args.num_input_nodes
        self.C = args.C
        self.L = args.L
        self.ln = nn.LayerNorm([self.C * self._multiplier, self.L])

        # input features is a joint list of visual_features and skel_features
        for i in range(self._steps):
            for j in range(self.num_input_nodes+i):
                op = FusionMixedOp(self.C, self.L, self.args)
                self._ops.append(op)
        
        self._initialize_step_nodes(args)

    def _initialize_step_nodes(self, args):
        for i in range(self._steps):
            num_input = self.num_input_nodes + i
            # step_node = AttentionSumNode(args, num_input)
            step_node = FusionNode(args.node_steps, args.node_multiplier, args)
            self._step_nodes.append(step_node)

    def arch_parameters(self):
        self._arch_parameters = []
        for i in range(self._steps):
            self._arch_parameters += self._step_nodes[i].arch_parameters()
        return self._arch_parameters

    def forward(self, input_features, weights):
        states = []
        for input_feature in input_features:
            states.append(input_feature)

        offset = 0
        for i in range(self._steps):
            step_input_features = []
            step_input_feature = sum(self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
            s = self._step_nodes[i](step_input_feature, step_input_feature)
            offset += len(states)
            states.append(s)

        out = torch.cat(states[-self._multiplier:], dim=1)
        out = self.ln(out)
        out = F.relu(out)

        out = out.view(out.size(0), -1)
        return out

class FusionNetwork(nn.Module):

    def __init__(self, steps, multiplier, num_input_nodes, num_keep_edges, args, 
                criterion=None, logger=None):
        super().__init__()
        
        self.logger = logger
        self._steps = steps
        self._multiplier = multiplier
        self._criterion = criterion

        # input node number in a cell
        self._num_input_nodes = num_input_nodes
        self._num_keep_edges = num_keep_edges

        # self.cells = nn.ModuleList()
        self.cell = FusionCell(steps, multiplier, args)
        self.cell_arch_parameters = self.cell.arch_parameters()
        # self.cells += [cell]

        self._initialize_alphas()
        self._arch_parameters = [self.alphas_edges] + self.cell_arch_parameters

    def forward(self, input_features):
        assert self._num_input_nodes == len(input_features)
        weights = F.softmax(self.alphas_edges, dim=-1)
        out = self.cell(input_features, weights)
        return out

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(self._num_input_nodes+i))
        num_ops = len(PRIMITIVES)
        self.alphas_edges = Variable(1e-3*torch.randn(k, num_ops), requires_grad=True)
    
    def _loss(self, input_features, labels):
        logits = self(input_features)
        return self._criterion(logits, labels) 

    def arch_parameters(self):  
        return self._arch_parameters

    def genotype(self):
        def _parse(weights):
            gene = []
            # n = 2
            n = self._num_input_nodes
            start = 0

            # force non_repeat node pairs
            selected_edges = []
            selected_nodes = []

            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                # alpha edges, only link two most important nodes
                # edges = sorted(range(i + self._num_input_nodes), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:self._num_keep_edges]
                # from_list = list(range(i + self._num_input_nodes))
                
                # sample strategy v3
                from_list = list(range(self._num_input_nodes))

                node_pairs = []
                for j_index, j in enumerate(from_list):
                    for k in from_list[j_index+1:]:
                        # if [j, k] not in selected_edges:
                        if (j not in selected_nodes) or (k not in selected_nodes):

                            W_j_max = max(W[j][t] for t in range(len(W[j])) if t != PRIMITIVES.index('none'))
                            W_k_max = max(W[k][t] for t in range(len(W[k])) if t != PRIMITIVES.index('none'))

                            node_pairs.append([j, k, W_j_max * W_k_max])

                selected_node_pair = sorted(node_pairs, key=lambda x: -x[2])[:1][0]
                edges = selected_node_pair[0:2]
                selected_edges.append(edges)
                selected_nodes += edges
                selected_nodes = list(set(selected_nodes))
                #  choose the most important operation
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k != PRIMITIVES.index('none'):
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    gene.append((PRIMITIVES[k_best], j))
                    # gene.append((PRIMITIVES[k_second_best], j))
                start = end
                n += 1

            return gene

        def _parse_step_nodes():
            gene_steps = []
            for i in range(self._steps):
                step_node_genotype = self.cell._step_nodes[i].node_genotype()
                gene_steps.append(step_node_genotype)
            return gene_steps
        
        # beta edges
        gene_edges = _parse(F.softmax(self.alphas_edges, dim=-1).data.cpu().numpy())
        gene_steps = _parse_step_nodes()
        
        gene_concat = range(self._num_input_nodes+self._steps-self._multiplier, self._steps+self._num_input_nodes)
        gene_concat = list(gene_concat)

        genotype = Genotype(
            edges=gene_edges, 
            concat=gene_concat,
            steps=gene_steps
        )

        return genotype