import torch.nn as nn
from .node_operations import *
from .operations import *
from .node_operations import *

from IPython import embed

class Found_NodeCell(nn.Module):
    def __init__(self, node_steps, node_multiplier, args, step_genotype):
        super().__init__()
        self.args = args
        self.node_steps = node_steps
        self.node_multiplier = node_multiplier
        
        self.edge_ops = nn.ModuleList()
        self.node_ops = nn.ModuleList()
        
        self.C = args.C
        self.L = args.L
        
        self.num_input_nodes = 2

        op_names, indices = zip(*step_genotype.inner_edges)
        inner_steps = step_genotype.inner_steps
        self.compile(op_names, indices, inner_steps)

        if self.node_multiplier != 1:
            self.out_conv = nn.Conv1d(self.C * self.node_multiplier, self.C, 1, 1)
            self.bn = nn.BatchNorm1d(self.C)
            self.out_dropout = nn.Dropout(args.drpt)
        
        # skip v3 and v4
        self.ln = nn.LayerNorm([self.C, self.L])
        self.dropout = nn.Dropout(args.drpt)

    def compile(self, edge_op_names, edge_indices, inner_steps):
        for name in edge_op_names:
            edge_op = OPS[name](self.C, self.L, self.args)
            self.edge_ops += [edge_op]
        self.edge_indices = edge_indices
        for name in inner_steps:
            node_op = STEP_STEP_OPS[name](self.C, self.L, self.args)
            self.node_ops.append(node_op)

    def forward(self, x, y):
        states = [x, y]
        
        offset = 0
        for i in range(self.node_steps):
            
            input_x = states[self.edge_indices[2*i]]
            input_y = states[self.edge_indices[2*i+1]]
            
            input_x = self.edge_ops[2*i](input_x)
            input_y = self.edge_ops[2*i+1](input_y)
            
            s = self.node_ops[i](input_x, input_y)
            offset += len(states)
            states.append(s)

        # # inner output step is sum
        # out = sum(states[-self.node_multiplier:])

        # inner output step is cat_conv_relu
        out = torch.cat(states[-self.node_multiplier:], dim=1)
    
        if self.node_multiplier != 1:
            out = self.out_conv(out)
            out = self.bn(out)
            out = F.relu(out)
            out = self.out_dropout(out)
        # # skip v4
        out += x
        out = self.ln(out)

        return out

class Found_FusionNode(nn.Module):
    def __init__(self, node_steps, node_multiplier, args, step_genotype):
        super().__init__()
        # self.logger = logger
        self.node_steps = node_steps
        self.node_multiplier = node_multiplier
        self.node_cell = Found_NodeCell(node_steps, node_multiplier, args, step_genotype)

        self.num_input_nodes = 2
        self.num_keep_edges = 2
        
    def forward(self, x, y):
        out = self.node_cell(x, y)        
        return out


class Found_DARTS_FusionNode(nn.Module):
    def __init__(self, node_steps, node_multiplier, args, step_genotype):
        super().__init__()
        # self.logger = logger
        self.node_steps = node_steps
        self.node_multiplier = node_multiplier
        self.num_input_nodes = 2
        self.num_keep_edges = 2
        
    def forward(self, x, y):
        out = x + y
        return out

class Found_MFAS_FusionNode(nn.Module):
    def __init__(self, node_steps, node_multiplier, args, step_genotype):
        super().__init__()
        # self.logger = logger
        self.node_steps = node_steps
        self.node_multiplier = node_multiplier
        self.num_input_nodes = 2
        self.num_keep_edges = 2

        self.C = args.C

        self.conv = nn.Conv1d(self.C * 2, self.C, 1, 1)
        self.bn = nn.BatchNorm1d(self.C)
        self.dropout = nn.Dropout(args.drpt)
        
    def forward(self, x, y):
        # inner output step is cat_conv_relu
        out = torch.cat([x, y], dim=1)
        out = self.conv(out)
        out = self.bn(out)
        out = F.relu(out)
        out = self.dropout(out)
        return out


class Found_AOA_FusionNode(nn.Module):
    def __init__(self, node_steps, node_multiplier, args, step_genotype):
        super().__init__()
        # self.logger = logger
        self.node_steps = node_steps
        self.node_multiplier = node_multiplier
        self.num_input_nodes = 2
        self.num_keep_edges = 2

        self.C = args.C
        self.L = args.L

        self.scale_dot_attn = STEP_STEP_OPS['scale_dot_attn'](self.C, self.L, args)
        self.cat_conv_glu = STEP_STEP_OPS['cat_conv_glu'](self.C, self.L, args)

    def forward(self, x, y):
        # inner output step is cat_conv_relu
        out1 = self.scale_dot_attn(x, y)
        out = self.cat_conv_glu(x, out1)
        return out


class Found_TwoHeadAttn_FusionNode(nn.Module):
    def __init__(self, node_steps, node_multiplier, args, step_genotype):
        super().__init__()
        # self.logger = logger
        self.node_steps = node_steps
        self.node_multiplier = node_multiplier
        self.num_input_nodes = 2
        self.num_keep_edges = 2

        self.C = args.C
        self.L = args.L

        self.scale_dot_attn1 = STEP_STEP_OPS['scale_dot_attn'](self.C, self.L, args)
        self.scale_dot_attn2 = STEP_STEP_OPS['scale_dot_attn'](self.C, self.L, args)

        self.conv = nn.Conv1d(self.C * 2, self.C, 1, 1)
        self.bn = nn.BatchNorm1d(self.C)
        self.dropout = nn.Dropout(args.drpt)
        
    def forward(self, x, y):
        # inner output step is cat_conv_relu
        out1 = self.scale_dot_attn1(x, y)
        out2 = self.scale_dot_attn2(x, y)
        
        out = torch.cat([out1, out2], dim=1)
        out = self.conv(out)
        out = self.bn(out)
        out = F.relu(out)
        out = self.dropout(out)

        return out