import torch
import torch.nn as nn
import torch.nn.functional as F

from .genotypes import *

OPS = {
    'none': lambda C, L, args: Zero(),
    'fc_relu': lambda C, L, args: FC_Relu(C, L, args),
    'fc_mish': lambda C, L, args: FC_Mish(C, L, args),
    'skip': lambda C, L, args: Identity()
}

class Zero(nn.Module):
    def __init__(self):
        super(Zero, self).__init__()

    def forward(self, x):
        out = x.mul(0.)
        return out

class FC_Relu(nn.Module):
    def __init__(self, C, L, args):
        super(FC_Relu, self).__init__()
        self.linear = nn.Linear(C, C)
        self.bn = nn.BatchNorm1d(C)
        self.dropout = nn.Dropout(args.drpt)

    def forward(self, x):
        # print(x.shape)
        out = x.transpose(1, 2)
        out = self.linear(out)
        out = out.transpose(1, 2)
        # print(out.shape)
        out = F.relu(out)
        out = self.bn(out)
        out = self.dropout(out)
        return out

class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        out = x * (torch.tanh(F.softplus(x)))
        return out 

class FC_Mish(nn.Module):
    def __init__(self, C, L, args):
        super(FC_Mish, self).__init__()
        self.linear = nn.Linear(C, C)
        self.mish = Mish()
        self.bn = nn.BatchNorm1d(C)
        self.dropout = nn.Dropout(args.drpt)

    def forward(self, x):
        # print(x.shape)
        out = x.transpose(1, 2)
        out = self.linear(out)
        out = out.transpose(1, 2)
        # print(out.shape)
        out = self.mish(out)
        out = self.bn(out)
        out = self.dropout(out)
        return out

# step node operation
class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads, drpt):
        super(Attention, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, drpt)
    def forward(self, q, k, v):
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)

        q = q.transpose(0, 2)
        k = k.transpose(0, 2)
        v = v.transpose(0, 2)

        out = self.attn(q, k, v, need_weights = False)[0]
        
        out = out.transpose(0, 2)
        out = out.transpose(0, 1)

        return out

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class FusionMixedOp(nn.Module):

    def __init__(self, C, L, args):
        super().__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            op = OPS[primitive](C, L, args)
            self._ops.append(op)

    def forward(self, x, weights):
        out = sum(w * op(x) for w, op in zip(weights, self._ops))
        return out