from collections import namedtuple

Genotype = namedtuple('Genotype', 'edges steps concat')
StepGenotype = namedtuple('StepGenotype', 'inner_edges inner_steps inner_concat')

PRIMITIVES = [
    'none',
    'skip'
]

STEP_EDGE_PRIMITIVES = [
    'none',
    'skip'
]

STEP_STEP_PRIMITIVES = [
    'Sum',
    'ScaleDotAttn',
    'LinearGLU',
    'ConcatFC'
]


# STEP_STEP_PRIMITIVES = [
#     'sum',
#     'scale_dot_attn',
#     'cat_conv_glu',
#     'cat_conv_relu'
# ]

# new_op_dict = {
#     'sum': 'Sum', 
#     'scale_dot_attn': 'ScaleDotAttn',
#     'cat_conv_glu': 'LinearGLU',
#     'cat_conv_relu': 'ConcatFC'
# }