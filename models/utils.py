import torch
import numpy as np
import random
import os
import argparse

# use the next 3 functions to initial a model, eg at the third function
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        if m.bias is not None:
            m.bias.data.fill_(0)

    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        if m.bias is not None:
            m.bias.data.fill_(0)

    elif classname.find('LSTM') != -1:
        for name, param in m.named_parameters():
            if 'bias' in name:
                torch.nn.init.constant(param, 0.0)
            elif 'weight' in name:
                torch.nn.init.orthogonal(param)

        # Initialize biases for LSTM’s forget gate to 1 to remember more by default. Similarly, initialize biases for GRU’s reset gate to -1.
        for names in m._all_weights:
            for name in filter(lambda n: "bias" in n, names):
                bias = getattr(m, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data[start:end].fill_(1.)

    elif classname.find('GRU') != -1:
        for name, param in m.named_parameters():
            if 'bias' in name:
                torch.nn.init.constant(param, 0.0)
            elif 'weight' in name:
                torch.nn.init.orthogonal(param)

def initial_model_weight(layers):
    for layer in layers:
        if list(layer.children()) == []:
            weights_init(layer)
            # print('weight initial finished!')
        else:
            for sub_layer in list(layer.children()):
                initial_model_weight([sub_layer])

class Cutout(object):
    # https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """

    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = random.randint(0, h - 1)
            x = random.randint(0, w - 1)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask
        # i = np.transpose(img.cpu().numpy(), (1,2,0))
        # i[:,:,0] *= 0.2023
        # i[:,:,1] *= 0.1994
        # i[:,:,2] *= 0.2010
        # i[:,:,0] += 0.4914
        # i[:,:,1] += 0.4822
        # i[:,:,2] += 0.4465
        # print(i.shape)
        # plot.imshow(i)
        # plot.imsave('aa{}.png'.format(random.randint(0,100)), i)

        return img

    def __repr__(self):
        return self.__class__.__name__ + '(num_output_channels={0})'.format(self.num_output_channels)

def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)
    print('Experiment dir : {}'.format(path))
    
    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)

# for resnext-101 pretrained model
def parse_opts(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', default='/mnt/scratch/xiaoxiang/yihang/NVGesture/nvgesture_arch', type=str, help='Directory path of Videos')
    parser.add_argument('--annotation_path', default='kinetics.json', type=str, help='Annotation file path')
    parser.add_argument('--store_name', default='model', type=str, help='Name to store checkpoints')
    parser.add_argument('--modality', default='RGB', type=str, help='Modality of generated model. RGB, Flow or RGBFlow')
    parser.add_argument('--dataset', default='kinetics', type=str, help='Used dataset (activitynet | kinetics | ucf101 | hmdb51)')
    parser.add_argument('--n_classes', default=400, type=int, help='Number of classes (activitynet: 200, kinetics: 400, ucf101: 101, hmdb51: 51)')
    parser.add_argument('--n_finetune_classes', default=400, type=int, help='Number of classes for fine-tuning. n_classes is set to the number when pretraining.')
    parser.add_argument('--sample_size', default=112, type=int, help='Height and width of inputs')
    parser.add_argument('--sample_duration', default=32, type=int, help='Temporal duration of inputs')
    parser.add_argument('--downsample', default=1, type=int, help='Downsampling. Selecting 1 frame out of N')
    parser.add_argument('--initial_scale', default=1.0, type=float, help='Initial scale for multiscale cropping')
    parser.add_argument('--n_scales', default=5, type=int, help='Number of scales for multiscale cropping')
    parser.add_argument('--scale_step', default=0.84089641525, type=float, help='Scale step for multiscale cropping')
    parser.add_argument('--train_crop', default='random', type=str, help='Spatial cropping method in training. random is uniform. corner is selection from 4 corners and 1 center.  (random | corner | center)')
    parser.add_argument('--learning_rate', default=0.04, type=float, help='Initial learning rate (divided by 10 while training by lr scheduler)')
    parser.add_argument('--lr_steps', default=[15, 25, 35, 45, 60, 50, 200, 250], type=float, nargs="+", metavar='LRSteps', help='epochs to decay learning rate by 10') # [15, 30, 37, 50, 200, 250]
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument('--dampening', default=0.9, type=float, help='dampening of SGD')
    parser.add_argument('--weight_decay', default=1e-3, type=float, help='Weight Decay')
    parser.add_argument('--mean_dataset', default='activitynet', type=str, help='dataset for mean values of mean subtraction (activitynet | kinetics)')
    parser.add_argument('--no_mean_norm', action='store_true', help='If true, inputs are not normalized by mean.')
    parser.set_defaults(no_mean_norm=False)
    parser.add_argument('--std_norm', action='store_true', help='If true, inputs are normalized by standard deviation.')
    parser.set_defaults(std_norm=False)
    parser.add_argument('--nesterov', action='store_true', help='Nesterov momentum')
    parser.set_defaults(nesterov=False)
    parser.add_argument('--optimizer', default='sgd', type=str, help='Currently only support SGD')
    parser.add_argument('--lr_patience', default=10, type=int, help='Patience of LR scheduler. See documentation of ReduceLROnPlateau.')

    parser.add_argument('--batch_size', default=128, type=int, help='Batch Size')
    # parser.add_argument('--test_batch_size', default=8, type=int, help='Batch Size')

    parser.add_argument('--n_epochs', default=250, type=int, help='Number of total epochs to run')
    parser.add_argument('--begin_epoch', default=1, type=int, help='Training begins at this epoch. Previous trained model indicated by resume_path is loaded.')
    parser.add_argument('--n_val_samples', default=1, type=int, help='Number of validation samples for each activity')
    parser.add_argument('--ft_portion', default='complete', type=str, help='The portion of the model to apply fine tuning, either complete or last_layer')
    parser.add_argument('--no_train', action='store_true', help='If true, training is not performed.')
    parser.set_defaults(no_train=False)
    parser.add_argument('--no_val', action='store_true', help='If true, validation is not performed.')
    parser.set_defaults(no_val=False)
    parser.add_argument('--test', action='store_true', help='If true, test is performed.')
    parser.set_defaults(test=False)
    parser.add_argument('--test_subset', default='val', type=str, help='Used subset in test (val | test)')
    parser.add_argument('--scale_in_test', default=1.0, type=float, help='Spatial scale in test')
    parser.add_argument('--crop_position_in_test', default='c', type=str, help='Cropping method (c | tl | tr | bl | br) in test')
    parser.add_argument('--no_softmax_in_test', action='store_true', help='If true, output for each clip is not normalized using softmax.')
    parser.set_defaults(no_softmax_in_test=False)
    parser.add_argument('--n_threads', default=16, type=int, help='Number of threads for multi-thread loading')
    parser.add_argument('--checkpoint', default=10, type=int, help='Trained model is saved at every this epochs.')
    parser.add_argument('--no_hflip', action='store_true', help='If true holizontal flipping is not performed.')
    parser.set_defaults(no_hflip=False)
    parser.add_argument('--norm_value', default=1, type=int, help='If 1, range of inputs is [0-255]. If 255, range of inputs is [0-1].')
    parser.add_argument('--model', default='resnet', type=str, help='(resnet | preresnet | wideresnet | resnext | densenet | ')
    parser.add_argument('--version', default=1.1, type=float, help='Version of the model')
    parser.add_argument('--model_depth', default=101, type=int, help='Depth of resnet (10 | 18 | 34 | 50 | 101)')
    parser.add_argument('--resnet_shortcut', default='B', type=str, help='Shortcut type of resnet (A | B)')
    parser.add_argument('--wide_resnet_k', default=2, type=int, help='Wide resnet k')
    parser.add_argument('--resnext_cardinality', default=32, type=int, help='ResNeXt cardinality')
    parser.add_argument('--groups', default=1, type=int, help='The number of groups at group convolutions at conv layers')
    parser.add_argument('--width_mult', default=1.0, type=float, help='The applied width multiplier to scale number of filters')
    parser.add_argument('--manual_seed', default=1, type=int, help='Manually set random seed')
    parser.add_argument('--train_validate', action='store_true', help='If true, test is performed.')
    parser.set_defaults(train_validate=False)
    
    parser.add_argument('--scales', default=None)
    parser.add_argument('--mean', default=None)
    
    opt = parser.parse_args("")

    
    opt.scales = [1.0, 0.84089641525, 0.7071067811803005, 0.5946035574934808, 0.4999999999911653]
    opt.mean = [114.7748, 107.7354, 99.475]    
    
    opt.video_path = args.datadir
    opt.annotation_path = os.path.join(args.checkpointdir, args.annotation)
    opt.batch_size = args.batchsize
    # opt.test_batch_size = args.test_batchsize
    opt.n_classes = args.num_outputs
    opt.n_finetune_classes = args.num_outputs
    # opt.groups = args.groups
    # opt.n_val_samples = args.n_val_samples
    # parser.add_argument('--groups', type=int, help='for resnext', default=1)
    # parser.add_argument('--n_val_samples', type=int, help='for resnext', default=1)

    opt.model = 'resnext'
    opt.model_depth = 101
    opt.modality = 'RGB-D'

    # opt.dataset = 'egogesture'

    return opt