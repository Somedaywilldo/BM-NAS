import torch
import torch.utils.data as data
from PIL import Image
# from spatial_transforms import *
import os
import math
import functools
import json
import copy
from numpy.random import randint
import numpy as np
import random
import glob

import pdb

from .nv_spatial_transforms import *
from .nv_temporal_transforms import *
from .nv_target_transforms import ClassLabel

def pil_loader(path, modality):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        #print(path)
        with Image.open(f) as img:
            if modality == 'RGB':
                return img.convert('RGB')
            elif modality == 'Flow':
                return img.convert('L')
            elif modality == 'Depth':
                return img.convert('L') # 8-bit pixels, black and white check from https://pillow.readthedocs.io/en/3.0.x/handbook/concepts.html


def accimage_loader(path, modality):
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def get_default_image_loader():
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader
    else:
        return pil_loader


def video_loader(video_dir_path, frame_indices, modality, sample_duration, image_loader):
    video = []
    if modality == 'RGB':
        for i in frame_indices:
            image_path = os.path.join(video_dir_path, '{:06d}.jpg'.format(i))
            if os.path.exists(image_path):
                video.append(image_loader(image_path, modality))
            else:
                print(image_path, "------- Does not exist")
                return video
    elif modality == 'Depth':

        for i in frame_indices:
            image_path = os.path.join(video_dir_path.rsplit(os.sep,2)[0] , 'Depth','depth' + video_dir_path[-1], '{:06d}.jpg'.format(i) )
            if os.path.exists(image_path):
                video.append(image_loader(image_path, modality))
            else:
                print(image_path, "------- Does not exist")
                return video
    elif modality == 'RGB-D':
        for i in frame_indices: # index 35 is used to change img to flow
            image_path = os.path.join(video_dir_path, '{:06d}.jpg'.format(i))
            image_path_depth = os.path.join(video_dir_path.rsplit(os.sep,2)[0] , 'Depth','depth' + video_dir_path[-1], '{:06d}.jpg'.format(i) )
    
            image = image_loader(image_path, 'RGB')
            image_depth = image_loader(image_path_depth, 'Depth')
            if os.path.exists(image_path):
                video.append(image)
                video.append(image_depth)
            else:
                print(image_path, "------- Does not exist")
                return video
    return video

def get_default_video_loader():
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader)


def load_annotation_data(data_file_path):
    with open(data_file_path, 'r') as data_file:
        return json.load(data_file)


def get_class_labels(data):
    class_labels_map = {}
    index = 0
    for class_label in data['labels']:
        class_labels_map[class_label] = index
        index += 1
    return class_labels_map


def get_video_names_and_annotations(data, subset):
    video_names = []
    annotations = []

    for key, value in data['database'].items():
        this_subset = value['subset']
        if this_subset in subset:
            label = value['annotations']['label']
            video_names.append(key.split('_')[0])
            annotations.append(value['annotations'])

    return video_names, annotations


def make_dataset(root_path, annotation_path, subset, n_samples_for_each_video,
                 sample_duration):
    if type(subset)==list:
        subset = subset
    else:
        subset =  [subset]
    data = load_annotation_data(annotation_path)
    video_names, annotations = get_video_names_and_annotations(data, subset)
    class_to_idx = get_class_labels(data)
    idx_to_class = {}
    for name, label in class_to_idx.items():
        idx_to_class[label] = name

    dataset = []
    list_subset = ''
    for x in subset:
        list_subset += x+',' 
    print("[INFO]: EgoGesture Dataset - " + list_subset + " is loading...")
    for i in range(len(video_names)):
        # if i % 1000 == 0:
        #     print('dataset loading [{}/{}]'.format(i, len(video_names)))

        video_path = os.path.join(root_path, video_names[i])
        
        if not os.path.exists(video_path):
            print(video_path + " does not exist")
            continue

        #### Add more frames from start end end
        begin_t = int(float(annotations[i]['start_frame']))
        end_t = int(float(annotations[i]['end_frame']))
        n_frames = end_t - begin_t + 1
        sample = {
            'video': video_path,
            'segment': [begin_t, end_t],
            'n_frames': n_frames,
            'video_id': i
        }
        if len(annotations) != 0:
            sample['label'] = class_to_idx[annotations[i]['label']]
        else:
            sample['label'] = -1

        if n_samples_for_each_video == 1:
            sample['frame_indices'] = list(range(begin_t, end_t + 1))
            dataset.append(sample)
        else:
            if n_samples_for_each_video > 1:
                step = max(1,
                           math.ceil((n_frames - 1 - sample_duration) /
                                     (n_samples_for_each_video - 1)))
            else:
                step = sample_duration
            for j in range(1, n_frames, step):
                sample_j = copy.deepcopy(sample)
                sample_j['frame_indices'] = list(
                    range(j, min(n_frames + 1, j + sample_duration)))
                dataset.append(sample_j)
    return dataset, idx_to_class


class EgoGesture(data.Dataset):
    """
    Args:
        root (string): Root directory path.
        spatial_transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        temporal_transform (callable, optional): A function/transform that  takes in a list of frame indices
            and returns a transformed version
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an video given its path and frame indices.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self,
                 args,
                 root_path,
                 annotation_path,
                 subset,
                 n_samples_for_each_video=1,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 sample_duration=16,
                 modality='RGB',
                 get_loader=get_default_video_loader):

        if subset == 'train_dev':
            subset = ['training', 'validation']

        self.data, self.class_names = make_dataset(
            root_path, annotation_path, subset, n_samples_for_each_video,
            sample_duration)

        if args.small_dataset:
            self.data = self.data[:64]

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.modality = modality
        self.sample_duration = sample_duration
        self.loader = get_loader()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """

        path = self.data[index]['video']

        frame_indices = self.data[index]['frame_indices']

        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)

        clip = self.loader(path, frame_indices, self.modality, self.sample_duration)
        oversample_clip =[]
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
        
        im_dim = clip[0].size()[-2:]
        clip = torch.cat(clip, 0).view((self.sample_duration, -1) + im_dim).permute(1, 0, 2, 3)
        
     
        target = self.data[index]
        if self.target_transform is not None:
            target = self.target_transform(target)

        
        return clip, target

    def __len__(self):
        return len(self.data)



def get_train_set(opt, spatial_transform, temporal_transform,
                     target_transform, args):
    subset = 'training'
    train_data = EgoGesture(
        args,
        opt.video_path,
        opt.annotation_path,
        subset,
        spatial_transform=spatial_transform,
        temporal_transform=temporal_transform,
        target_transform=target_transform,
        sample_duration=opt.sample_duration,
        modality=opt.modality
        # modality='RGB'
        )
    return train_data


def get_train_dev_set(opt, spatial_transform, temporal_transform,
                     target_transform, args):
    subset = 'train_dev'
    train_data = EgoGesture(
        args,
        opt.video_path,
        opt.annotation_path,
        subset,
        spatial_transform=spatial_transform,
        temporal_transform=temporal_transform,
        target_transform=target_transform,
        sample_duration=opt.sample_duration,
        modality=opt.modality
        # modality='RGB'
        )
    return train_data

def get_dev_set(opt, spatial_transform, temporal_transform, target_transform, args):
    dev_data = EgoGesture(
            args,
            opt.video_path,
            opt.annotation_path,
            'validation',
            1,
            spatial_transform,
            temporal_transform,
            target_transform,        
            # modality='Depth',
            modality=opt.modality,
            # modality='RGB',
            sample_duration=opt.sample_duration)
    return dev_data

def get_test_set(opt, spatial_transform, temporal_transform, target_transform, args):
    test_data = EgoGesture(
            args,
            opt.video_path,
            opt.annotation_path,
            'testing',
            1,
            spatial_transform,
            temporal_transform,
            target_transform,        
            # modality='Depth',
            modality=opt.modality,
            # modality='RGB',
            sample_duration=opt.sample_duration)
    return test_data

def get_train_loader(opt, args):
    # print("train loader opt:", opt)
    norm_method = Normalize(opt.mean, [1, 1, 1])
    crop_method = MultiScaleRandomCrop(opt.scales, opt.sample_size)
    spatial_transform = Compose([
        crop_method,
        ToTensor(opt.norm_value), 
        norm_method
    ])
    temporal_transform = TemporalRandomCrop(opt.sample_duration, opt.downsample)
    target_transform = ClassLabel()

    train_data = get_train_set(opt, spatial_transform, temporal_transform, target_transform, args)

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_threads,
        pin_memory=True
    )
    return train_loader

def get_train_dev_loader(opt, args):
    # print("train loader opt:", opt)
    norm_method = Normalize(opt.mean, [1, 1, 1])
    crop_method = MultiScaleRandomCrop(opt.scales, opt.sample_size)
    spatial_transform = Compose([
        crop_method,
        ToTensor(opt.norm_value), 
        norm_method
    ])
    temporal_transform = TemporalRandomCrop(opt.sample_duration, opt.downsample)
    target_transform = ClassLabel()

    train_dev_data = get_train_dev_set(opt, spatial_transform, temporal_transform, target_transform, args)

    train_dev_loader = torch.utils.data.DataLoader(
        train_dev_data,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_threads,
        pin_memory=True
    )
    return train_dev_loader

def get_dev_loader(opt, args):
    # print("test loader opt:", opt)
    norm_method = Normalize(opt.mean, [1, 1, 1])
    spatial_transform = Compose([
        Scale(opt.sample_size),
        CenterCrop(opt.sample_size),
        ToTensor(opt.norm_value), 
        norm_method
    ])
    temporal_transform = TemporalCenterCrop(opt.sample_duration, opt.downsample)
    target_transform = ClassLabel()

    dev_data = get_dev_set(opt, spatial_transform, temporal_transform,
                                     target_transform, args)

    dev_loader = torch.utils.data.DataLoader(
        dev_data,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_threads,
        pin_memory=True
    )
    return dev_loader

def get_test_loader(opt, args):
    # print("test loader opt:", opt)
    norm_method = Normalize(opt.mean, [1, 1, 1])
    spatial_transform = Compose([
        Scale(opt.sample_size),
        CenterCrop(opt.sample_size),
        ToTensor(opt.norm_value), 
        norm_method
    ])
    temporal_transform = TemporalCenterCrop(opt.sample_duration, opt.downsample)
    target_transform = ClassLabel()

    test_data = get_test_set(opt, spatial_transform, temporal_transform,
                                     target_transform, args)

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_threads,
        pin_memory=True
    )
    return test_loader

