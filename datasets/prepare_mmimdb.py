import json
import h5py
import numpy as np
import os
from IPython import embed

# dataset_h5 = h5py.File('/mnt/scratch/xiaoxiang/yihang/mmimdb/multimodal_imdb.hdf5', 'r')
dataset_h5 = h5py.File('<path to multimodal_imdb.hdf5>', 'r')
dataset_len = dataset_h5['features'].shape[0]

# save_dir = '/mnt/scratch/xiaoxiang/yihang/mmimdb/debug'
save_dir = "<dir to save mmimdb dataset>"
split_file = open('../checkpoints/mmimdb/mmimdb_split.json', 'r')
split_json = json.load(split_file)

train_len = 0
dev_len = 0
test_len = 0

os.makedirs(os.path.join(save_dir, 'train'))
os.makedirs(os.path.join(save_dir, 'dev'))
os.makedirs(os.path.join(save_dir, 'test'))

for i in range(dataset_len):
    # save img as .npy
    imdb_id = dataset_h5['imdb_ids'][i].decode('UTF-8')
    image = dataset_h5['images'][i]
    text = dataset_h5['features'][i]
    label = dataset_h5['genres'][i]
    
    stage = ''
    if imdb_id in split_json['train']:
        stage = 'train'
        train_len += 1
    elif imdb_id in split_json['dev']:
        stage = 'dev'
        dev_len += 1
    else:
        stage = 'test'
        test_len += 1
    
    data_id = split_json[stage].index(imdb_id)
    
    image_path = os.path.join(save_dir, stage, 'image_{:06}'.format(data_id))
    text_path = os.path.join(save_dir, stage, 'text_{:06}'.format(data_id))
    label_path = os.path.join(save_dir, stage, 'label_{:06}'.format(data_id))
    
    print("Processing:", i, imdb_id)
    np.save(label_path, label)
    np.save(image_path, image)
    np.save(text_path, text)

    # embed()
    # exit(0)

# check data
assert len(split_json['train']) == train_len
assert len(split_json['dev']) == dev_len
assert len(split_json['test']) == test_len

print("MM-IMDB dataset is prepared successfully!")