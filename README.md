# BM-NAS



## Dataset Preparation

```
pytorch
opencv
sklearn
tqdm
IPython
```


https://www.aliyundrive.com/s/1c55RCPdaAo

make dir final_exp

### MM-IMDB

First, download the MM-IMDB dataset from the official [site](http://lisi1.unal.edu.co/mmimdb/).

We use the ready-to-use fuel dataset **multimodal_imdb.hdf5**, since the text features are already converted to embedding vectors. We put the official split in **checkpoints/** as the **mmimdb_split.json** so you don't need to download the raw dataset.

Then, run the following script:
```shell
python datasets/prepare_mmimdb.py
```

=['LinearGLU'], inner_concat=[2]), StepGenotype(inner_edges=[('skip', 1), ('skip', 0)], inner_steps=['LinearGLU'], inner_concat=[2])], concat=[6, 7])
12/03 10:04:23 PM Current best dev F1: 0.0, at training epoch: 0
12/03 10:04:23 PM Current best test F1: 0.6291488517332355, at training epoch: 12
12/03 10:04:23 PM Epoch: 13
12/03 10:04:23 PM EXP: final_exp/mmimdb/search-C192-L16-S2-M2-NS1-NM1-drpt0.1-20201203-214527/eval-tune_backbones-20201203-215640
MACRO F1
0.5479939980034221
Experiment dir : final_exp/mmimdb/search-C192-L16-S2-M2-NS1-NM1-drpt0.1-20201203-214527/eval-tune_backbones-20201203-215640/test-tune_backbones-20201203-222520
12/03 10:25:20 PM args = Namespace(C=192, L=16, Ti=5, Tm=2, arch_learning_rate=0.0003, arch_weight_decay=0.001, average_text=False, batchnorm=False, batchsize=128, datadir='/mnt/scratch/xiaoxiang/yihang/mmimdb/', drpt=0.1, epochs=50, eta_max=0.001, eta_min=1e-06, eval_exp_dir='final_exp/mmimdb/search-C192-L16-S2-M2-NS1-NM1-drpt0.1-20201203-214527/eval-tune_backbones-20201203-215640', modality='both', momentum=0.9, multiplier=2, no_bad_skel=True, node_multiplier=1, node_steps=1, num_input_nodes=6, num_keep_edges=2, num_outputs=23, num_workers=32, save='final_exp/mmimdb/search-C192-L16-S2-M2-NS1-NM1-drpt0.1-20201203-214527/eval-tune_backbones-20201203-215640/test-tune_backbones-20201203-222520', search_exp_dir='final_exp', seed=2, small_dataset=False, steps=2, unrolled=False, use_dataparallel=True, verbose=True, weight_decay=0.0003)
100%|█████████████████████████████████████████████████████████████████████████████████████| 61/61 [00:28<00:00,  2.12it/s, batch_loss: 0.190, batch_f1: 0.638]
12/03 10:25:54 PM test Loss: 0.1898 F1: 0.5480
12/03 10:25:54 PM Fusion Model Params: 532416
12/03 10:25:54 PM Genotype(edges=[('skip', 0), ('skip', 4), ('skip', 1), ('skip', 4)], steps=[StepGenotype(inner_edges=[('skip', 0), ('skip', 1)], inner_steps=['LinearGLU'], inner_concat=[2]), StepGenotype(inner_edges=[('skip', 1), ('skip', 0)], inner_steps=['LinearGLU'], inner_concat=[2])], concat=[6, 7])
12/03 10:25:54 PM Final Test F1: 0.5479939980034221


### NTU

First, request and download the NTU RGB+D dataset (not NTU RGB+D 120) from the official [site](http://rose1.ntu.edu.sg/datasets/actionrecognition.asp). 

We only use the **3D skeletons (body joints)** and **RGB videos** modality. 

Then, run the following script to reshape all RGB videos to 256x256 with 30 fps:
```shell
python datasets/prepare_ntu.py --dir=<dir of RGB videos>
```

### EgoGesture

First, request and download the EgoGesture dataset from the official [site](http://www.nlpr.ia.ac.cn/iva/yfzhang/datasets/egogesture.html). 

We use the extracted images provided by the original EgoGesture dataset directly. The label information is in **checkpoints/egogestureall_but_None.json**, provided by [Real-time-GesRec](https://github.com/ahmetgunduz/Real-time-GesRec). 


## Model Checkpoints

### Pretrained Backbones

### Our Models



Ego的Test模式还没有写好



