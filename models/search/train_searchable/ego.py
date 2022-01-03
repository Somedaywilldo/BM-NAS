import torch
import models.auxiliary.scheduler as sc
import copy
from tqdm import tqdm
import os
from models.search.darts.utils import count_parameters, save, save_pickle

from IPython import embed
from collections import Counter
import pickle

# train model with darts mixed operations
def train_ego_track_acc(model, architect, criterion, optimizer, scheduler, dataloaders, dataset_sizes,
                device=None, num_epochs=200, parallel=False, logger=None,
                plotter=None, args=None, status='search'):

    best_genotype = None
    best_acc = 0
    best_epoch = 0
    
    best_test_genotype = None
    best_test_acc = 0
    best_test_epoch = 0
    # best_test_model_sd = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        # Each epoch has a training and validation phase
        logger.info("Epoch: {}".format(epoch) )
        logger.info("EXP: {}".format(args.save) )

        phases = []
        if status == 'search':
            phases = ['train', 'dev']
        else:
            # here the train is train + dev
            phases = ['train', 'test']

        for phase in phases:
            if phase == 'train':
                if not isinstance(scheduler, sc.LRCosineAnnealingScheduler):
                    scheduler.step()
                model.train()  # Set model to training mode
            elif phase == 'dev':
                if architect is not None:
                    architect.log_learning_rate(logger)
                model.train()
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            for param_group in optimizer.param_groups:
                logger.info("Learning Rate: {}".format(param_group['lr']))
                break

            with tqdm(dataloaders[phase]) as t:
                # Iterate over data.
                for data in dataloaders[phase]:
                    # get the inputs
                    inputs, labels = data

                    rgbs = inputs[:,0:3,:,:,:]
                    depths = inputs[:,3::,:,:,:]
                    # depths = inputs
                    
                    # device
                    rgbs = rgbs.to(device)
                    depths = depths.to(device)
                    labels = labels.to(device)
                    input_features = (rgbs, depths)

                    # updates darts cell
                    if status == 'search' and (phase == 'dev' or phase == 'test'):
                        if architect is not None:
                            architect.step(input_features, labels, logger)
                    
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train' or (phase == 'dev' and status == 'eval')):
                        output = model(input_features)
                        # print("labels:", labels)
                        _, preds = torch.max(output, 1)
                        # print("output:", preds)
                        loss = criterion(output, labels)
                        # backward + optimize only if in training phase
                        if phase == 'train' or (phase == 'dev' and status == 'eval'):
                            if isinstance(scheduler, sc.LRCosineAnnealingScheduler):
                                scheduler.step()
                                scheduler.update_optimizer(optimizer)
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * rgbs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                    batch_acc = torch.sum(preds == labels.data) * 1.0 / rgbs.size(0) 
                    postfix_str = 'batch_loss: {:.03f}, batch_acc: {:.03f}'.format(loss.item(), batch_acc)

                    t.set_postfix_str(postfix_str)
                    t.update()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc  = running_corrects.double() / dataset_sizes[phase]

            logger.info('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            
            print('dataset_size:', dataset_sizes[phase])

            genotype = None

            if parallel:
                num_params = 0
                for reshape_layer in model.module.reshape_layers:
                    num_params += count_parameters(reshape_layer)

                num_params += count_parameters(model.module.fusion_net)
                logger.info("Fusion Model Params: {}".format(num_params) )

                genotype = model.module.genotype()
            else:
                num_params = 0
                for reshape_layer in model.reshape_layers:
                    num_params += count_parameters(reshape_layer)

                num_params += count_parameters(model.fusion_net)
                logger.info("Fusion Model Params: {}".format(num_params) )

                genotype = model.genotype()
            logger.info(str(genotype))
            
            # deep copy the model
            if phase == 'dev' and epoch_acc >= best_acc:
                best_acc = epoch_acc
                # best_test_model_sd = copy.deepcopy(model.state_dict())
                best_genotype = copy.deepcopy(genotype)
                best_epoch = epoch

                if parallel:
                    save(model.module, os.path.join(args.save, 'best', 'best_model.pt'))
                else:
                    save(model, os.path.join(args.save, 'best', 'best_model.pt'))

                best_genotype_path = os.path.join(args.save, 'best', 'best_genotype.pkl')
                save_pickle(best_genotype, best_genotype_path)
                        
            # deep copy the model
            if phase == 'test' and epoch_acc >= best_test_acc:
                best_test_acc = epoch_acc
                # best_test_model_sd = copy.deepcopy(model.state_dict())
                best_test_genotype = copy.deepcopy(genotype)
                best_test_epoch = epoch

                if parallel:
                    save(model.module, os.path.join(args.save, 'best', 'best_test_model.pt'))
                else:
                    save(model, os.path.join(args.save, 'best', 'best_test_model.pt'))

                best_test_genotype_path = os.path.join(args.save, 'best', 'best_test_genotype.pkl')
                save_pickle(best_test_genotype, best_test_genotype_path)

        file_name = "epoch_{}".format(epoch)
        file_name = os.path.join(args.save, "architectures", file_name)

        plotter.plot(genotype, file_name, task='ego')
        logger.info("Current best dev accuracy: {}, at training epoch: {}".format(best_acc, best_epoch) )
        logger.info("Current best test accuracy: {}, at training epoch: {}".format(best_test_acc, best_test_epoch) )

    if status == 'search':
        return best_acc, best_genotype
    else:
        return best_test_acc, best_genotype

def test_ego_track_acc(model, dataloaders, criterion, genotype, 
                        dataset_sizes, device, logger, args):
    model.eval()
    # Each epoch has a training and validation phase
    logger.info("EXP: {}".format(args.save) )
    phase = 'test'

    running_loss = 0.0
    running_corrects = 0

    with tqdm(dataloaders[phase]) as t:
        # Iterate over data.
        for data in dataloaders[phase]:
            # get the inputs
            inputs, labels = data
            
            rgbs = inputs[:,0:3,:,:,:]
            depths = inputs[:,3::,:,:,:]

            # device
            rgbs = rgbs.to(device)
            depths = depths.to(device)
            labels = labels.to(device)
            input_features = (rgbs, depths)

            output = model(input_features)
            _, preds = torch.max(output, 1)
            loss = criterion(output, labels)

            # statistics
            running_loss += loss.item() * rgbs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            batch_acc = torch.sum(preds == labels.data) * 1.0 / rgbs.size(0) 
            postfix_str = 'batch_loss: {:.03f}, batch_acc: {:.03f}'.format(loss.item(), batch_acc)

            t.set_postfix_str(postfix_str)
            t.update()

    test_loss = running_loss / dataset_sizes[phase]
    test_acc  = running_corrects.double() / dataset_sizes[phase]
    
    logger.info(str(genotype))
    logger.info('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, test_loss, test_acc))
    return test_acc
