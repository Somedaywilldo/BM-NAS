import torch
import models.auxiliary.scheduler as sc
import copy
from sklearn.metrics import f1_score 
from tqdm import tqdm
import os
from IPython import embed
from models.search.darts.utils import count_parameters, save, save_pickle

def train_mmimdb_track_f1(  model, architect,
                            criterion, optimizer, scheduler, dataloaders,
                            dataset_sizes, device, num_epochs, 
                            parallel, logger, plotter, args,
                            f1_type='weighted', init_f1=0.0, th_fscore=0.3, 
                            status='search'):

    best_genotype = None
    best_f1 = init_f1
    best_epoch = 0

    best_test_genotype = None
    best_test_f1 = init_f1
    best_test_epoch = 0

    failsafe = True
    cont_overloop = 0
    while failsafe:
        for epoch in range(num_epochs):
            
            logger.info('Epoch: {}'.format(epoch))            
            logger.info("EXP: {}".format(args.save) )

            phases = []
            if status == 'search':
                phases = ['train', 'dev']
            else:
                # while evaluating, add dev set to train also
                phases = ['train', 'dev', 'test']
            
            # Each epoch has a training and validation phase
            for phase in phases:
                if phase == 'train':
                    if not isinstance(scheduler, sc.LRCosineAnnealingScheduler):
                        scheduler.step()
                    if architect is not None:
                        architect.log_learning_rate(logger)
                    model.train()  # Set model to training mode
                    list_preds = [] 
                    list_label = []        
                elif phase == 'dev':
                    if status == 'eval':
                        if not isinstance(scheduler, sc.LRCosineAnnealingScheduler):
                            scheduler.step()
                    model.train()
                    list_preds = [] 
                    list_label = []          
                else:
                    model.eval()  # Set model to evaluate mode
                    list_preds = [] 
                    list_label = []                    
    
                running_loss = 0.0
                running_f1 = init_f1

                with tqdm(dataloaders[phase]) as t:
                    # Iterate over data.
                    for data in dataloaders[phase]:
                                    
                        # get the inputs
                        image, text, label = data['image'], data['text'], data['label']
        
                        # device
                        image = image.to(device)
                        text = text.to(device)                
                        label = label.to(device)

                        if status == 'search' and (phase == 'dev' or phase == 'test'):
                            architect.step((text, image), label, logger)
        
                        # zero the parameter gradients
                        optimizer.zero_grad()
        
                        # forward
                        # track history if only in train
                        with torch.set_grad_enabled(phase == 'train' or (phase == 'dev' and status == 'eval')):
                            output = model((text, image))
                            
                            if isinstance(output, tuple):
                                output = output[-1]
        
                            _, preds = torch.max(output, 1)
                            loss = criterion(output, label)
                            preds_th = torch.sigmoid(output) > th_fscore

                            # backward + optimize only if in training phase
                            if phase == 'train' or (phase == 'dev' and status == 'eval'):
                                if isinstance(scheduler, sc.LRCosineAnnealingScheduler):
                                    scheduler.step()
                                    scheduler.update_optimizer(optimizer)
                                loss.backward()
                                optimizer.step()
                            
                            # if phase == 'dev':
                            list_preds.append(preds_th.cpu())
                            list_label.append(label.cpu()) 

                        # statistics
                        running_loss += loss.item() * image.size(0)

                        batch_pred_th = preds_th.data.cpu().numpy()
                        batch_true = label.data.cpu().numpy()

                        batch_f1 = f1_score(batch_pred_th, batch_true, average=f1_type, zero_division=1)                  

                        postfix_str = 'batch_loss: {:.03f}, batch_f1: {:.03f}'.format(loss.item(), batch_f1)
                        t.set_postfix_str(postfix_str)
                        t.update()
                            
                epoch_loss = running_loss / dataset_sizes[phase]
                
                y_pred = torch.cat(list_preds, dim=0).numpy()
                y_true = torch.cat(list_label, dim=0).numpy()
                
                # epoch_f1 = f1_score(y_true, y_pred, average='macro', zero_division=1)  
                epoch_f1 = f1_score(y_true, y_pred, average=f1_type, zero_division=1)                  
                
                logger.info('{} Loss: {:.4f}, {} F1: {:.4f}'.format(
                    phase, epoch_loss, f1_type, epoch_f1))
                
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

                if phase == 'train' and epoch_loss != epoch_loss:
                    logger.info("Nan loss during training, escaping")
                    model.eval()              
                    return best_f1
                
                if phase == 'dev' and status == 'search':
                    if epoch_f1 > best_f1:
                        best_f1 = epoch_f1
                        best_genotype = copy.deepcopy(genotype)
                        best_epoch = epoch
                        # best_model_sd = copy.deepcopy(model.state_dict())
                    
                        if parallel:
                            save(model.module, os.path.join(args.save, 'best', 'best_model.pt'))
                        else:
                            save(model, os.path.join(args.save, 'best', 'best_model.pt'))

                        best_genotype_path = os.path.join(args.save, 'best', 'best_genotype.pkl')
                        save_pickle(best_genotype, best_genotype_path)
                
                if phase == 'test':
                    if epoch_f1 > best_test_f1:
                        best_test_f1 = epoch_f1
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
            plotter.plot(genotype, file_name, task='mmimdb')

            logger.info("Current best dev {} F1: {}, at training epoch: {}".format(f1_type, best_f1, best_epoch) )
            logger.info("Current best test {} F1: {}, at training epoch: {}".format(f1_type, best_test_f1, best_test_epoch) )

        if best_f1 != best_f1 and num_epochs == 1 and cont_overloop < 1:
            failsafe = True
            logger.info('Recording a NaN F1, training for one more epoch.')
        else:
            failsafe = False
            
        cont_overloop += 1
    
    if best_f1 != best_f1:
        best_f1 = 0.0

    if status == 'search':
        return best_f1, best_genotype
    else:
        return best_test_f1, best_test_genotype

def test_mmimdb_track_f1(  model, criterion, dataloaders,
                           dataset_sizes, device, 
                           parallel, logger, args,
                           f1_type='weighted', init_f1=0.0, th_fscore=0.3):

    best_test_genotype = None
    best_test_f1 = init_f1
    best_test_epoch = 0
    
    model.eval()  # Set model to evaluate mode
    list_preds = [] 
    list_label = []                    

    running_loss = 0.0
    running_f1 = init_f1
    phase = 'test'

    with tqdm(dataloaders[phase]) as t:
        # Iterate over data.
        for data in dataloaders[phase]:
            # get the inputs
            image, text, label = data['image'], data['text'], data['label']
            # device
            image = image.to(device)
            text = text.to(device)                
            label = label.to(device)

            output = model((text, image))        
            if isinstance(output, tuple):
                output = output[-1]

            _, preds = torch.max(output, 1)
            loss = criterion(output, label)
            preds_th = torch.sigmoid(output) > th_fscore
            # if phase == 'dev':
            list_preds.append(preds_th.cpu())
            list_label.append(label.cpu()) 

            # statistics
            running_loss += loss.item() * image.size(0)

            batch_pred_th = preds_th.data.cpu().numpy()
            batch_true = label.data.cpu().numpy()
            batch_f1 = f1_score(batch_pred_th, batch_true, average='samples')  

            postfix_str = 'batch_loss: {:.03f}, batch_f1: {:.03f}'.format(loss.item(), batch_f1)
            t.set_postfix_str(postfix_str)
            t.update()
                
    epoch_loss = running_loss / dataset_sizes[phase]
    
    # if phase == 'dev':
    y_pred = torch.cat(list_preds, dim=0).numpy()
    y_true = torch.cat(list_label, dim=0).numpy()

    epoch_f1 = f1_score(y_true, y_pred, average=f1_type)                  

    logger.info('{} Loss: {:.4f}, {} F1: {:.4f}'.format(
                    phase, epoch_loss, f1_type, epoch_f1))
    
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
    best_test_f1 = epoch_f1
    return best_test_f1