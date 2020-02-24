# import necessary libraries
import torch
import time
import math
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import confusion_matrix
from utils import normalize_confusion_matrix

import numpy as np
from sklearn.metrics import f1_score

def get_stat(model, loader, loss_fn):
    '''
    evaluate model on given loader

    Inputs
        - model: a model to get statistics
        - loader: data loader
        - loss_fn: loss function used to calculate
    Outputs
        - loss: loss value
        - acc: accuracy of the model
        - f1: f1 score of the model
    '''
    X, y = iter(loader).next()
        
    pred = model(X) # get model prediction
    loss = round(loss_fn(pred, y).item(), 4) # get loss and round by 4 decimal points
    correct = (y == torch.argmax(pred, dim=1)).sum().item() # count number of correct predictions
    acc = round(correct*100/X.shape[0], 4) # calculate accuracy and round by 4 decimal points
    f1 = f1_score(y, torch.argmax(pred, axis=1), average='macro') # calculate f1 score
    
    return loss, acc, f1

def train(model, train_loader, val_loader, epochs, loss_fn, batch_size, lr=1e-3, optimizer='adam', weight_decay=0, momentum=None, seed=1234):
    '''
    train the given model according to the provided configuration

    Inputs
        - model: model initialized for training
        - train_loader: dataloader for training set
        - val_loader: dataloader for validation set
        - epochs: number of epoch for training
        - loss_fn: loss function used for training
        - batch_size: number of batch size used
        - lr: learning rate of an optimizer
        - optimizer: choice of optimizer used for training ('adam', 'sgd', or 'momentum' only)
        - weight_decay: weight decay of the parameters
        - momentum: value of momentum for 'momentum' optimizer
        - seed: random seed of `train` function
    Outputs
        model: trained model
        history: history of training, used for plotting training step loss/acc
    '''
    model.train()
    step_loss_train, step_acc_train, step_f1_train  = list(), list(), list()
    epoch_loss_val, epoch_acc_val, epoch_f1_val = list(), list(), list()
    
    total_params = sum(p.numel() for p in model.parameters()) # get number of total parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) # get number of trainable parameters
    
    N = np.sum([x[0].shape[0] for x in train_loader]) # get number of training set size
    num_step = len(train_loader) # get number of training step per epoch
    total_correct = 0
    print('Start Training...')
    print('='*10,'Hyperparameters','='*10)
    print('Batch size: {}\nEpochs: {}\nOptimizer: {}\nLearning Rate: {}'.format(batch_size, epochs, optimizer, lr))
    print('Number of Parameters: {}'.format(total_params))
    print('Number of Trainable Parameters: {}'.format(trainable_params))
    print('='*37)

    # define the optimizer used for training
    if optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer == 'momentum':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    start_time = time.time()
    
    # get initial validation statistic
    val_loss, val_acc, val_f1 = get_stat(model, val_loader, loss_fn)
    val_f1 = round(val_f1, 4)
    print('>'*5, 'EPOCH 0: validation loss: {}, validation accuracy: {}%, validation f1: {}'.format(val_loss, val_acc, val_f1), '<'*5)
        
    epoch_loss_val.append(val_loss)
    epoch_acc_val.append(val_acc)
    epoch_f1_val.append(val_f1)
    # iterate over number of epochs
    for epoch in range(epochs):
        num_step = 0
        print('EPOCH {}/{}'.format(epoch+1, epochs))
        # iterate over each training step in each epoch
        # also iterate over each batch in train_loader
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            
            batch, y_true = data
            n = batch.shape[0]

            y_pred = model(batch) # get prediction
            loss = loss_fn(y_pred, y_true) # calculate loss from batch

            # optimize the parameters
            loss.backward()
            optimizer.step()
            
            # calculate f1 score from the batch after update parameters
            f1 = round(f1_score(y_true, torch.argmax(y_pred, dim=1), average='macro'), 4)
            
            # get the number of correct predictions
            num_correct = (y_true == torch.argmax(y_pred, dim=1)).sum().item()
            acc = round(num_correct*100/n, 4) # calculate accuracy of the batch

            disp_loss = round(loss.item(), 4)
            elapsed_time = round(time.time()-start_time, 2)
            running_samples = n*(i+1)
            
            step_loss_train.append(disp_loss)
            step_acc_train.append(acc)
            step_f1_train.append(f1)
            
            # display training loss, accuracy, and f1 score of each step
            print('({}s) [{}/{}] loss: {}, acc: {}%, f1: {}'.format(elapsed_time, running_samples, N, disp_loss, acc, f1))
            num_step += 1
        # calculate validation stat after one epoch finished
        val_loss, val_acc, val_f1 = get_val_stat(model, val_loader, loss_fn)
        val_f1 = round(val_f1, 4)
        
        epoch_loss_val.append(val_loss)
        epoch_acc_val.append(val_acc)
        epoch_f1_val.append(val_f1)
        
        # display validating loss, accuracy, and f1 score of each step
        print('>'*5, 'EPOCH {}: validation loss: {}, validation accuracy: {}%, validation f1: {}'.format(epoch+1, val_loss, val_acc, val_f1), '<'*5)
    step_per_epoch = math.ceil(N/batch_size)
    epoch_plot = [step_per_epoch*i for i in range(epochs+1)]
    return model, ((step_loss_train, step_acc_train, step_f1_train), (epoch_loss_val, epoch_acc_val, epoch_f1_val, epoch_plot))

def evaluate(model, test_loader):
    '''
    evaluate the model from test data

    Inputs
        - model: the model used for evaluate
        - test_loader: dataloader for testing set
    Outputs
        - acc: accuracy from evaluation
        - cm: confusion matrix of the predictions
        - cm_norm: normalized confusion matrix
    '''
    X_test, y_test = iter(test_loader).next() # iterate over each testing samples
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        images, labels = X_test, y_test # get X, y from test_loader
        outputs = model(images) # posterior probability of each X_test
        predicted = torch.argmax(outputs.data, 1) # predicted class from maximum prob
    total = labels.size(0) # number of each testing batch
    correct = (predicted == labels).sum().item() # get number of correct predictions
    cm = confusion_matrix(labels.numpy(), predicted) # get unormalized confusion matrix
    cm_norm = normalize_confusion_matrix(cm) # calculate normalized confusion matrix
    return correct/total, cm, cm_norm