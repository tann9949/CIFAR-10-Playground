# import necessacy libraries
from data_loader import get_data
from model import VGG_lite
from train_utils import train, evaluate
from utils import plot_cm, plot_history

import torch
from torch import nn

import numpy as np
import argparse

if __name__ == '__main__':

    # add parser options
    parser = argparse.ArgumentParser(description='Run task 1, 2 questions')
    parser.add_argument('--cifar-root', metavar='DIR', default='../', help='Path to CIFAR-10')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--batch-size', type=int, default=256, help='number of batch size')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs')
    parser.add_argument('--augment', type=bool, default=True, help='set whether to augment the data')
    parser.add_argument('--fig-path', metavar='DIR', default='./figures/', help='Path to saved figure')
    parser.add_argument('--weight-path', metavar='DIR', default='./weights/final.pth')
    parser.add_argument('--val-split', type=float, default=0.2, help='validation split')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')

    args = parser.parse_args()
    cifar_dir = args.cifar_root
    fig_path = args.fig_path
    validation_split = args.val_split
    batch_size = args.batch_size
    epochs = args.epochs
    weight_path = args.weight_path
    weight_decay = args.weight_decay
    lr = args.lr

    SEED = args.seed # set random seed (default as 1234)

    # split train, val, test from `get_data` function
    train_loader, val_loader, test_loader = get_data(cifar_dir=cifar_dir, batch_size=batch_size, augment=True, validation_split=validation_split)

    # load model
    model = VGG_lite()
    # define loss
    loss = nn.CrossEntropyLoss()
    # train the model
    model, history = train(model, train_loader, val_loader, epochs, loss, batch_size, optimizer='adam', weight_decay=weight_decay, lr=lr)

    # save the model accordeing to `weight_path` from parser (default to './weights/final.pth')
    torch.save(model.state_dict(), weight_path)

    plot_history(history, fig_path) # save figures

    acc, cm, cm_norm = evaluate(model, test_loader) # evaluate trained model
    plot_cm(cm, cm_norm, fig_path) # save confusion matrix figures
    print('Test Accuracy: {}%'.format(round(acc*100, 4))) # print the model test accuracy