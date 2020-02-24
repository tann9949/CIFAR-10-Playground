import pickle
import numpy as np
import matplotlib.pyplot as plt

# set dict to map index to class name
idx2class = {
    0:'airplane',
    1:'automobile',
    2:'bird',
    3:'cat',
    4:'deer',
    5:'dog',
    6:'frog',
    7:'horse',
    8:'ship',
    9:'truck'
}

def unpickle(file):
    '''
    Load CIFAR-10 image from pickle file
    Input
        - file: path to CIFAR-10 pickle file
    Output
        - dict: a dictionary of pickled data
    '''
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_cifar(cifar_root):
    '''
    Load cifar data
    Input
        - cifar_root: a path to parent of `cifar-10-patches-py`
    Output
        - X: an array of CIFAR-10 features
        - y: an array of label corresponded to X
    '''
    X, y = list(), list() # initializa empty lists
    # iterate over each directories of CIFAR-10 batches
    for i in range(1,6):
        d = unpickle(cifar_root+'/cifar-10-batches-py/data_batch_{}'.format(i))
        [X.append(x) for x in d[b'data']] # append each image to X
        [y.append(gt) for gt in d[b'labels']] # append each label to y
    return np.array(X), np.array(y) # return X, y
