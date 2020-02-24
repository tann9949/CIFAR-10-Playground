import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score

from utils import idx2class

def score(y_true, y_pred):
    '''
    Calculate a score of the evaluated model
    Inputs:
        - y_true: a ground truth label of y
        - y_pred: a predicted class of each samples corresponded to `y_true`
    Outputs:
        - f1_score: per-class f1-scores of the predicted samples
        - accuracy: accuracy of the predicted samples
    '''
    return f1_score(y_true, y_pred, average=None), accuracy_score(y_true, y_pred)



def print_f1(f1):
    '''
    Display per-class f1-score in a formatted form
    Inputs:
        - f1: f1 scores obtained from `score` function`
    '''
    for i, f in enumerate(f1):
        print('\t{}: {}'.format(idx2class[i], f))

def kfold_SVM(X, y, seed, n_fold=10, kernel='linear', transform=None):
    '''
    Run k-fold SVM and display a log of each training and evaluation
    Inputs:
        - X: a dataset which will be split to training and validation set
        - y: a ground truth of each data corresponded to `X`
        - seed: random seed of the settings
        - n_fold: number of fold (default as 10)
        - kernel: type of kernel used for doing kfold_SVM (default as 'linear')
        - transform: optional transform features model (such as PCA, StandardScaler, etc.)
    Outputs
        - avg_f1: average f1-score of all folds
        - avg_acc: average accuracy of all fols
        - model: trained model
    '''
    # transform input features if transform is not None
    if transform:
        X = transform.transform(X)
    kf = KFold(n_splits=n_fold, shuffle=True, random_state=seed) # get train, val split index from sklearn's KFold
    avg_f1, avg_acc = list(), list()
    # iterate over each fold
    for i, index in enumerate(kf.split(X)):
        start_time = time.time()
        # split train, val set
        train_index, val_index = index
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        print('='*20,'Fold {}'.format(i+1), '='*20)
        
        model = SVC(kernel=kernel) # initialize SVM classifier model
        
        fitted = model.fit(X_train, y_train) # fit the model with training set
        
        pred_train = model.predict(X_train)
        pred_val = model.predict(X_val)
        
        # set scores from train and val set
        train_f1, train_acc = score(y_train, pred_train)
        val_f1, val_acc = score(y_val, pred_val)
        
        # accumulate results
        avg_f1.append(val_f1)
        avg_acc.append(val_acc*100)
        
        # display train, val results
        print('Finished in {} seconds'.format(round(time.time()-start_time, 2)))
        print('train_accuracy: {}%\tval_accuracy: {}%'.format(round(train_acc*100, 2), round(val_acc*100, 2)))
        print('train_f1:')
        print_f1(train_f1)
        print('val_f1:')
        print_f1(val_f1)
    print('*'*20)
    print('\nAverage F1-Score:')
    print_f1(np.mean(avg_f1, axis=0))
    print('\nAverage Accuracy: {}'.format(round(np.mean(avg_acc), 2)))
    return avg_f1, avg_acc, model

def eval_svm(X_test, y_test, model):
    '''
    evaluate SVM model using given test set
    Inputs
        - X_test: a testing samples
        - y_test: ground truth of test samples
        - model: model use for evaluation
    Outputs
        - f1: per-class f1-score of testing set
        - acc: accuracy of testing set
    '''
    f1, acc = score(y_test, model.predict(X_test))
    print('Test F1-Score:')
    print_f1(f1)
    print('Test Accuracy:')
    print('\t{}%'.format(round(acc*100,2)))
    return f1, acc

def plot_f1(f1, xticks, title, save_path):
    '''
    Save figures of the per-class f1-scores plot
    Input
        - f1: f1 scores to plot
        - xticks: xticks of the plot
        - title: title of the plot
        - save_path: path to saved figure
    '''
    plt.figure(figsize=(20,7))
    for i in range(f1.shape[1]):
        plt.plot(f1[:,i],'o--', label=idx2class[i])
    plt.legend()
    plt.xticks(np.arange(f1.shape[0]),xticks)
    plt.grid()
    plt.title(title)
    plt.savefig(save_path)
    
def plot_acc(acc, xticks, title, save_path):
    '''
    Save figures of the accuracy plot
    Input
        - acc: accuracy scores to plot
        - xticks: xticks of the plot
        - title: title of the plot
        - save_path: path to saved figure
    '''
    plt.figure(figsize=(20,5))
    plt.plot(acc, 'o--')
    plt.xticks(np.arange(acc.shape[0]), xticks)
    plt.title(title)
    plt.grid()
    plt.savefig(save_path)

