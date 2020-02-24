# import necessary libraries
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
from seaborn import heatmap
import matplotlib.pyplot as plt

# define mapping dictionary from class index to class name
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

def normalize_confusion_matrix(cm):
    '''
    return a copy of normalized confusion matrix
    Input
        - cm: input confusion matrix to normalize
    Output
        - norm_cm: a normalized confusion matrix
    '''
    return cm / cm.astype(np.float).sum(axis=1)

def plot_cm(cm, cm_norm, fig_path):
    '''
    Save a confusion matrix and norrmalized confusion matrix to the specific path
    Inputs
        - cm: unnormalized confusion matrix
        - cm_norm: normalized confusion matrix
        - fig_path: path to saved figure
    '''
    # convert confusion matrix (np.array) into pandas DataFrame format (for an ease of setting x,y ticks)
    df_cm = pd.DataFrame(cm, index = [idx2class[i] for i in range(10)], columns = [idx2class[i] for i in range(10)])
    df_cm_norm = pd.DataFrame(cm_norm, index = [idx2class[i] for i in range(10)], columns = [idx2class[i] for i in range(10)])
    plt.figure(figsize=(25,10)) # set figure size
    
    # plot confusion matrix
    plt.subplot(121)
    ax1 = heatmap(df_cm, annot=True)
    plt.title('Confusion Matrix')
    
    # plot normalized confusion matrix
    plt.subplot(122)
    ax2 = heatmap(df_cm_norm, annot=True)
    plt.title('Normalized Confusion Matrix')
    plt.savefig(fig_path+'confusion_matrix.png')

def plot_history(history, fig_path):
    '''
    Save figures of the training and validation step 
    from history obtained from training

    Inputs
        history: training/validating log obtained from `train` function
        fig_path: path to saved figure
    '''
    S_plot, E_plot = history
    
    plt.figure(figsize=(30,5))
    plt.subplot(131)
    plt.title('Model Loss')
    plt.plot(S_plot[0], label='train_loss')
    plt.plot(E_plot[3], E_plot[0], linewidth=3, label='val_loss')
    plt.legend()
    plt.subplot(132)
    plt.title('Model Accuracy')
    plt.plot(S_plot[1], label='train_acc')
    plt.plot(E_plot[3], E_plot[1], linewidth=3, label='val_acc')
    plt.legend()
    plt.subplot(133)
    plt.title('Model F1 Score (macro)')
    plt.plot(S_plot[2], label='train_f1')
    plt.plot(E_plot[3], E_plot[2], linewidth=3, label='val_f1')
    plt.legend()
    plt.savefig(fig_path+'results.png')