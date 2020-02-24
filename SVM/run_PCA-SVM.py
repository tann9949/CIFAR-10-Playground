# import neccessary libraries
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
import argparse

# import defined function
from utils import load_cifar
from model import score
from model import kfold_SVM, eval_svm, plot_acc, plot_f1

if __name__ == '__main__':

    # add parser options
    parser = argparse.ArgumentParser(description='Run task 1, 2 questions')
    parser.add_argument('--cifar-root', metavar='DIR', help='Path to parent folder of `cifar-10-batches-py` directory')
    parser.add_argument('--fig-path', metavar='DIR', default='./figures/', help='Path to saved figure')
    parser.add_argument('--num-data', type=int, default=5000, help='number of training data')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')

    args = parser.parse_args()
    cifar_path = args.cifar_root
    fig_path = args.fig_path
    num_data = args.num_data
    SEED = args.seed

    X, y = load_cifar(cifar_path) # load data
    X_svm, y_svm = X[:num_data], y[:num_data] # select only 5000 data
    X_train, X_test, y_train, y_test = train_test_split(X_svm, y_svm, test_size=0.2, random_state=SEED, stratify=y_svm) # split train/test set

    print('='*10,'TASK 1 : Implementing PCA on data','='*10)

    # PCA using sklearn's Pipeline
    # by applying standard scaler (shift data to zero mean, unit variance) before PCA
    pca_010 = Pipeline([('scaling', StandardScaler()), ('pca', PCA(n_components=int(0.1*X_train.shape[1])))]).fit(X_train) # 10% of components
    pca_030 = Pipeline([('scaling', StandardScaler()), ('pca', PCA(n_components=int(0.3*X_train.shape[1])))]).fit(X_train) # 30% of components
    pca_050 = Pipeline([('scaling', StandardScaler()), ('pca', PCA(n_components=int(0.5*X_train.shape[1])))]).fit(X_train) # 50% of components
    pca_070 = Pipeline([('scaling', StandardScaler()), ('pca', PCA(n_components=int(0.7*X_train.shape[1])))]).fit(X_train) # 70% of components
    scaler = StandardScaler().fit(X_train) # scaler for 100% components

    print('='*10,'TASK 2','='*10)
    print('='*5,'TASK 2 Q1 : Apply linear SVM and do 10-fold cross validation','='*5)
    
    print('\n>>>>>>>>Training SVM on 10% dimensions PCA...\n')
    f1_010_val, acc_010_val, SVC_010 = kfold_SVM(X_train, y_train, transform=pca_010, seed=SEED) # applying k-fold to 10% components
    print('\n>>>>>>>>Training SVM on 30% dimensions PCA...\n')
    f1_030_val, acc_030_val, SVC_030 = kfold_SVM(X_train, y_train, transform=pca_030, seed=SEED) # applying k-fold to 30% components
    print('\n>>>>>>>>Training SVM on 50% dimensions PCA...\n')
    f1_050_val, acc_050_val, SVC_050 = kfold_SVM(X_train, y_train, transform=pca_050, seed=SEED) # applying k-fold to 50% components
    print('\n>>>>>>>>Training SVM on 70% dimensions PCA...\n')
    f1_070_val, acc_070_val, SVC_070 = kfold_SVM(X_train, y_train, transform=pca_070, seed=SEED) # applying k-fold to 70% components
    print('\n>>>>>>>>Training SVM on 100% dimensions PCA...\n')
    f1_linear_val, acc_linear_val, SVC_linear = kfold_SVM(X_train, y_train, transform=scaler, seed=SEED) # applying k-fold to 100% components

    X_test_010 = pca_010.transform(X_test) # transform test data to 10% PCA components
    X_test_030 = pca_030.transform(X_test) # transform test data to 30% PCA components
    X_test_050 = pca_050.transform(X_test) # transform test data to 50% PCA components
    X_test_070 = pca_070.transform(X_test) # transform test data to 70% PCA components
    X_test = scaler.transform(X_test)

    # evaluate SVM models
    print('>'*10,'10% PCA dimensions')
    f1_010_test, acc_010_test = eval_svm(X_test_010, y_test, SVC_010) 
    print('>'*10,'30% PCA dimensions')
    f1_030_test, acc_030_test = eval_svm(X_test_030, y_test, SVC_030)
    print('>'*10,'50% PCA dimensions')
    f1_050_test, acc_050_test = eval_svm(X_test_050, y_test, SVC_050)
    print('>'*10,'70% PCA dimensions')
    f1_070_test, acc_070_test = eval_svm(X_test_070, y_test, SVC_070)
    print('>'*10,'100% PCA dimensions')
    f1_linear_test, acc_linear_test = eval_svm(X_test, y_test, SVC_linear)

    print('='*5,'TASK 2 Q2 : Plot per-class f1 and accuracy','='*5)

    # plot per-class f1-scores
    f1_PCA = np.array([f1_010_test, f1_030_test, f1_050_test, f1_070_test, f1_linear_test])
    xticks = ['307 component (10% dimensions)','921 components (30% dimensions)','1536 components (50% dimensions)','2150 components (70% dimensions)','3072 components (100% dimensions)']
    title = 'per-class F1 Score of different PCA components on linear SVM'
    plot_f1(f1_PCA, xticks, title, fig_path+'f1_pca.png')

    # plot accuracy
    acc_PCA = np.array([acc_010_test, acc_030_test, acc_050_test, acc_070_test, acc_linear_test])
    title = 'Accuracy of different PCA components on linear SVM'
    plot_acc(acc_PCA, xticks, title, fig_path+'acc_pca.png')

    print('='*5,'TASK 2 Q3 : Apply different kernel SVM','='*5)

    # run K-fold on different SVM kernels
    print('>>>>>>>>Training SVM on RBF Kernel\n')
    f1_rbf_val, acc_rbf_val, SVC_rbf = kfold_SVM(X_train, y_train, kernel='rbf', transform=scaler, seed=SEED)
    print('>>>>>>>>Training SVM on Polynomial Kernel\n')
    f1_poly_val, acc_poly_val, SVC_poly = kfold_SVM(X_train, y_train, kernel='poly', transform=scaler, seed=SEED)

    # evaluate different SVM kernels
    print('>'*10,'RBF Kernel')
    f1_rbf_test, acc_rbf_test = eval_svm(X_test, y_test, SVC_rbf)
    print('>'*10,'Polynomial Kernel')
    f1_poly_test, acc_poly_test = eval_svm(X_test, y_test, SVC_poly)

    print('='*5,'TASK 2 Q4 : Plot Results of Task2 Q3','='*5)

    # plot per-class f1 scores
    f1_kernels = np.array([f1_linear_test, f1_rbf_test, f1_poly_test])
    xticks = ['linear', 'rbf', 'poly']
    title = 'per-class F1 Score of different SVM kernels'
    plot_f1(f1_kernels, xticks, title, fig_path+'f1_kernels.png')

    # plot accuracy
    acc_kernels = np.array([acc_linear_test, acc_rbf_test, acc_poly_test])
    title = 'Accuracy of different SVM kernels'
    plot_acc(acc_kernels, xticks, title, fig_path+'acc_kernels.png')