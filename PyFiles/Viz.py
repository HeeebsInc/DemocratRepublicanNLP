import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, f1_score, auc, confusion_matrix 
import numpy as np
import pandas as pd
import itertools
import seaborn as sns
from sklearn.preprocessing import label_binarize
from itertools import cycle

def plot_model_cm(test_cm, train_cm, classes,
                          theme, model_type, cmap=plt.cm.Blues, path = None, normalize=False):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.style.use(theme)

    if normalize:
        test_cm = test_cm.astype('float') / test_cm.sum(axis=1)[:, np.newaxis]
        train_cm = train_cm.astype('float') / train_cm.sum(axis=1)[:, np.newaxis]


    fig, ax = plt.subplots(1,2, figsize = (8,8))
    
    #Test Set
    ax[0].imshow(test_cm, interpolation='nearest', cmap=cmap)
    ax[0].set_title('CM for Test')
    tick_marks = np.arange(len(classes))
    ax[0].set_xticks(tick_marks)
    ax[0].set_xticklabels(classes)
    ax[0].set_yticks(tick_marks)
    ax[0].set_yticklabels(classes)

    fmt = '.2f' if normalize else 'd'
    thresh = test_cm.max() / 2.
    for i, j in itertools.product(range(test_cm.shape[0]), range(test_cm.shape[1])):
        ax[0].text(j, i, format(test_cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if test_cm[i, j] > thresh else "black")

    ax[0].set_ylabel('True label')
    ax[0].set_xlabel('Predicted label')
    if model_type.upper() == 'MASK': 
        ax[0].set_ylim(1.5, -.5)
    if model_type.upper() == 'EMOTION': 
        ax[0].set_ylim(2.5, -.5)

    
    
    #Train Set
    ax[1].imshow(train_cm, interpolation='nearest', cmap=cmap)
    if model_type.upper() == 'MASK':
        ax[1].set_title('CM for Validation')
    if model_type.upper() == 'EMOTION':
        ax[1].set_title('CM for Train')
    tick_marks = np.arange(len(classes))
    ax[1].set_xticks(tick_marks)
    ax[1].set_xticklabels(classes)
    ax[1].set_yticks(tick_marks)
    ax[1].set_yticklabels(classes)

    fmt = '.2f' if normalize else 'd'
    thresh = train_cm.max() / 2.
    for i, j in itertools.product(range(train_cm.shape[0]), range(train_cm.shape[1])):
        ax[1].text(j, i, format(train_cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if train_cm[i, j] > thresh else "black")

    ax[1].set_ylabel('True label')
    ax[1].set_xlabel('Predicted label')
    if model_type.upper() == 'MASK': 
        ax[1].set_ylim(1.5, -.5)
    if model_type.upper() == 'EMOTION': 
        ax[1].set_ylim(2.5, -.5)
    
    plt.tight_layout()
    
    if path: 
        plt.savefig(path)
    plt.show() 