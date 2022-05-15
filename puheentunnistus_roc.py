#!/usr/bin/env python
# coding: utf-8

# In[112]:


import numpy as np
import sklearn.metrics
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsOneClassifier
from scipy import interp
from sklearn.metrics import roc_auc_score
import seaborn as sns

# Defining the location of the files
SCOREFILE = "C:/Users/Juuso Paakkunainen/OneDrive - University of Eastern Finland/kandi/20210909_rocs/scores_test_H"
TRIALSFILE = "C:/Users/Juuso Paakkunainen/OneDrive - University of Eastern Finland/kandi/20210909_rocs/trials_H"

# Importing the scores
scores = np.genfromtxt(SCOREFILE, dtype=str)
scores_keys = scores[:, 2].astype(np.float64)

# Importing the labels (nontarget/target)
labels = np.genfromtxt(TRIALSFILE, dtype=str)
labels_keys = labels[:, 2]

# Concatenateing the score and label arrays
scoreslabels = np.concatenate((scores, labels), axis=1)

# Getting the target scores from array
targets = scoreslabels[scoreslabels[:, 5] == "target"]
target_scores = targets[:, 2].astype(np.float64)

# Getting the nontarget scores from array
nontargets = scoreslabels[scoreslabels[:, 5] == "nontarget"]
nontarget_scores = nontargets[:, 2].astype(np.float64)

print(target_scores)
sns.distplot(target_scores, hist=False, kde=True, 
             bins=int(180/5), color = '#e1812caa',
             hist_kws= {'edgecolor':'black'},
             kde_kws = {'shade': True, 'linewidth': 1},
             label='Positive')

sns.distplot(nontarget_scores, hist=False, kde=True, 
             bins=int(180/5), color = '#3274a1aa',
             hist_kws= {'edgecolor':'black'},
             kde_kws = {'shade': True, 'linewidth': 1},
             label='Negative')

plt.legend(prop={'size': 10}, title = 'Label')
plt.xlabel('Score')
plt.ylabel('Density')


# In[113]:


def compute_det_curve(target_scores, nontarget_scores):

    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate((np.ones(target_scores.size), np.zeros(nontarget_scores.size)))

    # Sort labels based on scores
    indices = np.argsort(all_scores, kind='mergesort')
    labels = labels[indices]

    # Compute false rejection and false acceptance rates
    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - (np.arange(1, n_scores + 1) - tar_trial_sums)

    frr = np.concatenate((np.atleast_1d(0), tar_trial_sums / target_scores.size))  # false rejection rates
    far = np.concatenate((np.atleast_1d(1), nontarget_trial_sums / nontarget_scores.size))  # false acceptance rates
    thresholds = np.concatenate((np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices]))  # Thresholds are the sorted scores
   
    return frr, far, thresholds


# In[114]:


def compute_eer(target_scores, nontarget_scores):
    """ Returns equal error rate (EER) and the corresponding threshold. """
    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))
    
    plt.title('DET-käyrä')
    plt.xlabel("False Positive Rate")
    plt.ylabel("False Negative Rate")
    plt.plot(
         frr, 
         far, 
         lw=2,
         label="DET-käyrä (EER = %0.3f)" % eer,
        )
    plt.legend(loc="upper right")
    plt.show()

    #return eer, thresholds[min_index]

    print('   EER            = {:8.3f}  (Equal error rate (target vs. nontarget discrimination)'.format(eer))
    print('   THRESHOLD      = {:8.3f} '.format(thresholds[min_index]))
    
compute_eer(target_scores, nontarget_scores)


# In[116]:


def compute_roc_curve(scores_keys, labels_keys):

    fpr, tpr, thresholds = metrics.roc_curve(labels_keys, scores_keys, pos_label="target")

    auc = np.trapz(tpr,fpr)
    
    plt.title('ROC-käyrä')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.plot(fpr, 
             tpr, 
             lw=2,
             label="ROC-käyrä (AUC = %0.3f)" % auc,
            )
    plt.plot([0, 1], [0, 1], color="grey", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.legend(loc="lower right")
    plt.show()
    
    
    
    print("AUC: ", auc)
    
compute_roc_curve(scores_keys, labels_keys)
    

