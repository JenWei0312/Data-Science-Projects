# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Bone and Joint
# This notebook implements the problem as a binary classification problem that detects the cases of Continuing care. The leaking features were pointed out by the doctors

import pickle
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import make_scorer
import hyperopt
from hyperopt import hp, tpe
from hyperopt.fmin import fmin
import lightgbm as lgbm
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_val_predict,ShuffleSplit
from imblearn.pipeline import Pipeline
import os
import numpy as np
import matplotlib.pyplot as plt
NPSEED = 1337
np.random.seed(NPSEED)
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

data = pickle.load(open('data/encoded_data.p','rb')) # Refer to data/process_data.py for implementation details
# display(data.head())
# print(f"Data shape {data.shape}")
# data['episodeDispositionCode'].value_counts().plot(kind='bar')
# plt.show()
# class_names =  ['Not continuing', 'Continuing Care']
features = list(data.keys())
features.remove('episodeDispositionCode')

print('Features used:')
for feature in features:
    print(feature, end = ', ')

X = data.drop('episodeDispositionCode', axis=1)
y = data['episodeDispositionCode']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify = y)
X_train_final = X_train.copy()
y_train_final = y_train.copy()
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.20, random_state=42, stratify = y_train)
feat_names = list(X_train.keys())

smt = ADASYN(random_state=0)
X_train, y_train = smt.fit_resample(X_train, y_train)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

clf = lgbm.LGBMClassifier().fit(X_train,y_train) #altalab search used to find best pipline
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:,1]
y_val_pred = clf.predict(X_valid)
y_val_probs = clf.predict_proba(X_valid)[:,1]

# +
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def get_metrics(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    class_names =  ['Not continuing', 'Continuing Care']    
    df_cm = pd.DataFrame(cm, columns=class_names, index = class_names)
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    sns.set(font_scale=1.4)#for label size
    sns.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16},  fmt='d')
    plt.show()
    metrics = np.array( precision_recall_fscore_support(y_test, y_pred) )
    df_cm = pd.DataFrame(metrics, index=['Precision', 'Recall', 'F','Support'], columns = class_names)
    df_cm.index.name = 'Class'
    df_cm.columns.name = 'Metrics'
    display(df_cm)


# -

plt.title('Test Metrics with 0.5 threshold')
get_metrics(y_test, y_pred)

import seaborn as sns
mod_y_prob = 0.5
pos_probas = y_val_probs[(y_valid == y_val_pred) & (y_valid == 1)]
neg_probas = y_val_probs[(y_valid != y_val_pred)& (y_valid == 1)]
fig, ax = plt.subplots(1, 1)
sns.distplot(pos_probas, ax=ax,hist = True, kde = False, label='Correct predictions', color = 'g', norm_hist=False)
sns.distplot(neg_probas, ax=ax,hist = True, kde = False,  label='Incorrect predictions', color = 'r', norm_hist=False)
ax.set_xlim((0, 1))
plt.ylabel('Count per class')
plt.xlabel('class_probability')
plt.title("Distribution of classifer confidence for `Continuing care`.")
plt.legend()
plt.show()
mod_y_prob = 0.5
pos_probas = y_val_probs[(y_valid == y_val_pred) & (y_valid == 0)]
neg_probas = y_val_probs[(y_valid != y_val_pred)& (y_valid == 0)]
fig, ax = plt.subplots(1, 1)
sns.distplot(pos_probas, ax=ax,hist = True, kde = False,  label='Correct predictions', color = 'g', norm_hist=False)
sns.distplot(neg_probas, ax=ax,hist = True, kde = False,  label='Incorrect predictions', color = 'r', norm_hist=False)
ax.set_xlim((0, 1))
plt.ylabel('Count per class')
plt.xlabel('class_probability')
plt.title("Distribution of classifer confidence for `Not Continuing care`.")
plt.legend()
plt.show()

# We can see a good seperation of the two classes at the ouput of the classifer. Continuing care cases are largely concentrated at probability ouputs closer to 1 and not continuing care close to 0. 

pos_probas = y_val_probs[(y_valid == y_val_pred)]
neg_probas = y_val_probs[(y_valid != y_val_pred)]
fig, ax = plt.subplots(1, 1)
sns.distplot(pos_probas, ax=ax,hist = True, kde = False, label='Correct predictions', color = 'g', norm_hist=False)
sns.distplot(neg_probas, ax=ax,hist = True, kde = False,  label='Incorrect predictions', color = 'r', norm_hist=False)
ax.set_xlim((0, 1))
plt.ylabel('Count per class')
plt.xlabel('class_probability')
plt.title("Distribution of classifer confidence.")
plt.legend()
plt.show()

# Distribution plots of classifer probability for correct and incorrect decisions show that larger number of errors are seen when the classifer confidence is in [0.4,0.6] range.  

# # Effect of removing uncertain predictions
# To test if removing uncertain predictions (from above) can help improve the performance of the classifier, we select two thresholds (`th1` and `th2`) that sweep across the range of probability values of the classifer and pick the ones that maximizes the F score. Anything with a probability between the two thresholds will be considered as uncertain. 

# +
##                <----------Classifer output (probability)--------->
                 
##   |--------------------------------*--------------------*---------------------------|

##   0   'Not continuing care'       th1    'Uncertain'   th2    'Continuing care'     1
# -

y_pred_2 = 2*np.ones(y_valid.shape)
vmax = -999999
pmax = 0
for th1 in np.linspace(0,1,10):
    for th2 in np.linspace(0,1,10):
        y_pred_2 = 2*np.ones(y_valid.shape)
        y_pred_2[y_val_probs <= th1] = 0
        y_pred_2[y_val_probs >= th2] = 1
        if th1>th2:
            continue
        cm = confusion_matrix(y_valid, y_pred_2)
        tp = cm[0][0]
        fp = cm[0][1]
        fn = cm[1][0]
        tn = cm[1][1]
        if cm.shape[1] == 3:
            class_names =  [ 'Not continuing care', 'Continuing Care', 'unknown']    
        else:
            class_names =  [ 'Continuing Care'] 
            
        if cm.shape[1] == 3:
            metrics = np.array( precision_recall_fscore_support(y_valid, y_pred_2) )
        else:
            metrics = np.array( precision_recall_fscore_support(y_valid, y_pred_2, average='binary') )
            
        
        df_cm = pd.DataFrame(metrics, index=['Precision', 'Recall', 'F','Support'], columns = class_names)
        
        df_cm.index.name = 'Class'
        df_cm.columns.name = 'Metrics'
        var_to_be_maximized =   np.mean(metrics[2])
        print(f"Not continuing care <{np.around(th1,2)}, uncertain [{np.around(th1,2)}, {np.around(th2,2)}]  and continuing care >{np.around(th2,2)}: F score = {np.around(var_to_be_maximized,2)}")
        if vmax < np.mean(var_to_be_maximized):
            vmax = np.mean(var_to_be_maximized)
            pmax = [th1, th2]
print(f'Best cutoffs F = {vmax} for Not continuing care <{np.around(pmax[0],2)}, uncertain [{np.around(pmax[0],2)},{np.around(pmax[1],2)}]  and continuing care >{np.around(pmax[1],2)}')

# We see that including any prediction in undertain class reduces the performance of the algorithm.

plt.title(f'Test Metrics with {np.around(pmax[0],2)} threshold')
get_metrics(y_test, y_prob>pmax[0])

# We see that with further tuning performance of the classifier can be improved. We can now train on data including the validation set to get maximum performance on the test.

# ## Explaining the decision
# We use a method called SHAP (SHapley Additive exPlanations). 
# This plot is made of all the points in the validation data. It demonstrates the following information:
# - Feature importance: Variables are ranked in descending order.
# - Impact: The horizontal location shows whether the effect of that value is associated with a higher or lower prediction.
# - Original value: Color shows whether that variable is high (in red) or low (in blue) for that observation.
# - Correlation: The “high” comes from the red color, and the “positive” impact is shown on the X-axis. 
#
# The explanations of why the model performed the way it did on the test set is as follows.

import shap
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(pd.DataFrame(X_test, columns = feat_names ))
shap.summary_plot(shap_values[1], pd.DataFrame(X_test, columns = feat_names ))

# - From the SHAP plot above, we can see that the most siginficant effect comes from`raDementia`. Larger values of the feature pushes the model to categorize the patient more to Continuing care. 
# - Similarly `admitAge` pushes the model to categorize the patient more to Continuing care. 
# - `institutionTypeIdFrom` plays a major role as well. We see that if the patient is coming from `institutionTypeIdFrom` 3 or 5, the patient is likely to be predicted as needing contiuing care. 
# - If the patient is coming from `institutionTypeIdFrom` 2,1,4, the patient is likely to be predicted as not needing contiuing care. 
# - There is a weak effect of `episodeHoursToAdmit`. This is not clear but large values seem to push the model to recommend continuing care. 
# - A similar effect is, if `institutionZoneId` is 5 (North), the model is recommending continuing care.  
#

# ## Test of reliability
# To test if the classifer can maintain performance we create 10 random splits of the data in the training-test set and then repeat the process of evaluating the algorithm.

clf_not_tuned =  lgbm.LGBMClassifier() 
pipeline = Pipeline([('oversample', smt), ('scale', scaler), ('classifer', clf_not_tuned)])
metrics = []
means = []
standard_devs = []
for metric in [ 'recall','precision','f1']:
    kfold = ShuffleSplit(n_splits=10, test_size=0.20, random_state=NPSEED)
    all_fold_results = cross_val_score(pipeline, X, y, cv=kfold, scoring=f'{metric}_macro')
    m = all_fold_results.mean()*100.0
    s = all_fold_results.std()*100.0
    print(f"{metric}: %.2f%% (%.2f%%)" % ( m, s))
    metrics.append(metric)
    means.append(m)
    standard_devs.append(s)
plt.errorbar(metrics, means, standard_devs, linestyle='None', marker='X')
plt.show()
