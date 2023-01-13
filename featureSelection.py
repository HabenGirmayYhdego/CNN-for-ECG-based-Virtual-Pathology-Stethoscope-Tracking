
import pandas as pd
import numpy as np
import pylab as pl

from sklearn import datasets, svm
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn import preprocessing

###############################################################################
train_file = r'C:\Users\nkida001\Google Drive\Summer 2014\Datasets\all_RB\features_0620.csv'
target_file =   r'C:\Users\nkida001\Google Drive\Summer 2014\Datasets\all_RB\target_0620.csv'
Morph_file =  r'C:\Users\nkida001\Google Drive\Summer 2014\Datasets\all_RB\morph_0620.csv'


X_morph = pd.read_csv(Morph_file,header = None )
X = pd.read_csv(train_file,header = None )
y = pd.read_csv(target_file, header = None )


# morph_features_names = ['QRSM0','QRSM1','QRSM2','QRSM3', 
#                         'TM0','TM1','TM2','TM3','TM4','TM5','TM6','TM7','TM8','TM9',
#                         'QRSM0_amp','QRSM1_amp','QRSM2_amp','QRSM3_amp', 
#                         'TM0_amp','TM1_amp','TM2_amp','TM3_amp','TM4',
#                        'TM5_amp','TM6_amp','TM7_amp','TM8_amp','TM9_amp'] 
                        
morph_features_names = ['QRSM0','QRSM1','QRSM2','QRSM3', 
                        'TM0','TM1','TM2','TM3','TM4','TM5',
                        'QRSM0_amp','QRSM1_amp','QRSM2_amp','QRSM3_amp', 
                        'TM0_amp','TM1_amp','TM2_amp','TM3_amp','TM4_amp',
                        'TM5_amp',] 
    
feature_names = ['RQon' , 'RQ' ,  'RS' ,'RSoff', 'RTon', 'RT','RToff', 
                      'RQon_amp','RQ_amp' ,'RS_amp' ,  'RSoff_amp' ,
                      'RT_amp' , 'RTon_amp' , 'RToff_amp' ]

amp_features = ['RQon_amp','RQ_amp' ,'RS_amp' ,  'RSoff_amp' ,
                      'RT_amp' , 'RTon_amp' , 'RToff_amp' ]


qrs_features = ['RQ' ,  'RS','RQ_amp' ,'RS_amp']

X.columns = feature_names
X_morph.columns= morph_features_names


# Test modes

QRS = 0
Amp = False
allfeatures = False
morph = False


if QRS:
    selected = qrs_features
elif Amp:
    selected = amp_features
elif allfeatures:
    selected = feature_names

elif morph:
  selected = morph_features_names
  X = X_morph
        
else :
    feature_all = feature_names + morph_features_names
    selected = feature_all
    X = X.join(X_morph)
   
X = X.ix[:,selected] 
label = preprocessing.LabelEncoder()
label.fit (np.ravel(y.values))
y = label.transform(np.ravel(y.values)) 

###############################################################################
pl.figure(1)
pl.clf()

X_indices = np.arange(X.shape[-1])

###############################################################################
# Univariate feature selection with F-test for feature scoring
# We use the default selection function: the 10% most significant features
selector = SelectPercentile(f_classif, percentile=10)
selector.fit(X, y)
scores = -np.log10(selector.pvalues_)
scores /= scores.max()
pl.bar(X_indices - .45, scores, width=.2,
       label=r'Univariate score ($-Log(p_{value})$)', color='g')

###############################################################################
# Compare to the weights of an SVM
clf = svm.SVC(kernel='linear')
clf.fit(X, y)

svm_weights = (clf.coef_ ** 2).sum(axis=0)
svm_weights /= svm_weights.max()

pl.bar(X_indices - .25, svm_weights, width=.2, label='SVM weight', color='r')

clf_selected = svm.SVC(kernel='linear')
clf_selected.fit(selector.transform(X), y)

svm_weights_selected = (clf_selected.coef_ ** 2).sum(axis=0)
svm_weights_selected /= svm_weights_selected.max()

pl.bar(X_indices[selector.get_support()] - .05, svm_weights_selected, width=.2,
       label='SVM weights after selection', color='b')

pl.xticks(range(34), feature_all ,fontsize=8)
pl.title("Comparing feature selection")
pl.xlabel('Feature number')
pl.yticks(())
pl.axis('tight')
pl.legend(loc='upper right')
pl.show()