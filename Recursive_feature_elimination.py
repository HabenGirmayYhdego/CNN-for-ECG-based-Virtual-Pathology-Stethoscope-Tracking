from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification
from sklearn.metrics import zero_one_loss

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

QRS = False
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



# Create the RFE object and compute a cross-validated score.
svc = SVC(kernel="linear")
rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(y, 2),
              scoring='accuracy')
rfecv.fit(X, y)

print("Optimal number of features : %d" % rfecv.n_features_)

# Plot number of features VS. cross-validation scores
import pylab as pl
pl.figure()
pl.xlabel("Number of features selected")
pl.ylabel("Cross validation score (nb of misclassifications)")
pl.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
pl.show()