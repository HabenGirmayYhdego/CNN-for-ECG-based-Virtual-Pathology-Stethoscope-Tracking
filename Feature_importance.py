import numpy as np
import pandas as pd 
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import preprocessing
# Build a classification task using 3 informative features
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
    t = X.join(X_morph)
   
X = t.ix[:,selected] 


label = preprocessing.LabelEncoder()
label.fit (np.ravel(y.values))
y = label.transform(np.ravel(y.values)) 



# Build a forest and compute the feature importances
forest = ExtraTreesClassifier(n_estimators=250,
                              random_state=0)

forest.fit(X, y)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(34):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
import pylab as pl
print len(importances[indices])

fig = pl.figure()
pl.title("Feature importances")
pl.bar(range(34), importances[indices],
       color="r", yerr=std[indices], align="center")
print indices
fig.autofmt_xdate()
new = [feature_all[i] for i in indices]
#new = [feature_all[i+2] for i in indices]
pl.xticks(range(34), new, fontsize = 12)
pl.xlim([-1, 34])
pl.show()