 # Subject-based evaluation
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
# create a list to store the results of each subject
results = []

# extract the unique subjects
unique_subjects = np.unique(subjects)

#iterate over the subjects
for subject in unique_subjects:
    # create a mask to select only the samples of the current subject
    mask = subjects == subject
    # extract the samples of the current subject
    X_subject = X[mask]
    y_subject = y[mask]
    # predict class labels for the subject samples
    y_pred_subject = model.predict(X_subject)
    # convert predicted class labels to binary
    y_pred_subject = (y_pred_subject > 0.5).astype(int)
    # Compute the confusion matrix
    cm = confusion_matrix(y_subject, y_pred_subject)
    # Compute the specificity and sensitivity
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    sensitivity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    # Predict class probabilities for the subject samples
    y_pred_proba_subject = model.predict(X_subject)
    # Compute the ROC curve
    fpr, tpr, thresholds = roc_curve(y_subject, y

# Sample-based evaluation
# Predict class labels for the test set
y_pred = model.predict(X_test)

# Convert predicted class labels to binary
y_pred = (y_pred > 0.5).astype(int)

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Compute the specificity and sensitivity
specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
sensitivity = cm[1, 1] / (cm[1, 0] + cm[1, 1])

# Predict class probabilities for the test set
y_pred_proba = model.predict(X_test)

# Compute the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:,1], pos_label=1)

# Compute the AUC
auc = roc_auc_score(y_test, y_pred_proba[:,1])

# Plot the ROC curve
plt.plot(fpr, tpr, label='ROC curve (AUC = {:.2f})'.format(auc))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='best')
plt.show()

print("Specificity: ", specificity)
print("Sensitivity: ", sensitivity)

