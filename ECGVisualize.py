from itertools import cycle
import numpy as np
from scipy.stats import sem
from scipy.misc import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from scipy.stats.kde import gaussian_kde 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


class PredVisualize(object):
    """
    class offering utility funcations for visualization of classifer
    predictions.
    """

    def __init__(self):
        """       
        :param labels :  numpy arrray of signal class labels 

        """
        self.colors = ['black','blue','red','magenta','green','Brown',
                      'DarkBlue','Tomato','Violet', 'Tan','Salmon','Pink',
                    'SaddleBrown', 'SpringGreen', 'RosyBrown','Silver']
        self.markers = [ 'o', 'd', 'h', '<', '>', '^']
       

    def scatter_Plot(self,X,y,features, title):
        """
        2d scatter plot 
        X: x values 
        y: y values
        features:  lables
        :title: Prints title of plot 
        """

        classId = y.unique()

    
        for id ,color, marker in zip(classId, cycle(self.colors),
                                     cycle(self.markers)):
            plt.scatter(X[y == i][features[0]],
                        X[y == i][features[1]],
                        c = color , marker = marker,
                        label = id, alpha =0.5)            
            plt.xlabel(features[0])
            plt.ylabel(features[1])
            plt.legend(loc='best')
            plt.title('Data points : %s' % title)
               
            
    def scatter_3dPlot(self,X,y,features, plotmode,name):            
  
        class_ID = y.unique()
    
        #fig = plt.figure(figsize=(20,6))
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
          
        for i ,c , m in zip (class_ID, cycle(colors),cycle(markers)):
           ax.scatter(X[y== i][features[0]], X[y == i][features[1]],
                      X[y == i][features[2]],c = c , marker = m, 
                      label = i, alpha =0.5)
           plt.legend(loc='best')
           ax.set_xlabel(features[0])
           ax.set_ylabel(features[1])
           ax.set_zlabel(features[2])

           if plotmode == 'Trian' :
                plt.title('Training Data %s' )
           else :
                plt.title('Testing Data %s' %name)
        
    def plotPred(self,class_labels,class_ID,QRS_index,y_test_lable,y_pred,y_updated,files,
                 Test_Acc,updated_Acc):    
       
        #cm =  confusion_matrix(y_test_lable.values, y_pred, labels = class_labels)    
        height = 4
    
        fig =plt.figure()  
        ax = fig.add_subplot(111)
        p1a, = plt.plot(QRS_index, y_pred,'-oy',ms=20, lw=12, alpha =0.8,mfc='y')
        p1b, = plt.plot(QRS_index,    
                    y_test_lable.values,                
                    '-k',ms=10, lw=5, alpha =1,mfc='k')
        
        p1c, = plt.plot(QRS_index[y_pred!=y_test_lable.values],  
                       y_pred[y_pred!=y_test_lable.values]                
                       ,'--ro',ms=10, lw=0.001, alpha =1,mfc='red')
        if len(y_updated) != 0:
            p1d, = plt.plot(QRS_index, y_updated,'-ob',ms=10, lw=8,
                            alpha =0.7,mfc='b')
            plt.legend([p1a, p1b,p1c,p1d],["Predicted Label", "True label", 
                          "Missed label","Sequentially  Updated Label"]               
                             ,loc='lower right')
        plt.title('Classifier Predictions, Testing on :%s' %files)   
        #plt.title('Classifier Predictions, Testing on :%s' %files[80:])
        plt.axis([0, 15, -1.5, 5])    
        plt.yticks(class_labels, class_ID[:height])
        plt.xlabel('QRS_index')
        plt.ylabel('Predictions')
    
        textstr = '$Accuracy = %.2f$\n$Sequentially  Updated Accuracy=%.2f$\
                    '%(Test_Acc,updated_Acc)
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.06, 0.2, textstr, transform=ax.transAxes, fontsize=14,
                verticalalignment='top', bbox=props)
    
    
    def plotRunPred(self,class_labels,class_ID,True_signalLabel,predicted_signalLabel
                    , updated_signalLabel):   
        # Signal Classification
  
        height = 4
        predicted= accuracy_score(True_signalLabel,predicted_signalLabel)
        updated =accuracy_score(True_signalLabel,updated_signalLabel)
        
        index= np.arange(len(predicted_signalLabel))
    
        fig =plt.figure(figsize=(25, 7))
        ax = fig.add_subplot(111)
        p1a, = plt.plot(index, predicted_signalLabel,'-y',ms=22, lw=12,alpha =0.8,mfc='y')
        p1b, = plt.plot(index,True_signalLabel,'-k',ms=10, lw=5, alpha =1,mfc='k')
        p1c, = plt.plot(index[predicted_signalLabel!=True_signalLabel],  
                       predicted_signalLabel[predicted_signalLabel!=True_signalLabel]                
                       ,'--ro',ms=15, lw=0.001, alpha =1,mfc='red',)               
    
        p1d, = plt.plot(index, updated_signalLabel, '-ob',ms=10, lw=8, alpha =0.7,mfc='b')
    
    
        #plt.legend
        plt.title('Classification for each Run')
        plt.axis([-1, 83, -1.5, 5])
        plt.legend([p1a, p1b,p1c,p1d],["Predicted Label", "True label", 
                                       "Missed label","Sequentially  Updated label" ] 
                                       ,loc='lower right')
        plt.yticks(class_labels, class_ID[:height])
        plt.xlabel('Runs')
        plt.ylabel('Predictions')
    
        textstr1 = 'Mean_Signal_Accuracy :\n$predicted = %.2f$\n $Sequentially Updated=%.2f$ \
                    ' %(predicted,updated)
    
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)      
        ax.text(0.05, 0.2, textstr1, transform=ax.transAxes, fontsize=14,
                  verticalalignment='top', bbox=props) 


    def featureimportancePlot (self,clf,feature_names):
        feature_importance = clf.feature_importances_
        # make importances relative to max importance
        feature_importance = 100.0 * (feature_importance / feature_importance.max())
        sorted_idx = np.argsort(feature_importance)
        pos = np.arange(sorted_idx.shape[0]) + .5
        plt.figure()
        plt.subplot(1, 2, 2)
        plt.barh(pos, feature_importance[sorted_idx], align='center', alpha=0.4)
        plt.yticks(pos, feature_names[sorted_idx[0]:])
        plt.xlabel('Relative Importance')
        plt.title('Feature Importance')
        plt.show()


    def  disPlot (self,samples):
        """
            Plots distribuation of data points and fittes gaussian curve 
        """

        def mean_score(scores):
            """Print the empirical mean score and standard error of the mean."""
            return (": {0:.3f} (+/-{1:.3f})").format(
                np.mean(scores), sem(scores))
    
        plt.figure()
        plt.hist(samples, range=(0, 1), bins=30, alpha=0.2)
        x = np.linspace(0, 1, 1000)
        smoothed = gaussian_kde(samples).evaluate(x)
        plt.plot(x, smoothed, label="Smoothed distribution")
        top = np.max(smoothed)
        plt.vlines([np.mean(samples)], 0, top, color='r',
                    label="Mean cross_validation score = %r" %mean_score(samples))
    
        plt.vlines([np.median(samples)], 0, top, color='b', 
                    linestyles='dashed',label="Median test score")
        plt.legend(loc='best')
        plt.title("Cross Validated test distribution of weighted majority algorthim " ) 
    
    
    def  confusion_matrixPlot (self, class_labels,class_ID,True_signalLabel
                                , predicted_signalLabel, mode  ): 
    
       
        cm = confusion_matrix(True_signalLabel, predicted_signalLabel,
                              labels = class_labels)
    
        width = len(cm)
        height = len(cm[0])
    
        if mode !=1:
            fig = plt.figure()
            plt.clf()
            ax = fig.add_subplot(111)
            ax.set_aspect(1)
            res = ax.imshow(np.array(cm), cmap=plt.cm.Blues, 
                            interpolation='nearest')
        
            width = len(cm)
            height = len(cm[0])
        
            for x in xrange(width):
                for y in xrange(height):
                    ax.annotate(str(cm[x][y]), xy=(y, x), 
                                horizontalalignment='center',
                                verticalalignment='center',color = 'r')
        
            fig.colorbar(res)
            plt.title('Confusion matrix')
            plt.xticks(range(width), class_ID[:width])
            plt.yticks(range(height), class_ID[:height])
            plt.ylabel('True label')
            plt.xlabel('Predicted label')

     
        colors = ['orange', 'g', 'c', 'y']
        markers = [ 'o', '<', 'D',  's']
        errc='red'  
        fig = plt.figure()
        np.random.seed(1)
        for x ,c , m in zip ( xrange(height), cycle(colors),cycle(markers)):
            for y in xrange(width):
                if x == y:        
                    if x == 0:
                       p1a=plt.scatter(np.random.uniform(120,140,cm[x,y]),np.random.uniform(90,110,cm[x,y])
                                 ,marker = m,s=50, lw=2, alpha=0.7, c=c)
                           
                    if x == 1:
                       p1b=plt.scatter(np.random.uniform(200,220,cm[x,y]),np.random.uniform(150,170,cm[x,y])
                                    ,marker = m,s=50, lw=2, alpha=0.7, c=c)
                    if x == 2:
                       p1c= plt.scatter(np.random.uniform(160,180,cm[x,y]),np.random.uniform(90,110,cm[x,y])
                                       ,marker = m,s=50, lw=2, alpha=0.7, c=c)
                    elif x == 3:
                        p1d=plt.scatter(np.random.uniform(150,170,cm[x,y]),np.random.uniform(150,170,cm[x,y])
                                    ,marker = m,s=50, lw=2, alpha=0.7, c=c)
    
                else: 
                    if x == 0:
                     p1m =  plt.scatter(np.random.uniform(120,140,cm[y,x]),np.random.uniform(90,110,cm[y,x])
                                 ,marker = ('x' if mode == 0 else m),s=50, lw=2, alpha=0.7, c=errc)
                    if x == 1:
                       p1m= plt.scatter(np.random.uniform(200,220,cm[y,x]),np.random.uniform(150,170,cm[y,x])
                                   ,marker = ('x' if mode == 0 else m),s=50, lw=2, alpha=0.7, c=errc)
                    if x == 2:
                       p1m= plt.scatter(np.random.uniform(160,180,cm[y,x]),np.random.uniform(90,110,cm[y,x])
                                       ,marker = ('x' if mode == 0 else m),s=50, lw=2, alpha=0.7, c=errc)
                    elif x == 3:
                       p1m= plt.scatter(np.random.uniform(150,170,cm[y,x]),np.random.uniform(150,170,cm[y,x])
                                    ,marker = ('x' if mode == 0 else m),s=50, lw=2, alpha=0.7, c=errc)
        plt.legend([p1a, p1b,p1c,p1d,p1m],["Aortic", "Mitral", 
                              "Pulmonic","Tricuspid","Miss Classified"]               
                                 ,loc='best')
    
        datafile=r'C:\Users\nkida001\Google Drive\Scientific Visualization\auscultation.jpg'
        img = imread(datafile)
        plt.imshow(img, zorder=0,alpha =0.5)
        plt.title("Sequentially  Updated Classifications " )
        return (cm)
    
    def  reportPrint (class_ID,True_signalLabel, predicted_signalLabel):     
        report = classification_report(True_signalLabel, predicted_signalLabel,
                                     target_names=class_ID)
        print (report) 
        return(report)                             
                                 
    
    
    
