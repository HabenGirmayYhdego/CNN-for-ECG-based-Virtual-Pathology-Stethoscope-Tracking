
#==============================================================================
# SP_Individualized_version -Development Version
#
# Author: Kidane Nahom
#
#==============================================================================

from __future__ import division
from os.path import basename
import glob
import re
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import Queue
import logging
from pprint import pprint
from DataCollector import SensorApp
from Signalpreprocessor import ECGPreprocessor
import FeatureExtractor
from Detection_main


import logging
logging.basicConfig(level=logging.DEBUG)
# logging.basicConfig(level=logging.DEBUG,
#                    format='[%(levelname)s] (%(threadName)-10s) %(message)s',)
logger = logging.getLogger(__name__)


def Extrac_Features(wavefile):

    #=========================================================================
    # Training
    #=========================================================================
    # Read File:

    #filename = r'C:\Users\nkida001\Desktop\SP_Individualized_version\Datasets\ECG\SP3ST0 (3).wav'

    process = ECGPreprocessor()
    process.wavread(wavefile, test_mode=True)
    print process.sample_rate
  
    #-------------------------------------------------------------------------
    # Preprocessing :
    #-------------------------------------------------------------------------
    # remove powerline noise
    process.down_sample()
    print process.sample_rate
    FIR_para = {'cutoff': 40, 'tranwidth': 10, 'Rp': 45}
    process.design_FIR(FIR_para)
    # process.plot_taps()
    # process.plot_response()
    process.apply_FIRfilter(process.downsampled_data)
    # process.spectral_density(process.filtered_data)

    # Remove baseline wandering

    # FIR_para = {'cutoff': 0.5, 'tranwidth': 5, 'Rp': 60}
    # process.design_FIR(FIR_para)
    # process.plot_taps()
    # process.apply_FIRfilter(process.filtered_data, powerline=False)
    # process.plot_response()

    # process.Visualize_signal()

    #-------------------------------------------------------------------------
    #  Feature Extraction
    #-------------------------------------------------------------------------


    ecg = FeatureExtracror.Ecg(process.filtered_data, {'samplingrate': process.sample_rate})
    ecg.qrsDetect()

    #ecg.visualize_qrs_detection()

    ecg.visualize_qrs_detection2()

    #s, m = ecg.segmentation()

    features, Morph_Features = ecg. Features_Extraction()

    #feature_names = ['RQon', 'RQ', 'RS',  'RSoff', 'RT', 'RTon', 'RToff',
    #                'RQon_amp', 'RQ_amp', 'RS_amp', 'RSoff_amp',
    #                'RT_amp', 'RTon_amp', 'RToff_amp']

    feature_names = ['RQ', 'RS','RQ_amp', 'RS_amp']

    X_test = np.array([features[key] for key in feature_names])
    X_test = X_test.T
   
    return X_test


def Train_classifer(train_file, target_file,Morph_file):
    #-------------------------------------------------------------------------
    #  Training Classifier
    #-------------------------------------------------------------------------

    logger.info('Reading CSV file ...')

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
                          
   

    qrs_features = ['RQon','RQ', 'RS','RQ_amp', 'RS_amp','RSoff']

    X.columns = feature_names
    X_morph.columns= morph_features_names
    
    
    # Test modes

    QRS = 1
    Amp = 0
    allfeatures = 0
    morph = 0
    
    
    if QRS:
        selected = qrs_features
    elif Amp:
        selected = amp_features
    elif allfeatures:
        selected = feature_names
    
    elif morph:
      selected = morph_features_names
      X = X_morph
            
    else  :
        feature_all = feature_names + morph_features_names
        selected = feature_all
        X = X.join(X_morph)

    
   
    traindata = X.ix[:,selected] 
    logger.debug('training data file sample \n %s:...',  traindata.head())
 

    foo = Classifier(traindata,np.ravel(y.values))
    print foo   
    foo.start()
    foo.feature_names = selected
    #foo.scatter_Plot(traindata,y,selected)
    foo.dis_Plot(foo.cross_scores)
    foo.feature_importance_Plot(foo.Modelclf, foo.feature_names)
    foo.predict(foo.X_test,foo.y_test)
    foo.confusion_matrix_Plot(foo.y_test)
    foo.plot_Pred(foo.y_test)
    plt.show()

    return foo



def RealTime(clf, True_label):
    #=========================================================================
    # Real Time
    #=========================================================================
    #filename = r'C:\Users\nkida001\Desktop\SP_Individualized_version\Datasetsb\RB\Tricuspid_RB02.csv'
    readECG = SensorApp()
    process = ECGPreprocessor()      
    process.raw_data = readECG.sensor_data
   
    #process.sample_rate = 110
    filename = 'None'
    process.wavread(filename, test_mode=False)

    #=========================================================================
    # Save to file
    #=========================================================================
    write_filename = r'D:\Real-Time anlysis\Datasets\135\SP1SRB_M135_01.csv'
    np.savetxt(write_filename, process.raw_data, delimiter=",")


    #-------------------------------------------------------------------------
    # Preprocessing :
    #-------------------------------------------------------------------------
    # remove powerline noise
    process.down_sample()  
    FIR_para = {'cutoff': 40, 'tranwidth': 10, 'Rp': 45}
    process.design_FIR(FIR_para)
    process.apply_FIRfilter(process.downsampled_data)
    process.Visualize_signal()

    #-------------------------------------------------------------------------
    #  Feature Extraction
    #-------------------------------------------------------------------------

    SP_info = { 'name' : 'foo', 'age'  :27 , 'sex'  : 'Male' , 'sampling_rate' :1000 }
    FE = FeatureExtractor.FeatureExtractor(process.filtered_data,SP_info)

    FE.qrsDetect()

    #FE.visualize_QRS_detection()

    FE.visualize_fiducialPoints()
    #FE.beat_segmentor()
    plt.show()

    features, Morph_Features = FE. features_extracted()

  

    #ecg.visualize_qrs_detection()

    #ecg.visualize_qrs_detection2()

    #features, Morph_Features = ecg. Features_Extraction()

        
    feature_names = ['RQon' , 'RQ' ,  'RS' ,'RSoff', 'RTon', 'RT','RToff', 
                          'RQon_amp','RQ_amp' ,'RS_amp' ,  'RSoff_amp' ,
                          'RT_amp' , 'RTon_amp' , 'RToff_amp' ]

    feature_names = ['RQon','RQ', 'RS','RQ_amp', 'RS_amp','RSoff']

    X_test = np.array([features[key] for key in feature_names])
    X_test = X_test.T
   
    #-------------------------------------------------------------------------
    # Make Predication ---------------------------------------------------------------------------

    # make labels:

    # X_test = np.array([features[key]
    #                       for key in ['RQ', 'RS', 'RQ_amp', 'RS_amp']])
    # X_test = X_test.T
    
    print X_test
    
    # TODO : Make this dynamic ( user import)
    y_test = np.zeros(len(X_test))    
    #y_test.fill(clf.label.transform (True_label))
    y_test.fill(1)
    clf.predict(X_test, y_test)
    print "y_test :", y_test
    print "True label :", clf.label.inverse_transform (1) 
    print "prediced :" , clf.y_pred
    print "Predicted Area :", clf.label.inverse_transform (clf.y_pred)
    #clf.confusion_matrix_Plot(y_test)
    clf.plot_Pred(y_test)
    plt.show()

if __name__ == "__main__":

    # Training Feature extration
    #=========================================================================
    

    # path = r'C:\Users\nkida001\Desktop\SP_Individualized_version\Datasets\RB'
    # allwav_Files = glob.glob(path + "/*.csv")
    # last = len(allwav_Files)
    # temp = []
    # target = []
    # n_files = 0
    # count = 0
    # for filename in allwav_Files:
    #     print filename
    #     Area = re.search("^[^_]*", basename(filename)).group(0)
    #     pprint(Area)
    #     X_test = Extrac_Features(filename)
    #     temp.append(X_test)
    #     [target.append(Area) for x in X_test]
    
    # # feature_names = ['RQon', 'RQ', 'RS',  'RSoff', 'RT', 'RTon', 'RToff',
    # #                  'RQon_amp', 'RQ_amp', 'RS_amp', 'RSoff_amp',
    # #                  'RT_amp', 'RTon_amp', 'RToff_amp']

    # Features = np.row_stack(temp)
    # target = np.row_stack(target)
    # extractedfeatures = r'C:\Users\nkida001\Desktop\SP_Individualized_version\Datasets\RB\RB_features.csv'
    # np.savetxt(extractedfeatures, Features, delimiter=",")

    # target_file = r'C:\Users\nkida001\Desktop\SP_Individualized_version\Datasets\RB\RB_target.csv'
    # np.savetxt(target_file, target, delimiter=",", fmt="%s")

    # feature_names_file = r'C:\Users\nkida001\Desktop\SP_Individualized_version\Datasets\WB\RB_Colnames.csv'
    # np.savetxt(feature_names_file, feature_names, delimiter=",", fmt="%s")

# Training Classifier
#==============================================================================
    train_file = r'..\Datasets\all_RB\features_0620.csv'
    target_file =   r'..\Datasets\all_RB\target_0620.csv'
    Morph_file =  r'..\Datasets\all_RB\morph_0620.csv'
    logger.info('Reading CSV file ...')  


    clf = Train_classifer(train_file, target_file,Morph_file )


#==============================================================================

    #Real Time
    RealTime(clf,'Aortic')
