#-----------------------------------------------------------------------------
# TODO : Add doc info  Pan-Tompkins QRS Detection.
# TODO: Change amplitude to from mV to V
# TODO : Add refence to Pan
# TODO: add discribtion on all methods  
#-----------------------------------------------------------------------------

from __future__ import division
from itertools import cycle 
import matplotlib.pyplot as plt
import scipy
from scipy import signal
import datetime
import pandas as pd
import numpy as np
import numba as nb 


from sklearn import preprocessing

import logging 
logging.basicConfig(level = logging.INFO)
logger = logging.getLogger('Feature_Extractor')


class QrsDetectionError(Exception):
    """ Raise error related to qrs detection"""
    def __init__(self, value):
        self.parameter = value
    
    def _str_(self):
        return repr(self.parameter)
    

class FeatureExtractor(object):

    def __init__(self,ecgdata,info): 
        """
        - Ecgdata  : The ECG signal as an array vector.
        """
        logger.info(' Starting...')
        self.info = info
        self._readinfo()
        self.data = scipy.array(ecgdata)
        
        # convert vector to colum array
        if len(self.data.shape)== 1:
            self.data = scipy.array ([self.data]).transpose()
        
        self.points, self.leads = self.data.shape
        
        if  len(self.data.shape)> 1 and self.leads > self.points:
            raise QrsDetectionError("Data has more columns than rows")
        

        self.plot_peaks = []
        self.plot_sig_thresh = [] 
        
        # Start:
        #self.qrsDetect()            

    def _readinfo(self):
        # TODO: Move read info to data collector
        """Read the info and fill missing values with defaults
        - info : A dictionary object holding subject info.
        - 'name' = Patient name e.g. SP01.
        - 'age'  = Age of Patient in years.
        - 'sex'  = 'Male' , 'female' or 'u' for unknown. 
        - 'sampling_rate' = sampling frequency (fs) in Hz.
        -  all are optional
        """
        logger.info(' Reading Info...')

        self.name = self.info.get('name', '')
        self.age = self.info.get('age', 0)
        self.sex = self.info.get('sex', 'u')
        self.ascultation_region = self.info.get('ascultation_region', '')
        try:
            self.sampling_rate = self.info.get('sampling_rate')
        except KeyError:
            self.sampling_rate = 1000
            self._warning("Info does not contain sampling rate, assuming 1000")     

        logger.debug("SP data info-> %s", self.info)            
       
    def _sample_to_time(self, sample):
        # TODO : Change this two min and seconds only 
        """convert from sample number to time
        in a format required for the annotation file.
        This is in the form (hh):mm:ss.sss"""
        time_ms = int(sample*1000 / self.sampling_rate)
        hr, min, sec, ms = time_ms//3600000 % 24, time_ms//60000 % 60, \
                           time_ms//1000 % 60, time_ms % 1000
        timeobj = datetime.time(hr, min, sec, ms*10000) # last val is microsecs
        timestring = timeobj.isoformat()
        # if there is no ms value, add it
        if not '.' in timestring:
            timestring += '.000000'
        return timestring[6:-3] # back to ms

    def _process_ecg(self):
        "process the raw ecg signal"
        logger.info("preprocessing----")
            
        #TODO: Include all preproccessing steps in here
        #TODO: Check all steps for QRS detection See if classification resluts change
        
        # Step 1: Normalize the ECG to value between

        #self.Norm1_ecg = self.raw_ecg/ np.max(abs(self.raw_ecg))
        self.Norm1_ecg = preprocessing.scale(self.raw_ecg) #scaling

      
        # Step 2: Bandpass between 5 and 15 to isolate the QRS wave
        self.filtered_ecg = self._bpfilter(self.Norm1_ecg)

        # Step 3: Differentiate the filtered signal                 
        self.diff_ecg  = scipy.diff(self.filtered_ecg)
        
        # Step 4: intergate using moving window
        self.Norm_ecg = self.diff_ecg/ np.max(abs(self.diff_ecg))
        self.abs_ecg = abs(self.Norm_ecg) 
        self.int_ecg = self._mw_integrate(self.abs_ecg )

    def _initializeBuffers(self, start = 0):
        #TODO: Test feature extractor buffer size for less than 3 seconds
        #TODO : Make window size dynamic
        """Initialize the buffers using values from the first 3 seconds"""   
        srate = self.sampling_rate
        self.signal_peak_buffer = [max(self.int_ecg[int(start + i * srate):int(start + i * srate + srate)])
                               for i in range(3)]
        self.noise_peak_buffer = [0] * 3
        self.rr_buffer = [int(srate)] * 3 #rr buffer initialized with one sec
        self._updateThreshold()       
             
    def qrsDetect(self, qrslead=0):
        """Detect QRS onsets using modified PT algorithm"""
        # If ecg is a vector, it will be used for qrs detection.
        # If it is a matrix, use qrslead (default 0)

        if len(self.data.shape) == 1:
         self.raw_ecg = self.data
        else:
         self.raw_ecg = self.data[:,qrslead]
         
        self._process_ecg() #creates diff_ecg and int_ecg
             
        # Construct buffers with last 3 values 
        self._initializeBuffers()

        #Start :
        peaks = self._peakdetect(self.int_ecg)
        self._checkPeaks(peaks)
        self.fiducial_points()
        #self.QRSpeaks -= int(w/2)
        # compensate for delay during integration
        #self.QRSpeaks -= 40 * (self.sampling_rate / 1000)
                
        #self.FP = { 'Q' : (),'R' : (),'S' : ()}          

    def _write_ann(self, annfile):
        # TODO  write as text or CSV 
        """Write annotation file in a format that is usable with wrann"""
        fi = open(annfile, 'w')
        for qrspeak in self.QRSpeaks:
            fi.write('%s %s %s %s %s %s\n' %(
                self._sample_to_time(qrspeak), qrspeak, 'N', 0, 0, 0))
        fi.close()           

    def _updateThreshold(self):
        #TODO:  check and modify from Arzeno et.al/Pan et.al
        # modify this per Subject 
        """Update thresholds from buffers"""
        noise = scipy.mean(self.noise_peak_buffer)
        sign = scipy.mean(self.signal_peak_buffer)
        self.threshold = noise + 0.6 * (sign - noise)
        #self.threshold = noise + 0.3125 * (signal - noise)
        self.meanrr = scipy.mean(self.rr_buffer)           
  
    def _bpfilter(self, ecg):
        """Bandpass filter the ECG from 5 to 15 Hz"""
        #TODo: See how lowering the band pass will affect the classifer
        #TODO: If bandpass is good do it with FIR?
        logger.debug("Bandpass filter to isolate the QRS complex")
        # TODO: Explore - different filter parameters 
        Nyq = self.sampling_rate / 2
        wn = [5/ Nyq, 30/ Nyq]
        b,a = signal.butter(2, wn, btype = 'bandpass') 

        logger.debug("QRS filter :%s" ,[5, 30] )      
        return signal.filtfilt(b,a,ecg)

    def _mw_integrate(self, ecg):
        """ Integrate the ECG signal over a defined time period"""
        # TODO : Check different window sizes eg 80 -150ms for window integrator   
        # window of 80 ms - better than using a wider window
        logger.debug("Moving window Integration")
        self.window_length = int(13* (self.sampling_rate / 256))
        logger.info("using integrator window size : %s", self.window_length)
        int_ecg = scipy.zeros_like(ecg)
        cs = ecg.cumsum()
        int_ecg[ self.window_length:] = (cs[ self.window_length:] - cs[:- self.window_length]
                                   ) /  self.window_length
        int_ecg[: self.window_length] = cs[: self.window_length] / scipy.arange(
                                                   1,  self.window_length + 1)
        return int_ecg
    
    def _peakdetect(self, ecg):
        """Detect all local maxima with no larger maxima within 200 ms"""
        #*----- R-peak detection ----*
        # TODO: Review this code and possibly reuse for Q and S detection    

        # list all local maxima
        all_peaks = [i for i in range(1,len(ecg)-1)
                     if ecg[i-1] < ecg[i] > ecg[i+1]]
        peak_amplitudes = [ecg[peak] for peak in all_peaks]
        final_peaks = []
        minRR = self.sampling_rate * 0.2

        # start with first peak
        peak_candidate_index = all_peaks[0]
        peak_candidate_amplitude = peak_amplitudes[0]
        # test successively against other peaks
        for peak_index, peak_amplitude in zip(all_peaks, peak_amplitudes):
            close_to_lastpeak = peak_index - peak_candidate_index <= minRR
            # if new peak is less than minimumRR away and is larger,
            # it becomes candidate
            if close_to_lastpeak and peak_amplitude > peak_candidate_amplitude:
                peak_candidate_index = peak_index
                peak_candidate_amplitude = peak_amplitude
            # if new peak is more than 200 ms away, candidate is added to
            # final peak and new peak becomes candidate
            elif not close_to_lastpeak:
                final_peaks.append(peak_candidate_index)
                peak_candidate_index = peak_index
                peak_candidate_amplitude = peak_amplitude
            else:
                pass
        return final_peaks

    def _acceptasQRS(self, peak, amplitude):
        # if we are in relook mode, a qrs detection stops that
        if self.RELOOK:
            self.RELOOK = False
        # add to peaks, update signal buffer
        self.QRSpeaks.append(peak)
        self.signal_peak_buffer.pop(0)
        self.signal_peak_buffer.append(amplitude)
        # update rr buffer
        if len(self.QRSpeaks) > 1:
            self.rr_buffer.pop(0)
            self.rr_buffer.append(self.QRSpeaks[-1] - self.QRSpeaks[-2])
        self._updateThreshold()
    
        self.plot_peaks.append(peak)
        self.plot_sig_thresh.append(self.threshold)

    def _acceptasNoise(self, peak, amplitude):
        self.noise_peak_buffer.pop(0)
        self.noise_peak_buffer.append(amplitude)
        self._updateThreshold()
    
        self.plot_peaks.append(peak)
        self.plot_sig_thresh.append(self.threshold)

    def _checkPeaks(self, peaks):
        """Go through the peaks one by one and classify as qrs or noise
        according to the changing thresholds"""

        #TODO: Review code 
        srate = self.sampling_rate
        ms10 = np.int(10 * 1000 / srate)
        amplitudes = [self.int_ecg[peak] for peak in peaks]
        self.QRSpeaks = [-360] #initial val which we will remove later
        self.RELOOK = False # are we on a 'relook' run?
    
        for index in range(len(peaks)):
            peak, amplitude = peaks[index], amplitudes[index]
            # booleans
            above_thresh = amplitude > self.threshold
            distant = (peak-self.QRSpeaks[-1])*(1000/srate) > 360
            classified_as_qrs = False
            distance = peak - self.QRSpeaks[-1] # distance from last peak
            
            # Need to go back with fresh thresh if no qrs for 8 secs
            if distance > srate * 3:
                # If this is a relook, abandon
                if self.RELOOK: 
                    self.RELOOK = False
                else:
                    # reinitialize buffers
                    self.RELOOK = True
                    index = peaks.index(self.QRSpeaks
                                        [-1])
                    self._initializeBuffers(peaks[index])
    
            # If distance more than 1.5 rr, lower threshold and recheck
            elif distance > 1.5 * self.meanrr:
                i = index - 1
                lastpeak = self.QRSpeaks[-1]
                last_maxder = np.amax(self.abs_ecg[lastpeak-ms10:lastpeak+ms10])
                while peaks[i] > lastpeak:
                    this_maxder = max(self.abs_ecg[ms10:peak+ms10])
                    above_halfthresh = amplitudes[i] > self.threshold*0.5
                    distance_inrange = (peaks[i] - lastpeak)*(1000/srate) > 360
                    slope_inrange = this_maxder > last_maxder * 0.6
                    if above_halfthresh and distance_inrange and slope_inrange:
                        self._acceptasQRS(peaks[i], amplitudes[i])
                        break
                    else:
                        i -= 1
            
            # Rule 1: > thresh and >360 ms from last det
            if above_thresh and distant:
                classified_as_qrs = True
    
            # Rule 2: > thresh, <360 ms from last det
            elif above_thresh and not distant:
                this_maxder = max(self.abs_ecg[peak-ms10:peak+ms10])
                lastpeak = self.QRSpeaks[-1]
                last_maxder = np.amax(self.abs_ecg[lastpeak-ms10:lastpeak+ms10])
                if this_maxder >= last_maxder * 0.85: #modified to 0.6
                    classified_as_qrs = True
    
            if classified_as_qrs: 
                self._acceptasQRS(peak, amplitude)
            else:
                self._acceptasNoise(peak, amplitude)
    
        self.QRSpeaks.pop(0) # remove that -360
        self.QRSpeaks = scipy.array(self.QRSpeaks)
        return

    def fiducial_points(self):

        # fiducial points:
        w =  self.window_length 
        self.FP = { 'Qon' : (),
                    'Q' : (),
                    'R' : (),
                    'S' : (), 
                    'Son': (),
                    'T' : (), 
                    'Ton' : (),
                    'Toff' : (),
                    'QRS_Morph': (),
                    'T_Morph': ()}
        
        N = len(self.raw_ecg)
        ten_seconds = 10 * self.sampling_rate
        
        if N > ten_seconds:
            segmentend = ten_seconds
        else:
            segmentend = N
        
        #---------------------------------------------------------------------
        # S peaks
        #---------------------------------------------------------------------


        #S peaks Candidates
            

        S_peaks = [peak for peak in self.QRSpeaks if peak < segmentend]
        self.FP['S'] =  S_peaks
        
        list_min = []
        index = 1
        for S in S_peaks:  
           while (self.Norm1_ecg[S] > self.Norm1_ecg[(S) +index])and (S) +index != S+ np.ceil(w*0.2):
                 index+=1
           list_min.append((S)+index)
     
        S_peaks = list_min 

        #---------------------------------------------------------------------
        # R peaks
        #---------------------------------------------------------------------


        self.QRSpeaks -= int(w/2)
        R_peaks = [peak for peak in self.QRSpeaks if peak < segmentend] 


        #---------------------------------------------------------------------
        # Q peaks
        #---------------------------------------------------------------------     
        #TODO : improve Q peak detection 
        self.QRSpeaks -= int(w/2) 
        Q_peaks = [peak for peak in self.QRSpeaks if peak < segmentend]
  

        # list_min = []
        # index = 1
        # for Q in Q_peaks:  
        #    while (self.Norm1_ecg[Q] > self.Norm1_ecg[(Q-10)-index]and (Q-10)-index != Q- np.ceil(w*0.6)):
        #         index+=1
        #    list_min.append((Q-10)-index)

        # Q_peaks = list_min      
        
        #---------------------------------------------------------------------
        # Q_on      
        #--------------------------------------------------------------------- 
        
        
        list_min = []
        onset = np.zeros(len(Q_peaks))
        index = 1
        for Q in Q_peaks:  
           while (self.int_ecg[Q-10] > self.int_ecg[(Q-10)-index]and (Q-10)-index != Q-np.ceil(w*0.6)):
                index+=1
           list_min.append((Q-10)-index)

        onset = list_min

        
        #---------------------------------------------------------------------
        # S_off    
        #--------------------------------------------------------------------- 
 
        list_min = []
        offset = np.zeros(len(S_peaks))
        index = 1
        for S in S_peaks:  
           while (self.int_ecg[S+12] > self.int_ecg[(S+12) +index])and (S+12) +index != S+ np.ceil(w*0.75):
                index+=1
           list_min.append((S+12)+index)

        offset = list_min


        #---------------------------------------------------------------------
        # T Peak  
        #---------------------------------------------------------------------
        #TODO: Check T wave detection for neagtive R peak case ?

        list_max = []
        T_off = np.zeros(len(S_peaks))
        T_peaks = np.zeros(len(S_peaks))
        ST_interval = int(0.32 *self.sampling_rate)
        for left,R,S  in zip( offset, R_peaks, S_peaks):
            search = self.Norm1_ecg[left+100 : left + ST_interval]
            if self.Norm1_ecg[R] > self.Norm1_ecg[S] :
                list_max.append (left + 100+np.argmax(search))
            else :
                list_max.append (left + 100+np.argmin(search))


        T_peaks =list_max

        #---------------------------------------------------------------------
        # T_on and T_off 
        #---------------------------------------------------------------------

        list_min2 = []
        list_min1 = []
        T_on = np.zeros(len(T_peaks))
        T_off = np.zeros(len(T_peaks))
        T_interval = np.int(0.16 *self.sampling_rate)
        half_interval = np.int(T_interval/2)
        for peak, R ,S in zip( T_peaks , R_peaks ,S_peaks): 
            search1 = self.Norm1_ecg[peak-half_interval : peak ]
            search2 = self.Norm1_ecg[peak : peak + half_interval]
            if self.Norm1_ecg[R] > self.Norm1_ecg[S] :
                list_min1.append (peak-half_interval+ np.argmin(search1))            
                list_min2.append (peak+ np.argmin(search2))
            else :   
                list_min1.append (peak-half_interval+ np.argmax(search1))            
                list_min2.append (peak+ np.argmax(search2))

        T_on = list_min1
        T_off = list_min2
        
        
        #---------------------------------------------------------------------
        # Morphology features (random points between the characteristics waves)  
        #---------------------------------------------------------------------
        # Between Qon and Soff  

        Mor_list=[] 
        for Q,R,S, in zip(onset,R_peaks,offset):  
            temp =[]
            off = int( (R - Q)*0.33)
            temp.append( np.arange(Q+off,R-off,off))
            off = int( (S- R)*0.33)
            temp.append( np.arange(R+off,S-off,off) )

            morph =np.concatenate(temp)
            Mor_list.append(np.int_(morph))

        # Between QRS and Toff 

        Mor_list2=[]     
        for S_of,T_of in zip(offset,T_off):  
            off = int( (T_of- S_of)*0.1)
            morph= np.linspace(S_of+off,T_of,6,endpoint = False)          
               
            Mor_list2.append(np.int_(morph))

        #---------------------------------------------------------------------
        # All fiducial_points (FP)
        #---------------------------------------------------------------------
        #QRS fiducial_pointsfiducial_points
        self.FP['S'] =  S_peaks
        self.FP['Q'] =  Q_peaks
        self.FP['R'] =  np.int_(R_peaks)      
        self.FP['Qon'] =  np.int_(onset)
        self.FP['Soff'] =  np.int_(offset)
        
        #T wave fiducial_points
        self.FP['T'] =  np.int_(T_peaks)
        self.FP['Ton'] =  np.int_(T_on)
        self.FP['Toff'] =  np.int_(T_off)

        # Morphology fiducial_points 
        self.FP['QRS_Morph'] =  np.int_(Mor_list)        
        self.FP['T_Morph'] =  np.int_(Mor_list2)  
        
        return  self.FP

    def features_extracted(self):
        """
        Save features to a dictionary. 
        The first and last beats are disregarded to safeguard for discontinuity.
        """
        N=len(self.FP['R'])
        

    
        self.Morph_Features = { 'RqrsMorph' : (),
                                'RTMorph' : (),
                                'RqrsMorph_amp' : (),
                                'RTMorph_amp' : () }
        
        self.Features = { 'RQon' : (),
                          'RQ' : (),
                          'RS' : (), 
                          'RSoff' : (),
                          'RT' : (), 
                          'RTon' : (),
                          'RToff' : (), 
                          'RQon_amp' : (),
                          'RQ_amp' : (),
                          'RS_amp' : (), 
                          'RSoff_amp' : (),
                          'RT_amp' : (), 
                          'RTon_amp' : (),
                          'RToff_amp' : (),  }
        
        #---------------------------------------------------------------------
        # interval Features
        #---------------------------------------------------------------------
        # characteristics waves
                                     
        self.Features ['RQon'] = np.subtract(self.FP['R'][1:N-1],self.FP['Qon'][1:N-1])
        self.Features ['RQ']   = np.subtract(self.FP['R'][1:N-1],self.FP['Q'][1:N-1])
        self.Features ['RS'] = abs(np.subtract(self.FP['R'][1:N-1],self.FP['S'][1:N-1]))
        self.Features ['RSoff'] = abs(np.subtract(self.FP['R'][1:N-1],self.FP['Soff'][1:N-1]))
        self.Features ['RTon'] = abs(np.subtract(self.FP['R'][1:N-1],self.FP['Ton'][1:N-1]))
        self.Features ['RT'] = abs(np.subtract(self.FP['R'][1:N-1],self.FP['T'][1:N-1]))
        self.Features ['RToff'] = abs(np.subtract(self.FP['R'][1:N-1],self.FP['Toff'][1:N-1]))
        
        # Morphology Features
        temp1 =[]
        temp2 =[]            
        for M,M2,R in  zip(self.FP ['QRS_Morph'][1:N-1],
                        self.FP ['T_Morph'][1:N-1],self.FP['R'][1:N-1]) :

            temp1.append(abs(np.subtract(R,M)))
            temp2.append(abs(np.subtract(R,M2)))
        self.Morph_Features ['RqrsMorph'] = np.int_(temp1) 
        self.Morph_Features ['RTMorph'] = np.int_(temp2) 

            

        #---------------------------------------------------------------------
        # Amplitude features
        #---------------------------------------------------------------------   
           
        # characteristics waves
        self.Features ['RQon_amp']  = np.subtract(self.Norm1_ecg[self.FP['R'][1:N-1]],
                                            self.Norm1_ecg[self.FP['Qon'][1:N-1]])
        self.Features ['RQ_amp']    = np.subtract(self.Norm1_ecg[self.FP['R'][1:N-1]],
                                                 self.Norm1_ecg[self.FP['Q'][1:N-1]])
        self.Features ['RS_amp']    = np.subtract(self.Norm1_ecg[self.FP['R'][1:N-1]],
                                                self.Norm1_ecg[self.FP['S'][1:N-1]])   
        self.Features ['RSoff_amp'] = np.subtract(self.Norm1_ecg[self.FP['R'][1:N-1]],
                                                 self.Norm1_ecg[self.FP['Soff'][1:N-1]])
        self.Features ['RTon_amp']  = np.subtract(self.Norm1_ecg[self.FP['R'][1:N-1]],
                                                 self.Norm1_ecg[self.FP['Ton'][1:N-1]])   
        self.Features ['RT_amp']    = np.subtract(self.Norm1_ecg[self.FP['R'][1:N-1]],
                                                 self.Norm1_ecg[self.FP['T'][1:N-1]])              
        self.Features ['RToff_amp'] = np.subtract(self.Norm1_ecg[self.FP['R'][1:N-1]],
                                                 self.Norm1_ecg[self.FP['Toff'][1:N-1]])   
                                                 
        # Morphology Features
        temp1 =[]
        temp2 =[]            
        for M,M2,R in  zip(self.FP ['QRS_Morph'][1:N-1],
                        self.FP ['T_Morph'][1:N-1],self.FP['R'][1:N-1]) :                                       
            temp1.append(np.subtract(self.Norm1_ecg[R],self.Norm1_ecg[M]))
            temp2.append(np.subtract(self.Norm1_ecg[R],self.Norm1_ecg[M2])) 
        self.Morph_Features ['RqrsMorph_amp'] = np.float_(temp1)                                  
        self.Morph_Features ['RTMorph_amp'] = np.float_(temp2)                                        

      
        return self.Features,self.Morph_Features
   
    def visualize_QRS_detection(self, savefilename = False):
        """Plot the ecg at various steps of processing for qrs detection.
        Will not plot more than 10 seconds of data.
        If filename is input, image will be saved"""
        ecglength = len(self.raw_ecg)
        ten_seconds = 10 * self.sampling_rate
        
        if ecglength > ten_seconds:
            segmentend = ten_seconds
        else:
            segmentend = ecglength
    
        segmentQRSpeaks = [peak for peak in self.FP['R'] if peak < segmentend]
        
    
        plt.figure("%s region"  %self.ascultation_region,figsize  = (20, 15))
        plt.subplot(611)
        plt.plot(self.data[:segmentend])
        plt.ylabel('Filtered ECG', rotation='vertical')
        plt.subplot(612)
        plt.plot(self.Norm1_ecg[:segmentend])
        plt.ylabel('Normalized ECG',rotation='vertical')
        plt.subplot(613)
        plt.plot(self.diff_ecg[:segmentend])
        plt.ylabel('Differential',rotation='vertical')
        plt.subplot(614)
        plt.plot(self.abs_ecg[:segmentend])
        plt.ylabel('Squared differential',rotation='vertical')
        plt.subplot(615)
        plt.hold(True)
        plt.plot(self.int_ecg[:segmentend])
        plt.ylabel('Integrated', rotation='vertical')
        plt.subplot(616)
        plt.plot(self.raw_ecg[:segmentend])
        plt.plot(segmentQRSpeaks, self.raw_ecg[segmentQRSpeaks], 'ro')
        plt.hold(False)
        plt.ylabel('QRS peaks', rotation='vertical')

        if savefilename:
            plt.savefig(savefilename)
        else:
            pass

    def visualize_fiducialPoints(self):  
        
        win = 10
        ecglength = len(self.raw_ecg)
        ten_seconds = win * self.sampling_rate
        
        if ecglength > ten_seconds:
            segmentend2 = ten_seconds
            
        else:
            segmentend2= ecglength
            #win = 10
        
        
        #print segmentend2 
        t =np.linspace(0,len(self.raw_ecg[:int(segmentend2)]),len(self.raw_ecg[:int(segmentend2)]))
        t = t / self.sampling_rate 
       # """

        f, (ax1) = plt.subplots(figsize  = (30, 5) )
        ax1.hold(True)
        ax1.plot(t,self.raw_ecg[:int(segmentend2)],color = '0.5')
        colors = ['Orange', 'g', 'r', 'm', 'y', 'b','c','k']
        Halignment = ['left','right','left', 'right' ]
        valignment= ['top','bottom','top','bottom']
        ax1.set_title('Fiducial Points')
        for name,c,h,v  in zip(self.FP,cycle(colors),cycle(Halignment), cycle(valignment)):
            if name == 'QRS_Morph' or name == 'T_Morph' :
                for peak1,value1 in zip(self.FP[name][:][:win+1] 
                                      , self.raw_ecg[self.FP[name][:][:win+1] ]):
                    ax1.plot(peak1/self.sampling_rate, value1, 'o', mfc= 'none',color= c,linewidth=5 )
            else :
                for peak,value in zip(self.FP[name][:win+1] 
                                  , self.raw_ecg[self.FP[name][:win+1]]):
                    ax1.plot(peak/self.sampling_rate, value, 'o', color = c)
                    ax1.annotate(name, xy=(peak/self.sampling_rate, value),  xycoords='data',
                    xytext=(3, 3), textcoords='offset points',horizontalalignment=h, verticalalignment=v) 
        ax1.set_ylabel("Amplitude(V)")
        ax1.set_xlabel("Time [s]")
        ax1.set_title("%s region" %self.ascultation_region) 
        
        #TODO:Debuging figure- should be removed or refactor

        # f, (ax1, ax2) = plt.subplots(2, sharex=True,figsize  = (20, 10) )
        # ax1.hold(True)
        # ax1.plot(t,self.raw_ecg[:segmentend2],color = '0.75')
        # colors = ['Orange', 'g', 'r', 'm', 'y', 'b','c','k']
        # Halignment = ['left','right','left', 'right' ]
        # valignment= ['top','bottom','top','bottom']
        # ax1.set_title('Fiducial Points')
        # for name,c,h,v  in zip(self.FP,cycle(colors),cycle(Halignment), cycle(valignment)):

        #     if name == 'QRS_Morph' or 'T_Morph':
        #         for peak1,value1 in zip(self.FP[name][:][:win+1] 
        #                               , self.raw_ecg[self.FP[name][:][:win+1] ]):
        #             ax1.plot(peak1/self.sampling_rate, value1, 'o', color= c,linewidth=5 )
                    
        #     else:
        #         for peak,value in zip(self.FP[name][:win+1] 
        #                               , self.raw_ecg[self.FP[name][:win+1]]):
        #             ax1.plot(peak/self.sampling_rate, value, 'o', color = c)
        #             # ax1.annotate(name, xy=(peak/self.sampling_rate, value),  xycoords='data',
        #             # xytext=(3, 2), textcoords='offset points',horizontalalignment=h, verticalalignment=v) 

        # ax1.set_ylabel("Amplitude(mV)")
        # ax1.hold(False)
        
                  
        # ax2.set_title('Moving Window Integrated')
        
        # ax2.hold(True)
        # ax2.plot(t,self.int_ecg[:segmentend2])  
        # t2 = self.plot_peaks[:win+2]
        
        # t2 =[elem/self.sampling_rate for elem in t2]
        
        # ax2.plot(t2, self.plot_sig_thresh[: win +2], 'r')
        
        # for name,c,h,v  in zip(self.FP,cycle(colors),cycle(Halignment), cycle(valignment)):
           
        #    if name != 'QRS_Morph' or 'T_Morph':
        #        for peak,value in zip(self.FP[name][:win+1] 
        #                                                   , self.int_ecg[self.FP[name][:win+1]]):
        #             ax2.plot(peak/self.sampling_rate, value, 'o', color = c)
        #             # ax2.annotate(name, xy=(peak/self.sampling_rate, value),  xycoords='data',
        #             # xytext=(3, 2), textcoords='offset points',horizontalalignment=h, verticalalignment=v)    
        # ax2.set_xlabel("Time [s]"); 
        # ax2.set_ylabel("Amplitude(mV)")
        # ax2.set_title("%s region"  %self.ascultation_region) 

    def beat_segmentor(self):
        #TODO: clean the beat segmentor code
        import matplotlib.gridspec as gridspec
        from mpl_toolkits.mplot3d import axes3d
        from matplotlib import cm
        
        N=len(self.FP['R'])
        # Normal max intervals in ms 
        RR_interval = 0.12
        QRS_interval = 0.12
        ST_interval = 0.32
        QT_interval = 0.42  
        
        def timeAxis (len_data ,fs):
            start =  np.true_divide(1, fs) 
            stop  =  np.true_divide(len_data, fs) 
            t =np.linspace(start,stop,len_data)
            return t
        
        ecglength = len(self.Norm1_ecg)
        offset_left =  int(0.2 * self.sampling_rate)
        offset_right = int(0.42 * self.sampling_rate)
        self.segments =[]
        
        for Q,R in zip(self.FP['Q'][1:N-1] ,self.FP['R'][1:N-1] ):  
            self.segments.append (self.Norm1_ecg[R-offset_left: Q +offset_right ] )
            
        self.seg_df = pd.DataFrame( self.segments)
        self.seg_df =self.seg_df.fillna(0)
        mean_beat = self.seg_df.mean(axis=0)
        Std_beat  = self.seg_df.std(axis=0)    
       
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k','orange']
    
        fig = plt.figure(figsize  = (32,7 ))
        gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1,2])
        ax = fig.add_subplot(gs[0])
        
        #plt.subplot(112)
        ax.hold(True)
        for seg,c  in  zip(self.segments,cycle(colors)):
            t =timeAxis (len(seg),self.sampling_rate)
            ax.plot(t,seg,color =c)
            ax.set_xlabel("Time [s]")
            ax.set_title('Segmented ECG')
            break
        
        ax = fig.add_subplot(gs[1] )
        t =timeAxis (len(mean_beat),self.sampling_rate)    
        ax.plot( t, mean_beat, 'k',  label = 'Mean',alpha = 0.8,linewidth=2)
        ax.errorbar(t,mean_beat, yerr=Std_beat, fmt='g--',label = 'Std',alpha = 0.3)
        ax.legend(frameon=False, fontsize=14,loc= 'best')
        ax.set_xlabel("Time [s]"); 
        ax.set_ylabel("Normalized Amplitude(V)")
        ax.set_title('Average and Standard deviation for %r ECG beats from %s' %(len(self.seg_df) ,self.ascultation_region))
        #plt.show()
                

        t = timeAxis (len(mean_beat),self.sampling_rate)
        
        list= []
        Y = pd.DataFrame()
        Z = self.seg_df.values
        X = self.seg_df.index
        for i in xrange (len(self.seg_df)):
            data = pd.DataFrame(t)
           
            list.append( data.transpose()) 
            Y = pd.concat(list)
        x = t
        y = self.seg_df.index
        y,X  = np.meshgrid(x, y)
        #%plt qt 
        #%plt inline
       
        
        #fig = plt.figure( figsize  = (15, 10))
        ax = fig.add_subplot(gs[2], projection='3d')
        ax.hold(True)
        surf=ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.8,cmap=cm.coolwarm,
              linewidth=0, antialiased=False)
        ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10,alpha=0.7,color="Orange", linewidth=1)
        cset = ax.contourf(X, Y, Z, zdir='x', offset=-5, cmap=cm.coolwarm)
        fig.colorbar( surf, shrink=0.5)       
        ax.set_title('Segmented ECG')
        ax.set_xlabel('QRS')
        ax.set_xlim(-2, 10)
        ax.set_ylabel('Time [s]')
        ax.set_ylim(-0.1, 0.7)
        ax.set_zlabel('Normalized Amplitude(V)')
        ax.set_zlim(min((mean_beat))-0.07, max(abs(mean_beat))+0.07)
        
        for ii in xrange(0,360,1):
           ax.view_init(elev=30, azim=ii*-45)   
   

       
        return  self.seg_df.values,mean_beat
        

if __name__ == "__main__":

    # filename = r'C:\Users\User\Google Drive\Summer 2014\Datasets\RB\Aortic_RB01.csv'
    # raw_data = np.genfromtxt(filename, delimiter=',')
 
    # SP info command line input: 
    # SP_info = { 'name' : (), 'age'  : () , 'sex'  : (), 'sampling_rate' :() }
    # SP_info ['name'] = raw_input('Name -->')
    # SP_info ['age'] =  raw_input('Age -->')
    # SP_info ['sex'] = raw_input('Sex (Male/ female)-->')
    # SP_info ['sampling_rate'] = raw_input('sampling_rate-->')
    SP_info = { 'name' : 'foo', 'age'  :27 , 'sex'  : 'Male' , 'sampling_rate' :1000,
                'ascultation_region' : ()}
             
    # Testing Feature extraction
    #=========================================================================

    # Preprocessing :
    #-------------------------------------------------------------------------
    # remove powerline noise
    from Signalpreprocessor import ECGPreprocessor
    from os.path import basename
    import glob
    import re
    from pprint import pprint

    morph_features_names = ['RqrsMorph','RTMorph', 'RqrsMorph_amp' ,
                                'RTMorph_amp' ]
        
    features_names = ['RQon' , 'RQ' ,  'RS' ,'RSoff', 'RTon', 'RT','RToff', 
                          'RQon_amp','RQ_amp' ,'RS_amp' ,  'RSoff_amp' ,
                          'RT_amp' , 'RTon_amp' , 'RToff_amp' ]
        


    path = r'C:\Users\User\Google Drive\Summer 2014\Datasets\RB\test'
    allwav_Files = glob.glob(path + "/*.csv")
    last = len(allwav_Files)
    temp = []
    temp_morph = []
    target = []
    n_files = 0
    count = 0
    process = ECGPreprocessor()
    for filename in allwav_Files:
        print filename
        # SP_info['ascultation_region'] = re.search("^[^_]*", basename(filename)).group(0)
        SP_info['ascultation_region'] = re.search("^[^_]*", basename(filename)).group(0)
        logger.info('Ascultation Region : %s' ,SP_info['ascultation_region'])

        process.wavread(filename, test_mode=True)

        process.down_sample()
      
        FIR_para = {'cutoff': 40, 'tranwidth': 10, 'Rp': 45}
        process.design_FIR(FIR_para)
        process.apply_FIRfilter(process.downsampled_data)

        FE = FeatureExtractor(process.filtered_data,SP_info)
        FE.qrsDetect()

        #FE.visualize_QRS_detection()

        FE.visualize_fiducialPoints()
        #FE.beat_segmentor()
        plt.show()
     

        features, Morph_Features = FE.features_extracted()

        features_test = np.array([features[key] for key in features_names])
        # if  SP_info['ascultation_region'] != 'Aortic':
        #     break
        Morph_Features_test = np.hstack([Morph_Features[m] for m in morph_features_names])
        
        features_test= features_test.T
        #Morph_Features_test = Morph_Features_test.T
        temp.append(features_test)
        temp_morph.append(Morph_Features_test)
        [target.append( SP_info['ascultation_region']) for x in features_test]

    #print Morph_Features_test.shape    
    # MorphFeatures = np.row_stack(temp_morph)
    # Features = np.row_stack(temp)
    # target = np.row_stack(target)
    # extractedfeatures = r'C:\Users\nkida001\Google Drive\Summer 2014\Datasets\all_RB\features_0620.csv'
    # np.savetxt(extractedfeatures, Features, delimiter=",")

    # target_file = r'C:\Users\nkida001\Google Drive\Summer 2014\Datasets\all_RB\target_0620.csv'
    # np.savetxt(target_file, target, delimiter=",", fmt="%s")

    # morpfeatures_file = r'C:\Users\nkida001\Google Drive\Summer 2014\Datasets\all_RB\morph_0620.csv'
    # np.savetxt(morpfeatures_file, MorphFeatures,delimiter=",")



