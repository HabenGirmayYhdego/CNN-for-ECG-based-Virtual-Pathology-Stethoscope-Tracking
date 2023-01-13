#!/usr/bin/python
# Signalpreprocessor.py

from __future__ import division
import numpy as np
import numba as nb
from scipy import signal
import matplotlib.pyplot as plt
from scipy.io import wavfile
import glob
#import zmq_client_v1 as zq
#btype = ['bandpass', 'lowpass', 'highpass', 'bandstop']


class Signalpreprocessor(object):

    # defalut sampling rate is 1e3
    nyq_rate = None       
    sample_rate = None
    def __init__(self,sample_rate=1e3):
 
        self.nyq_rate = sample_rate / 2.0       
        self.sample_rate = sample_rate  
      
        

    def design_IIR(self,filter_para,ftype='butter', btype='Bandpass'):        
        """
        IIR filter design ulitity class, uses scipy.signal.irrdesign
        input:
            filter_para: disc
            fpass: starting frequency(Hz)                    
            fstop: stopping frequency (Hz)will be changed to (Ws = start/nyquistWs)
      
            gpass: passband maximum loss (gpass)
            gstop: stoppand min attenuation (gstop)
            btaps: type of filter defalut is bandpass

        usage:
            frequencies will be normalized from 0 to 1, where 1 is the Nyquist 
            frequency, pi radians/sample. 
            (wp and ws are thus in half-cycles / sample.)
                Lowpass: wp = 0.2, ws = 0.3
                Highpass: wp = 0.3, ws = 0.2
                Bandpass: wp = [0.2, 0.5], ws = [0.1, 0.6]
                Bandstop: wp = [0.1, 0.6], ws = [0.2, 0.5]
        return   
        coefficients
                  
        """ 
        filter_para = {'fpass':5, 'fstop':30 , 'gpass':1.0, 'gstop':60}
                   
        # The butter and cheby1 need less constraint spec
        coefficients = signal.iirdesign(filter_para['fpass'] / self.nyq_rate, 
                                        filter_para['fstop'] / self.nyq_rate, 
                                        Rp,As,btype,
                                        ftype=ftype, output ='ba')
          

        return coefficients

    def design_FIR(self,filter_para,pass_zero=True):        
        """
        FIR filter design ulitity class, uses scipy.signal.irrdesign
        input:
            filter_para: disc
            width: the width of the transition region                  
            Rp: passband maximum ripple
            cut:cutoff hz can be array for BP
            pass_zero : bool
            If True, the gain at the frequency 0 (i.e. the "DC gain") is 1.
             Otherwise the DC gain is 0. deafult true

        usage:
            frequencies will be normalized from 0 to 1, where 1 is the Nyquist 
            frequency, pi radians/sample. 
            (wp and ws are thus in half-cycles / sample.)
                Lowpass: pass_zero = True
                Highpass: pass_zero = False
                Bandpass: pass_zero = False
                Bandstop: pass_zero = True
            return 
            
        """ 
        self.cut_off =filter_para['cutoff'] 
        # Compute the order and Kaiser parameter for the FIR filter.

        self.M, beta = signal.kaiserord(filter_para['Rp'], 
                                   filter_para['tranwidth'] / self.nyq_rate)
         #TODO : save to file after finding the best one.

                              
        # Use signal.firwin with a Kaiser window to create a lowpass FIR
        # filter.
        self.taps = signal.firwin(self.M,  self.cut_off / self.nyq_rate, 
                             window=('kaiser', beta),
                             pass_zero = pass_zero)
    
    def plot_taps(self):

        #------------------------------------------------
        # Plot the FIR filter coefficients.
        #------------------------------------------------

        plt.figure('FIR filter coefficients')
        plt.plot(self.taps, 'bo-', linewidth=2)
        plt.title('Filter Coefficients (%d taps)' % self.M)
        plt.grid(True)
        plt.show()

    def plot_response(self):
        #------------------------------------------------
        # Plot the magnitude response of the filter.
        #------------------------------------------------
        #TODO: make dynamic, better plots
        plt.figure( 'Magnitude Response')
        plt.clf()
        w, h = signal.freqz(self.taps, worN=8000)
        plt.plot((w / np.pi) * self.nyq_rate, np.absolute(h), linewidth=2)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Gain')
        plt.title('Frequency Response')
        plt.ylim(-0.05, 1.05)
        plt.grid(True)

        # Upper inset plot.
        ax1 = plt.axes([0.42, 0.6, .45, .25])
        plt.plot((w / np.pi) * self.nyq_rate, np.absolute(h), linewidth=2)
        plt.xlim(0,15)
        plt.ylim(0.9985, 1.001)
        plt.grid(True)

        # Lower inset plot
        ax2 = plt.axes([0.42, 0.25, .45, .25])
        plt.plot((w / np.pi) * self.nyq_rate, np.absolute(h), linewidth=2)
        plt.xlim(self.cut_off+5,self.cut_off+15)
        plt.ylim(0.0, 0.0025)
        plt.grid(True)
        plt.show()


class  ECGPreprocessor(Signalpreprocessor):

    def __int__(self):

        super(Signalpreprocessor, self).__init__()

        self.testmode = True

    def down_sample(self,new_rate = 1e3):    
        """ Down-sample the signal by using a filter.
        By default, an order 8 Chebyshev type I filter is used. 
        A 30 point FIR filter with hamming window is used if ftype is
        
        input:
        x: the input signal 
        sample_rate: the input signal sampling rate 
        new_rate: the new sample rate
        return:
            the downsampled signal
        """
        # Decimation Rate
        
        factor = np.int(self.sample_rate / new_rate)
        if factor == 0:
            factor =1
        else:
            pass

        if self.testmode:
             self.rawTime_index = self.time_index 
        else:
            pass
       
        self.sample_rate = new_rate
        self.downsampled_data = signal.decimate(self.raw_data,factor)
        self.data_window = 10 * self.sample_rate 
        self.data_length  = len(self.downsampled_data)
        self.time_index =  np.arange(self.data_length) / self.sample_rate
      

    def smooth(self,window_len=11,window='hanning'):
        """smooth the data using a window with requested size.
    
        This method is based on the convolution of a scaled window with the signal.
        The signal is prepared by introducing reflected copies of the signal 
        (with the window size) in both ends so that transient parts are minimized
        in the begining and end part of the output signal.
    
        input:
            x: the input signal 
            window_len: the dimension of the smoothing window; should be an odd integer
            window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
                flat window will produce a moving average smoothing.

        return:
            the smoothed signal
        
     
        NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
        """ 
        data = self.downsampled_data
        if data.ndim != 1:
            raise ValueError, "smooth only accepts 1 dimension arrays."

        if data.size < window_len:
            raise ValueError, "Input vector needs to be bigger than window size."
        

        if window_len < 3:
            return data
    
    
        if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
            raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
    
        s = np.r_[data[window_len - 1:0:-1],data,data[-1:-window_len:-1]]
        
        if window == 'flat': #moving average
            w = np.ones(window_len,'d')
        else:
            w = eval('np.' + window + '(window_len)')
    
        y = np.convolve(w / w.sum(),s,mode='valid')
        print len(data)
        print len(y)
        
        self.smooth_data = (y[(window_len / 2 - 1):-(window_len / 2)])
        print len(self.smooth_data)

    def apply_IIRfilter(self,data):
        """Bandpass filter the ECG from 5 to 15 Hz"""
        # TODO: Explore - different filter designs
        nyq_rate = sample_rate / 2.0  
        #wn = [30/ nyq_rate, 40/ nyq_rate]
        wn = [30 / nyq_rate]
        b,a = signal.butter(2, wn, btype = 'lowpass')        
        return signal.filtfilt(b,a,data)
        
    def apply_FIRfilter(self,data, powerline = True):
        """Bandpass filter the ECG from 0.7 to 30 Hz"""
        if powerline:
            x = signal.lfilter(self.taps, 1.0, data)
            delay = 0.5 * (self.M-1) / self.sample_rate
            self.delayedtime_index = self.time_index[self.M-1:]-delay
            #self.filtered_data = x[self.M-1:]
            self.filtered_data = x[self.M-1:]
            self.delay1= delay
        else:
            
            x = signal.lfilter(self.taps, 1.0, data)
            delay = (self.M-1) / self.sample_rate  + self.delay1 * 0.5
            #delay = (self.M-1) / self.sample_rate  + self.delay1 * 0.5
            self.delayedtime_index2 = self.delayedtime_index[self.M-1:]-(delay)
            self.baseline =   x[self.M-1:]
            self.BLfiltered_data = self.filtered_data[self.M-1:]-self.baseline
            self.BLfiltered_data = self.BLfiltered_data- np.mean(self.BLfiltered_data)


    def wavread(self,filename,test_mode = True):

        """Read ECG signals from file or real time"""
        if test_mode:
        # mmap:read data as memory mapped
            #self.sample_rate, data = wavfile.read(filename,mmap=True) 
            self.raw_data = np.genfromtxt(filename, delimiter=',')
            self.sample_rate = len(self.raw_data)/10.0
            #self.raw_data = data[:,0]
            self.testmode = test_mode
            self.data_length  = len(self.raw_data)
            self.rawTime_index =  np.arange(self.data_length) / self.sample_rate

        else:
            self.raw_data = np.asfarray (self.raw_data)
            self.sample_rate = len(self.raw_data)/10.0
            self.testmode = test_mode 
        self.data_length  = len(self.raw_data)         
        self.data_window = 10 * self.sample_rate        
        self.time_index =  np.arange(self.data_length) / self.sample_rate
    

    #TODO : 

    def Visualize_signal(self, savefilename = False):
        """Plot the ecg at various steps of processing for qrs detection.
        Will not plot more than 10 seconds of data.
        If filename is input, image will be saved"""
           
        colors = ['red','green','Brown','blue',
                'DarkBlue','Tomato','Violet', 'Tan','Salmon','Pink',
                'SaddleBrown', 'SpringGreen', 'RosyBrown','Silver']
        if self.data_length > self.data_window:
            segmentend = self.data_window
        else:
            segmentend = self.data_length
      
      
     
        if self.testmode:
            plt.figure('Raw ECG')
            #plt.plot(self.rawTime_index,self.raw_data,
             #   c= colors [0],label='raw ECG')
            plt.plot(self.rawTime_index,self.raw_data,
                c= colors [0],label='raw ECG')

            plt.figure(figsize  = (17, 10))                    
            plt.plot(self.time_index[:int(segmentend)],self.downsampled_data[:int(segmentend)],
                    c= colors [0],label='downsampled ECG') 


            plt.plot(self.delayedtime_index[
                        :int(segmentend)], self.filtered_data[:int(segmentend)],
                        c=colors[1], label='filtered ECG')

#==============================================================================
#             plt.plot(self.delayedtime_index2[
#                         :segmentend], self.baseline[:segmentend],
#                     c=colors[2], label='baseline')
# 
#             plt.plot(self.delayedtime_index2[
#                         :segmentend], self.BLfiltered_data[:segmentend],
#                      c=colors[3], label='baseline_removed ECG')
#==============================================================================


        else:  
            #plt.figure(figsize  = (17, 10))    
            # plt.plot(self.time_index,self.raw_data,
            #                     c= colors [0],label='raw ECG')
            plt.plot(self.time_index[:int(segmentend)],self.downsampled_data[:int(segmentend)],
                    c= colors [0],label='downsampled ECG')  

            plt.plot(self.delayedtime_index[
                    :int(segmentend)], self.filtered_data[:int(segmentend)],
                    c=colors[1], label='filtered ECG')
        



        plt.legend(loc='best')
        plt.title('Preprocessing')

    
        if savefilename:
            plt.savefig(savefilename)
        else:
            plt.show()


    def spectral_density (self,x):
        plt.figure(" periodogram")

        f, Pper_spec = signal.periodogram(x, self.sample_rate, 'flattop', scaling='spectrum')

        plt.semilogy(f, Pper_spec)
        plt.xlabel('frequency [Hz]')
        plt.ylabel('PSD')
        plt.grid()
        plt.show()



        plt.figure("welch")
        f, Pxx_den = signal.welch(x ,self.sample_rate, nperseg=1024)
        #f, Pxx_den = signal.welch(x ,self.sample_rate, nperseg= 2014)
        plt.semilogy(f, Pxx_den)
        plt.grid(True)
        plt.xlabel('frequency [Hz]')
        plt.xlim(0,100)
        plt.ylabel('PSD ')
        plt.show()
        print "Noise power :%r"  %np.mean(Pxx_den[50:])    
 
 
        plt.figure("welch")
        f, Pxx_spec = signal.welch(x, self.sample_rate, 'flattop', 1024, scaling='spectrum')
        #f, Pxx_spec = signal.welch(x, self.sample_rate, 1024, scaling='spectrum')        
        plt.figure(5)
        plt.xlim(0,100)
        plt.semilogy(f, np.sqrt(Pxx_spec))
        plt.xlabel('frequency [Hz]')
        plt.ylabel('Linear spectrum ')
        plt.show()     

    
if __name__ == "__main__":

    filename = r'D:\ALL_Male_Seated\Aortic\SP5SA90 (2).wav'

    #path = r'D:\ALL_Male_Seated\Aortic'
    #allFiles = glob.glob(path + "/*.wav")  
     
    #for filename in allFiles:    
    process = ECGPreprocessor()
    process.wavread(filename)
        
    #    process.wavread(filename)
    #    process.down_sample()
    process.spectral_density(process.raw_data)
    process.down_sample()
    # process.spectral_density(process.raw_data)
        
    # Apply filter    
    # remove powerline noise
    # FIR_para = {'cutoff':60, 'tranwidth':10 , 'Rp':60}
    # process.design_FIR(FIR_para)
    # process.plot_taps()
    # process.plot_response() 
    # process.apply_FIRfilter(process.raw_data)
    # process.spectral_density(process.filtered_data)
        
    # remove baseline wandering 
    ##==========================================================================
    #    FIR_para = {'cutoff':1, 'tranwidth':5 , 'Rp':60}
    #    process.design_FIR(FIR_para)
    #    process.apply_FIRfilter(process.filtered_data,powerline =False)
        
    #    process.plot_taps()       
    #    #rocess.plot_response()  
    ##==========================================================================
             
             
     
    
    process.Visualize_signal()


# End of Signalpreprocessor.py