import scipy  as sc
import numpy as np
from scipy.fft import fft, fftfreq, rfft, rfftfreq
from ..MathFunctions.MFunction import func2D as func



class SignalProcessing():

    def __init__(self):
        pass

    @staticmethod
    def fft(SampleRate, Duration, func: func, key=""):
        '''
        Easy fft for signals!
        '''
        assert key in ['both', 'freq', ''], "key should be in ['both', 'freq', '']."
        
        Signal = func.NumericTerm['y']
        SampleLength = SampleRate * Duration
        Amplitudes = fft(Signal)
        Frequencies = fftfreq(SampleLength,1/SampleRate)
        # Returns
        if key == 'both':
            return Amplitudes, Frequencies
        elif key =='freq':
            return Frequencies
        elif key =='':
            return Amplitudes
        else:
            pass

    @staticmethod
    def rfft(SampleRate, Duration, Signal, key=""):
        '''
        Easy fft for signals!
        '''
        assert key in ['both', 'freq', ''], "key should be in ['both', 'freq', '']."
        SampleLength = int(SampleRate * Duration)
        Amplitudes = rfft(Signal)
        Frequencies = rfftfreq(SampleRate,1/SampleRate)
        # Returns
        if key == 'both':
            return Amplitudes, Frequencies
        elif key =='freq':
            return Frequencies
        elif key =='':
            return Amplitudes
        else:
            pass
'''
    @staticmethod
    def HurstExponent (Data: func, Ranges: list):
        # 
       # Given partial lenghts as a list this Function Calculates the Hurst Exponent of the function.
        #
        lenght = len(Data.NumericTerm['y'])
        d = Data.NumericTerm['y']
        Result = np.zeros(len(Ranges))

        for rindex,r in enumerate(Ranges):
            
            Steps = lenght // r
            PartialTimeSeries = np.zeros(Steps)
            Lowerbound = 0
            Upperbound = r

            for pindex, p in enumerate(PartialTimeSeries):
                
                mean = np.average(Data[Lowerbound:Upperbound]) # [x]
                Std  = np.std(Data[Lowerbound:Upperbound])     # [x]
                
                print(Std, mean)
                MeanAdjustedValues = np.subtract(d[Lowerbound:Upperbound] , mean)
                CumulativeSum = MeanAdjustedValues.cumsum() 

                R = np.max(CumulativeSum) - np.min(CumulativeSum)
                PartialTimeSeries[pindex] = R/Std

                Lowerbound += r
                Upperbound += r

                if Upperbound > lenght:
                    break

            
            Result[rindex] = np.average(PartialTimeSeries)

        RescaledRange = Result 
        Hurstexp = np.polyfit(x=Ranges , y = np.log(RescaledRange),deg=1)[0]
        
        return Hurstexp


    #@staticmethod
    #def Maxmin (Data: func, Range: list) -> list: # START
    #    pass
'''
