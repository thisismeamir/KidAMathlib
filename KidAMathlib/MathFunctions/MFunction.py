from math import ceil
import numpy as np
import scipy as sc
import pandas as pd
from ..Generators.GeneralGenerators import GeneralGenerators as gg
from ..Generators.GeneralGenerators import BrownianMotion 

'''
We first define a class of mathematical functions.
Symbolic functions are not yet implemented...
'''

class func2D():
    '''
    Class of mathematical Functions.
    '''
    def __init__(self, Numpy = "", Numeric = "", Symbols = "", SampleRate='44100'):
        '''
        This method is the initial function for making a 2D mathematical function.
        First define Symbols in your function and then,
        you can use self.Numpy('Numpy expression: np.cos(x)') 
        or          self.Numeric ({"x": [], "y": []})
        to define the function in terms of coordinate numbers or symbolic term
        '''
        # This is used only for the signal Processing and Generating methods 
        self.SampleRate = SampleRate
        self.Samples    = {}
        if Numpy != "":
            self.Numpy(Numpy,Symbols)
        elif Numeric != "":
            self.Numeric(Numeric)
        else:
            pass

    # ---------- Defining Methods ----------
    def Numpy(self, Expression, Symbols):
        '''
        This method gets a numpy function as a string and turns it to actual numpy function.
        '''

        Define = "lambda" + " " + Symbols + " :"
        exec('self.NumpyTerm = {0} {1}'.format(Define, Expression))
    
    def Numeric(self, Numeric):
        '''
        This method is resposible for recieving a set of coordinates and the coresponding value of the function.
        {'x': [1,3,4,5,6,7,8,15,17,18,100,230,...],
         'y': [13,15,1,6,4,52,37,37,563,6,745,...]}
         the lengths of 'x' and  'y' lists have to be equal. 
        '''
        assert len(Numeric['x']) == len(Numeric['y']), 'The two given lists are not equal'
        
        self.NumericTerm = Numeric
    
    # ---------- Signal Generator ----------
    def AddSample(self, Range: list, Length = 1, SampleName = "", Reverse = True):
        '''
        Making a Sample from a given numpy signal
        '''
        assert len(Range) == 2, "Range should be in the form  [a,b]."
        if Reverse:
            assert len(self.NumericTerm['y'][Range[0]:Range[1]]) * 2 == self.SampleRate, " The range you chose does not have enough smaples to produce the Sample."
            
            Sample = np.concatenate(self.NumericTerm['y'][Range[0]:Range[1]],np.flipud(self.NumericTerm['y'][Range[0]:Range[1]]))
            Sample = np.tile(Sample, Length)
        
        else:
            assert len(self.NumericTerm['y'][Range[0]:Range[1]]) == self.SampleRate, " The range you chose does not have enough smaples to produce the Sample."
            
            Sample = self.NumericTerm['y'][Range[0]:Range[1]]
            Sample = np.tile(Sample, Length)
        
        self.Samples[f'{SampleName}'] = Sample


    def RemoveSample(self, SampleName):
        del self.Samples[f'{SampleName}']

    def ShowSample(self, SampleName = 'All'):
        if SampleName == 'All':
            print(self.Samples)
        else:
            print(self.Samples[f'{SampleName}'])

    def PulseMaker(self, Range: list, Length = 1 ):
        '''
        Making a pulse of Data
        As you give a length and a pulse region, this method returns a numpy array with 
        '''
        assert len(Range) == 2, "Range should be in the form  [a,b]."
        assert len(self.NumericTerm['y'][Range[0]:Range[1]]) == self.SampleRate, " The range you chose does not have enough smaples to produce the Sample."
        
        Pulse = np.zeros(int(ceil(Length*self.SampleRate / 2)))
        Pulse = np.concatenate(Pulse,self.NumericTerm['y'][Range[0]:Range[1]])
        Pulse = np.concatenate(Pulse,np.zeros(int(ceil(Length*self.SampleRate / 2))))
        return Pulse

    # ---------- Turning Numeric ----------
    def TurnNumpyNumeric(self, Sample, Symbols="", Signal = False):
        '''
        Turns Numpy Expressions into Numeric functions.
        '''
        self.SampleLength = len(Sample)
        self.SampleRate   = self.SampleLength//self.SampleDuration
        Sample = Sample
        y =[]
        if Symbols == "":
            for x in Sample:
                Eval = self.NumpyTerm(x)
                y.append(Eval)
        else:
            for x in Sample:
                # Eval = 0
                Eval = eval("self.NumpyTerm({0},{1})".format(x,Symbols))
                y.append(Eval)
        if Signal:
            return {'x': Sample, 'y': np.array(y)}
        else:
            self.NumericTerm = {'x': Sample, 'y': np.array(y)}

    # ---------- Data Import and Cropping ----------
    def ImportFromCSV(self,path, columns =['','']):
        '''Importing Numeric Values from a CSV file.'''
        df = pd.read_csv(path, skipinitialspace=True)
        x = np.transpose(df[columns[0]].to_numpy())
        y = np.transpose(df[columns[1]].to_numpy())
        self.NumericTerm = {'x': x, 'y': y}
    
    def Keep(self, Range = list):
        assert len(Range) == 2,'Range must contain 2 elements both integer'
        ''' 
        Keeping the first 'n' elements of a Numeric value dictionary
        '''
        self.NumericTerm['x'] = self.NumericTerm['x'][Range[0]:Range[1]]
        self.NumericTerm['y'] = self.NumericTerm['y'][Range[0]:Range[1]]

    # ---------- Data Manipulation ---------
    
    def Avg(self, Range:list):
        '''
        Simple Average in a given Range
        '''
        return np.average(self.NumericTerm['y'][Range[0]:Range[1]])

    def Std(self, Range:list):
        '''
        Simple Standard deviation
        '''
        return np.std(self.NumericTerm['y'][Range[0]:Range[1]])
    
    def Normalize(self, Amplitude = 1, SampleName = 'All'):
        '''
        Generalized Normalizer
        '''
        if SampleName == 'All':
            for Sample in self.Samples:
                Sample = np.int16((Sample/np.max(Sample))*Amplitude)
        else:
            self.Samples[f'{SampleName}'] = np.int16((self.Samples[f'{SampleName}']/np.max(self.Samples[f'{SampleName}']))*Amplitude)
    
    def AddNoise(self, Range, std, SampleName = 'All'):
        '''
        Generates Noise to  Data
        '''
        if SampleName == 'All':
            for Sample in self.Samples:

                Noise = gg.GaussianNoise(Range, Sample, std)
                Sample = Sample + Noise
        else:
            Noise = gg.GaussianNoise(Range,  self.Samples[f'{SampleName}'], std)
            self.Samples[f'{SampleName}'] =  self.Samples[f'{SampleName}'] + Noise

    # ---------- Status ----------    
    def Ready(self,Stateof):
        '''
        Finding if an instance has symbolic,numeric,numpy expressions
        '''
        assert Stateof in ["Num", "Numpy"],'The argument should be "Num", "Numpy"'
        PossibleTerms = ["Num","Numpy"]
        terms= {"Num": self.NumericTerm, "Numpy": self.NumpyTerm}
        for term in PossibleTerms:
            if term == Stateof:
                if terms[term]!= "":
                    return True
                else:
                    return False
            else:
                return "Not Found."
        
class BrownianSignal(func):
    
    def __init__(self, BrownianMethod: str, HurstExponent: float, TimeFrame: int , Imaginary = False, Numpy="", Numeric="", Symbols="", ):
        '''
        This is a class to generate Brownian Moiton Signals using deifferent methods
        BrownianMethod: ['Cholesky', 'DaviesHarte', 'Hosking']
        Hurst exponent: 0.5  Normal Brownian Motion
                        <0.5 Anti-Presistent time series
                        >0.5 Presistent time series
        Timefram : integer
        '''
        assert int(TimeFrame) == TimeFrame, "Time Frame should be an integer"
        assert HurstExponent <=1 and HurstExponent >=0, "Hurst Exponent should be between 0 and 1"
        assert BrownianMethod in ['Cholesky', 'DaviesHarte', 'Hosking'], "Brownianmethod should be in ['Cholesky', 'DaviesHarte', 'Hosking']"


        super().__init__(Numpy, Numeric, Symbols)
        self.Method = BrownianMethod
        self.Hexp = HurstExponent
        self.Timeframe = TimeFrame
        self.Imag = Imaginary

        if self.Method == "Cholesky":
            self.Cholesky()
        elif self.Method == "DaviesHarte":
            self.DaviesHarte()
        elif self.Method == "Hosking":
            self.Hosking()

    
    def Cholesky(self):
        y  = BrownianMotion.Cholesky(self.Timeframe, self.Timeframe, self.Hexp)
        x  = gg.LinearSampleMaker([0,self.Timeframe], self.Timeframe, 3)
        self.Numeric({'x': x, 'y':y})


    def DaviesHarte(self):
        y  = BrownianMotion.DaviesHarte(self.Timeframe - 1, self.Timeframe - 1, self.Hexp)
        if  self.Imag == False:
            y = np.real(y)

        x  = gg.LinearSampleMaker([0,self.Timeframe], self.Timeframe, 3)
        self.Numeric({'x': x, 'y':y})
    
    def Hosking(self):
        y  = BrownianMotion.Hosking(self.Timeframe, self.Timeframe, self.Hexp)
        if  self.Imag == False:
            y = np.real(y)
        x  = gg.LinearSampleMaker([0,self.Timeframe], self.Timeframe, 3)
        self.Numeric({'x': x, 'y':y})
