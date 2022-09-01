import numpy as np
import scipy as sc



class GeneralGenerators():
    @staticmethod
    def GaussianNoise(Range: list, SampleVar, std=0):
        '''
        Generates a Gaussian noise distribution.
        '''
        NoiseSample = np.random.normal(loc=(Range[1] - Range[0])/2, scale=std, 
                                                  size=len(SampleVar))
        return NoiseSample
    
    @staticmethod
    def LinearSampleMaker(Range, SampleNumber, Decimalpoint):
        '''
        Given a list for range and a SampleNumber this method generates a linear sample
        '''
        assert len(Range)==2, 'you should give a list of two numbers for Range of your Sample'
        assert SampleNumber//1 == SampleNumber,"SampleNumber should be an integer"
        Sample = np.linspace(Range[0], Range[1], SampleNumber)
        if Decimalpoint == 0:
            Sample = np.int_(Sample)
        else:
            Sample = np.round(Sample,Decimalpoint)

        return Sample
    
    @staticmethod
    def LogSampleMaker(Range, SampleNumber):
        '''
        Given a list for range and a SampleNumber this method generates a logarithmic sample
        '''
        assert len(Range)==2, 'you should give a list of two numbers for Range of your Sample'
        assert SampleNumber//1 == SampleNumber,"SampleNumber should be an integer"
        Sample = np.logspace(Range[0], Range[1], SampleNumber)
        return Sample

class BrownianMotion ():
    '''
    Class of methods to produce a brownian motion.
    '''
    @staticmethod
    def SimpleBrownianMotion(Time, SampleNumber):
        # Not yet made...
        pass


    @staticmethod
    def Cholesky(T, N, H):
        '''
        Generates sample paths of fractional Brownian Motion using the Davies Harte method
        args:
            T:      length of time (in years)
            N:      number of time steps within timeframe
            H:      Hurst parameter
        Created on Sat Aug 15 18:46:02 2020
        @author: Justin Yu
        Implementation of Fractional Brownian Motion, Cholesky's Method
        '''
        gamma = lambda k,H: 0.5*(np.abs(k-1)**(2*H) - 2*np.abs(k)**(2*H) + np.abs(k+1)**(2*H))  
        
        L = np.zeros((N,N))
        X = np.zeros(N)
        V = np.random.standard_normal(size=N)

        L[0,0] = 1.0
        X[0] = V[0]
        
        L[1,0] = gamma(1,H)
        L[1,1] = np.sqrt(1 - (L[1,0]**2))
        X[1] = np.sum(L[1,0:2] @ V[0:2])
        
        for i in range(2,N):
            L[i,0] = gamma(i,H)
            
            for j in range(1, i):         
                L[i,j] = (1/L[j,j])*(gamma(i-j,H) - (L[i,0:j] @ L[j,0:j]))

            L[i,i] = np.sqrt(1 - np.sum((L[i,0:i]**2))) 
            X[i] = L[i,0:i+1] @ V[0:i+1]

        fBm = np.cumsum(X)*(N**(-H))
        return (T**H)*(fBm)

    @staticmethod
    def DaviesHarte(T, N, H):
        '''
        Generates sample paths of fractional Brownian Motion using the Davies Harte method
        
        args:
            T:      length of time (in years)
            N:      number of time steps within timeframe
            H:      Hurst parameter
        '''
        gamma = lambda k,H: 0.5*(np.abs(k-1)**(2*H) - 2*np.abs(k)**(2*H) + np.abs(k+1)**(2*H))  
        g = [gamma(k,H) for k in range(0,N)];    r = g + [0] + g[::-1][0:N-1]

        # Step 1 (eigenvalues)
        j = np.arange(0,2*N);   k = 2*N-1
        lk = np.fft.fft(r*np.exp(2*np.pi*complex(0,1)*k*j*(1/(2*N))))[::-1]

        # Step 2 (get random variables)
        Vj = np.zeros((2*N,2), dtype=np.complex); 
        Vj[0,0] = np.random.standard_normal();  Vj[N,0] = np.random.standard_normal()
        
        for i in range(1,N):
            Vj1 = np.random.standard_normal();    Vj2 = np.random.standard_normal()
            Vj[i][0] = Vj1; Vj[i][1] = Vj2; Vj[2*N-i][0] = Vj1;    Vj[2*N-i][1] = Vj2
        
        # Step 3 (compute Z)
        wk = np.zeros(2*N, dtype=np.complex)   
        wk[0] = np.sqrt((lk[0]/(2*N)))*Vj[0][0];          
        wk[1:N] = np.sqrt(lk[1:N]/(4*N))*((Vj[1:N].T[0]) + (complex(0,1)*Vj[1:N].T[1]))       
        wk[N] = np.sqrt((lk[0]/(2*N)))*Vj[N][0]       
        wk[N+1:2*N] = np.sqrt(lk[N+1:2*N]/(4*N))*(np.flip(Vj[1:N].T[0]) - (complex(0,1)*np.flip(Vj[1:N].T[1])))
        
        Z = np.fft.fft(wk);     fGn = Z[0:N] 
        fBm = np.cumsum(fGn)*(N**(-H))
        fBm = (T**H)*(fBm)
        path = np.array([0] + list(fBm))
        return path
    
    @staticmethod
    def Hosking(T, N, H):
        '''
        Generates sample paths of fractional Brownian Motion using the Davies Harte method
        
        args:
            T:      length of time (in years)
            N:      number of time steps within timeframe
            H:      Hurst parameter
        '''
        gamma = lambda k,H: 0.5*(np.abs(k-1)**(2*H) - 2*np.abs(k)**(2*H) + np.abs(k+1)**(2*H))  
        
        X = [np.random.standard_normal()]
        mu = [gamma(1,H)*X[0]]
        sigsq = [1 - (gamma(1,H)**2)]
        tau = [gamma(1,H)**2]
        
        d = np.array([gamma(1,H)])
        
        for n in range(1, N):
            
            F = np.rot90(np.identity(n+1))
            c = np.array([gamma(k+1,H) for k in range(0,n+1)])
                    
            # sigma(n+1)**2
            s = sigsq[n-1] - ((gamma(n+1,H) - tau[n-1])**2)/sigsq[n-1]
            
            # d(n+1)
            phi = (gamma(n+1,H) - tau[n-1])/sigsq[n-1]
            d = d - phi*d[::-1]
            d = np.append(d, phi)        
            
            # mu(n+1) and tau(n+1)
            Xn1 = mu[n-1] + sigsq[n-1]*np.random.standard_normal()
            
            X.append(Xn1)
            sigsq.append(s)
            mu.append(d @ X[::-1])
            tau.append(c @ F @ d)
        
        fBm = np.cumsum(X)*(N**(-H))    
        return (T**H)*fBm



    

