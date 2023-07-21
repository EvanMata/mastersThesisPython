"""
Python Dictionary for Phase retrieval in Python using functions defined in fth_reconstroction

2020
@authors:   RB: Riccardo Battistelli (riccardo.battistelli@helmholtz-berlin.de)
"""

import numpy as np
import fth_reconstruction as fth
#from skimage.draw import circle
import cupy as cp

#################
def PhaseRtrv(diffract,mask,mode='ER',Nit=500,beta_zero=0.5, beta_mode='const', Phase=0, bsmask=0,average_img=10, Fourier_last=True):
    
    '''
    Iterative phase retrieval function, with GPU acceleration
    INPUT:  diffract: far field hologram data
            mask: support matrix, defines where the retrieved reconstruction is supposed to be zeroa
            mode: string defining the algorithm to use (ER, RAAR, HIO, CHIO, OSS, HPR)
            Nit: number of steps
            beta_zero: starting value of beta
            beta_mode: way to evolve the beta parameter (const, arctan, exp, linear_to_beta_zero, linear_to_1)
            Phase: initial image to start from
            bsmask: binary matrix used to mask camera artifacts. where it is 1, the pixel will be left floating during the phase retrieval process
            average_img: number of images with a local minum Error on the diffraction pattern that will be summed to get the final image
            Fourier_last: Boolean, if true the magnitude constraint is going to be the lat one applied before returning the result. If False, the real space constraint is goint to be the last one
            
    OUTPUT:  complex retrieved diffraction pattern
    
     --------
    author: RB 2020
    '''

 
    #set parameters and BSmask
    (l,n) = diffract.shape
    alpha=None
    Error_diffr_list=[]

    
    #prepare beta function
    step=np.arange(Nit)
    if type(beta_mode)==np.ndarray:
        Beta=beta_mode
    elif beta_mode=='const':
        Beta=beta_zero*np.ones(Nit)
    elif beta_mode=='arctan':
        Beta=beta_zero+(0.5-np.arctan((step-np.minimum(Nit/2,700))/(0.15*Nit))/(np.pi))*(0.98-beta_zero)
    elif beta_mode=='smoothstep':
        start=Nit//50
        end=Nit-Nit//10
        x=np.array(range(Nit))
        y=(x-start)/(end-start)
        Beta=1-(1-beta_zero)*(6*y**5-15*y**4+10*y**3)
        Beta[:start]=1
        Beta[end:]=0
    elif beta_mode=='sigmoid':
        x=np.array(range(Nit))
        x0=Nit//20
        alpha=1/(Nit*0.15)
        Beta=1-(1-beta_zero)/(1+np.exp(-(x-x0)*alpha)) 
    elif beta_mode=='exp':
        Beta=beta_zero+(1-beta_zero)*(1-np.exp(-(step/7)**3))
    elif beta_mode=='linear_to_beta_zero':
        Beta= 1+(beta_zero-1)/Nit*step
    elif beta_mode=='linear_to_1':
        Beta= beta_zero+(1-beta_zero)/Nit*step

    guess = (1-bsmask)*diffract * np.exp(1j * np.angle(Phase))+ Phase*bsmask

  
    #previous result
    prev = None
    
    #shift everything to the corner
    bsmask=np.fft.fftshift(bsmask)
    guess=np.fft.fftshift(guess)
    mask=np.fft.fftshift(mask)
    diffract=np.fft.fftshift(diffract)
    
    BSmask_cp=cp.asarray(bsmask)
    guess_cp=cp.asarray(guess)
    mask_cp=cp.asarray(mask)
    diffract_cp=cp.asarray(diffract)
    
    first_instance_error=False
    
    #initialize prev
    prev=cp.fft.fft2( guess_cp*((1-BSmask_cp) *diffract_cp/np.abs(guess_cp) + BSmask_cp))
    
    
    for s in range(0,Nit):

        beta=Beta[s]
        
        #apply fourier domain constraints (only outside BS) and convert to REAL SPACE
        inv=cp.fft.fft2( guess_cp*((1-BSmask_cp) *diffract_cp/np.abs(guess_cp) + BSmask_cp))
            
        #support Projection
        if mode=='ER':
            inv*=mask_cp
        elif mode=='mine':
            inv += beta*(prev - 2*inv)*(1-mask_cp)
            
        prev=cp.copy(inv)
        
        ### GO TO FOURIER SPACE ### 
        guess_cp=cp.fft.ifft2(inv)
            
        #towards the end, let's create a list of the best reconstructions, minimizing Errore_diffr
        if s>=2 and s>= Nit-average_img*2:
            
            if first_instance_error==False:
                Best_guess=cp.zeros((average_img,l,n),dtype = 'complex_')
                Best_error=cp.zeros(average_img)
                first_instance_error=True
            
            #compute error to see if image has to end up among the best. Add in Best_guess array to sum them up at the end
            Error_diffr = Error_diffract_cp(cp.abs(guess_cp)*(1-BSmask_cp), diffract_cp*(1-BSmask_cp))
            Error_diffr_list.append(Error_diffr)
            
            if Error_diffr<=cp.amax(Best_error):
                j=cp.argmax(Best_error)
                if Error_diffr<Best_error[j]:
                    Best_error[j] = cp.copy(Error_diffr)
                    Best_guess[j,:,:]=cp.copy(guess_cp)


    #sum best guess images
    guess_cp=cp.sum(Best_guess,axis=0)/average_img
    
    
    #APPLY FOURIER COSTRAINT ONE LAST TIME (if Fourier_last==True)
    if Fourier_last:
        guess_cp = (1-BSmask_cp) *diffract_cp* cp.exp(1j * cp.angle(guess_cp)) + guess_cp*BSmask_cp
    
    guess=cp.asnumpy(guess_cp)

    #return final image
    return (np.fft.ifftshift(guess))


#############################################################
#    ERROR FUNCTIONS
#############################################################

def Error_diffract_cp(guess, diffract):
    '''
    Error on the diffraction attern of retrieved data. 
    INPUT:  guess, diffract: retrieved and experimental diffraction patterns 
            
    OUTPUT: Error between the two
    
    --------
    author: RB 2020
    '''
    Num=cp.abs(diffract-guess)**2
    Den=cp.abs(diffract)**2
    Error = Num.sum()/Den.sum()#cp.sum(Num)/cp.sum(Den)
    Error=10*cp.log10(Error)
    return Error
