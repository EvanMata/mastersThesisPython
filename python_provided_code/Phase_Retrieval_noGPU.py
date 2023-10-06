"""
Python Dictionary for Phase retrieval in Python using functions defined in fth_reconstroction

2021
@authors:   RB: Riccardo Battistelli (riccardo.battistelli@helmholtz-berlin.de)
"""

import numpy as np
import fth_reconstruction as fth
#from skimage.draw import circle

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
    author: RB 2021
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
    BSmask=np.fft.fftshift(bsmask)
    guess=np.fft.fftshift(guess)
    mask=np.fft.fftshift(mask)
    diffract=np.fft.fftshift(diffract)
    
    
    first_instance_error=False
    
    #initialize prev
    prev=np.fft.fft2( guess*((1-BSmask) *diffract/np.abs(guess) + BSmask))
    
    
    for s in range(0,Nit):

        beta=Beta[s]
        
        #apply fourier domain constraints (only outside BS) and convert to REAL SPACE
        inv=np.fft.fft2( guess*((1-BSmask) *diffract/np.abs(guess) + BSmask))
            
        #support Projection
        if mode=='ER':
            inv*=mask
        elif mode=='mine':
            inv += beta*(prev - 2*inv)*(1-mask)
            
        prev=np.copy(inv)
        
        ### GO TO FOURIER SPACE ### 
        guess=np.fft.ifft2(inv)
            
        #towards the end, let's create a list of the best reconstructions, minimizing Errore_diffr
        if s>=2 and s>= Nit-average_img*2:
            
            if first_instance_error==False:
                Best_guess=np.zeros((average_img,l,n),dtype = 'complex_')
                Best_error=np.zeros(average_img)
                first_instance_error=True
            
            #compute error to see if image has to end up among the best. Add in Best_guess array to sum them up at the end
            Error_diffr = Error_diffract(np.abs(guess)*(1-BSmask), diffract*(1-BSmask))
            Error_diffr_list.append(Error_diffr)
            
            if Error_diffr<=np.amax(Best_error):
                j=np.argmax(Best_error)
                if Error_diffr<Best_error[j]:
                    Best_error[j] = np.copy(Error_diffr)
                    Best_guess[j,:,:]=np.copy(guess)


    #sum best guess images
    guess=np.sum(Best_guess,axis=0)/average_img
    
    
    #APPLY FOURIER COSTRAINT ONE LAST TIME (if Fourier_last==True)
    if Fourier_last:
        guess = (1-BSmask) *diffract* np.exp(1j * np.angle(guess)) + guess*BSmask
    


    #return final image
    return (np.fft.ifftshift(guess))


#############################################################
#    ERROR FUNCTIONS
#############################################################

def Error_diffract(guess, diffract):
    '''
    Error on the diffraction attern of retrieved data. 
    INPUT:  guess, diffract: retrieved and experimental diffraction patterns 
            
    OUTPUT: Error between the two
    
    --------
    author: RB 2021
    '''
    Num=np.abs(diffract-guess)**2
    Den=np.abs(diffract)**2
    Error = Num.sum()/Den.sum()
    Error=10*np.log10(Error)
    return Error

from skimage.draw import circle
def FRC_GPU2(im1,im2,width_bin,center=0, start_Fourier=True):
    '''implements Fourier Ring Correlation. 
    RB June 2020 (https://www.nature.com/articles/s41467-019-11024-z)
    
    INPUT: 2 images in real space
            width of the bin, integer
    output: FRC histogram array
    
    RB 2020'''
    

    
    shape=im1.shape
    shape=(int(np.ceil(np.sqrt(2)*shape[0])),int(np.ceil(np.sqrt(2)*shape[0])))
    Num_bins=shape[0]//(2*width_bin)
    
    sum_num=np.zeros(Num_bins)
    sum_den=np.zeros(Num_bins)
    
    if type(center)==int:
        center = np.array([im1.shape[0]//2, im1.shape[1]//2])
    
    if start_Fourier:
        FT1=im1.copy()
        FT2=im2.copy()
    else:
        im1_cp=im1.copy()
        im2_cp=im2.copy()
        FT1=np.fft.fftshift(np.fft.fft2(np.fft.fftshift(im1_cp)))
        FT2=np.fft.fftshift(np.fft.fft2(np.fft.fftshift(im2_cp)))

    for i in range(Num_bins):
        yy_outer, xx_outer = circle((center[0], center[1]), (i+1)*width_bin)
        yy_inner, xx_inner = circle((center[0], center[1]), i*width_bin)
        if i==0:
            yy_inner, xx_inner = circle((center[0], center[1]), 1)

        outer=np.zeros((yy_outer.shape[0],2))
        outer[:,0]=xx_outer.copy()
        outer[:,1]=yy_outer.copy()

        in_delete=np.amax(outer, axis=1)>=(im1.shape[0])
        outer=np.delete(outer, in_delete, axis=0)
        in_delete=np.amin(outer, axis=1)<(0)
        outer=np.delete(outer, in_delete, axis=0)

        inner=np.zeros((yy_inner.shape[0],2))
        inner[:,0]=xx_inner.copy()
        inner[:,1]=yy_inner.copy()

        in_delete=np.amax(inner, axis=1)>=(im1.shape[0])
        inner=np.delete(inner, in_delete, axis=0)
        in_delete=np.amin(inner, axis=1)<(0)
        inner=np.delete(inner, in_delete, axis=0)

        inner=np.rint(inner).astype(int)
        outer=np.rint(outer).astype(int)
        #print(".",inner.size,outer.size)
        sum_num[i]=np.sum( (FT1* np.conj(FT2))[outer[:,1], outer[:,0]] ) - np.sum( (FT1* cp.conj(FT2))[inner[:,1], inner[:,0]] )
        sum_den[i]=np.sqrt( (np.sum(np.abs(FT1[outer[:,1], outer[:,0]])**2)-np.sum(cp.abs(FT1[inner[:,1], inner[:,0]])**2)) * (np.sum(np.abs(FT2[outer[:,1], outer[:,0]])**2)-np.sum(np.abs(FT2[inner[:,1], inner[:,0]])**2)) )
 

        #sum_num[i]=cp.sum( (FT1* cp.conj(FT2))[yy_outer, xx_outer] ) - cp.sum( (FT1* cp.conj(FT2))[yy_inner, xx_inner] )
        #sum_den[i]=cp.sqrt( (cp.sum(cp.abs(FT1[yy_outer, xx_outer])**2)-cp.sum(cp.abs(FT1[yy_inner, xx_inner])**2)) * (cp.sum(cp.abs(FT2[yy_outer, xx_outer])**2)-cp.sum(cp.abs(FT2[yy_inner, xx_inner])**2)) )
        
    FRC_array=sum_num/sum_den
    
    return FRC_array