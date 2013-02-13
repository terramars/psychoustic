import numpy as np
from numpy import pi
from PIL import Image, ImageDraw
from spectrum import FourierSpectrum
from scikits import audiolab
import shutil
import os
from scipy.misc import imsave
from scipy.signal import fftconvolve, resample, convolve2d, gaussian
from scipy.fftpack import fftn,ifftn,fftshift
from helper import *

def draw_points(xs,ys,colors,draw):
    for i in range(0,len(xs)):
        draw.point((ys[i],xs[i]),fill=colors[i])
        #draw.line(((ys[i-1],xs[i-1]),(ys[i],xs[i])),fill=colors[i-1])

def project_to_circle(data,rads):
    xs=np.array([np.cos(rads[i])*(data[i]) for i in range(len(data))])
    ys=np.array([np.sin(rads[i])*(data[i]) for i in range(len(data))])
    return xs,ys

def project_weird(data,rads,r1=1,r2=1.01):
    r0=100
    xs=np.array([data[i]*(r0*np.cos(rads[i])+r1/(r2+np.cos(rads[i]))) for i in range(len(data))])
    ys=np.array([data[i]*(r0*np.sin(rads[i])+r1/(r2+np.sin(rads[i]))) for i in range(len(data))])
    return xs,ys

def center_circle(xs,ys,shape):
    #xs/=max(100,np.abs(xs).max())
    #ys/=max(100,np.abs(ys).max())
    radius=min(shape)/2
    centerx=shape[0]/2
    centery=shape[1]/2
    xs*=radius
    ys*=radius
    xs+=centerx
    ys+=centery
    return xs,ys

def get_radians(size,final_angle=pi):
    rads=np.array(range(size))*1.0
    rads=np.log2(rads+1.0)
    rads/=rads.max()
    #rads=rads**2
    rads*=final_angle
    return rads

def get_radians_with_spectrum(psd,final_angle=pi):
    rads=np.array(range(psd.shape[0]))*1.0
    rads=np.log2(rads+1.0)**2
    offset=np.zeros(len(rads)-1)
    window_size=2
    for i in range(psd.shape[0]-1):
        if i+window_size<psd.shape[0]:
            offset[i]=psd[i:i+window_size].sum()
        else:
            offset[i]=psd[psd.shape[0]-window_size:].sum()
        window_size+=1
#    offset+=0.001
    offset/=(offset).max()+0.000001
    #offset=offset**2
    diffs=rads[:-1]-rads[1:]
    #print offset[:10]
    diffs=diffs*offset
    rads[1:]=rads[:-1]+diffs
    rads/=rads.max()
    #rads=rads**1.5
    rads*=final_angle
    return rads

def normalize_spectrum(spectrum,power,offset=-20,inv=1,scales=(24.0,8.0)):
    alt_offset=10*np.log10(power)-15
    if alt_offset>offset:
        offset=alt_offset
    #print power,alt_offset
    spectrum_plus=np.choose(spectrum>offset,(0,spectrum-offset))
    spectrum_minus=np.choose(spectrum<offset,(0,spectrum-offset))
    data=np.ones(spectrum.shape)+8
    if inv:
        data-=scales[0]*(1-1/np.log10(spectrum_plus+10))
        spectrum_minus*=-1
        data+=scales[1]*(1-1/np.log10(spectrum_minus+10))
    else:
        data+=scales[0]*(1-1/np.log10(spectrum_plus+10))
        spectrum_minus*=-1
        data-=scales[1]*(1-1/np.log10(spectrum_minus+10))
    data/=max(data.max(),10.0)
    return data

def data_to_circle(data,psd,sym=2,overlap=1):
    final_angle=2*pi/sym
    n=len(data)
    #if n%3:
    #    data=data[:-1*(n%3)]
    #    psd=psd[:-1*(n%3)]
    #    n-=n%3
    #rads=get_radians_with_spectrum(psd,final_angle)
    if not overlap:
        rads=get_radians(n,final_angle)
    else:
        rads=get_radians(n,final_angle*np.log2(n))
        rads=rads%final_angle
    offset=0
    direction=0
    #rad_low=get_radians(n/3,final_angle/2)
    #rad_hi=get_radians(2*n/3,final_angle/2)+final_angle/2
    #rads[:n/3]=rad_low
    #rads[n/3:]=rad_hi
    present_angles=rads.copy()
    finalx=np.zeros(n*sym)
    finaly=np.zeros(finalx.shape)
    #psdsum=psd.sum()
    #if psdsum<1.0:
    #    psdsum=1e10
    #else:
    #    psdsum=1+0.01/(np.log10(psdsum))+0.01
    for i in range(sym):
        xs,ys=project_to_circle(data,present_angles)
        #xs,ys=project_weird(data,present_angles,r2=psdsum)
        finalx[i*n:(i+1)*n]=xs
        finaly[i*n:(i+1)*n]=ys
        direction+=1
        offset+=final_angle
        if direction%2:
            present_angles=offset+final_angle-rads
        else:
            present_angles=rads+offset
        #rads=rads[::-1]
    return finalx, finaly

def draw_spectrum(psd,spectrum,shape,name,sym=2,inv=1):
    data=normalize_spectrum(spectrum,psd,inv=inv)
    xs,ys=data_to_circle(data,psd,sym=sym)
    xs,ys=center_circle(xs,ys,shape)
    im=Image.new('L',shape)
    draw=ImageDraw.Draw(im)
    colors=np.ones(xs.shape)*255
    nsemi=xs.shape[0]/sym
    for i in range(sym):
        draw_points(xs[i*nsemi:(i+1)*nsemi],ys[i*nsemi:(i+1)*nsemi],colors[i*nsemi:(i+1)*nsemi],draw)
    #im.save(name)
    return im

def edge_filter(im):
    filt=np.ones((3,3))
    filt[1,1]=0
    im=np.array(im)
    edge=convolve2d(np.choose(im>0,(0,1)),filt,'same')
    im=np.choose(edge==8,(im,0)).astype(np.float32)/255.0
    return im

def convolve_image(im1,im2,name,rot=0,filt=1):
    im1=edge_filter(im1)
    im2=edge_filter(im2)
    im=fftconvolve(im1,im2)
    im=np.choose(im>0,(0,im))
    im+=1.0
    im=np.log(im)**3.5
    im*=255.0/im.max()
    imsave(name,im.astype(np.uint8)[:-1,:-1])
    return im

def get_audio(fin):
    signal,fs,enc=audiolab.wavread(fin)
    return signal,fs

def get_spectrum(signal,i,fs,NFFT,channel=1):
    mult=2
    psd = np.zeros((NFFT*mult/2,channel))
    s = FourierSpectrum(signal[i*NFFT/2:i*NFFT/2+NFFT,0],sampling=fs,NFFT=NFFT*mult,detrend='mean')
    s.periodogram()
    psd[:,0]=s.psd[1:psd.shape[0]+1]
    if channel>1: 
        s = FourierSpectrum(signal[i*NFFT/2:i*NFFT/2+NFFT,1],sampling=fs,NFFT=NFFT*mult,detrend='mean')
        s.periodogram()
        psd[:,1]=s.psd[1:psd.shape[0]+1]
    return psd,s.power()
    #if signal.shape[1]==2:
    #    return (FourierSpectrum(signal[i*NFFT/2:i*NFFT/2+NFFT,0],sampling=fs,NFFT=NFFT,detrend='mean'),FourierSpectrum(signal[i*NFFT/2:i*NFFT/2+NFFT,1],sampling=fs,NFFT=NFFT,detrend='mean'))
    #else:
    #    return FourierSpectrum(signal[i*NFFT/2:i*NFFT/2+NFFT,0],sampling=fs,NFFT=NFFT,detrend='mean')

def get_spectrum_slices(signal,fs,framerate):
    ffts=[]
    NFFT=int(fs/framerate)
    signal=signal[:(signal.shape[0]*2/NFFT)*NFFT/2,:]
    for i in range(signal.shape[0]/NFFT*2):
        ffts.append(get_spectrum(signal,i,fs,NFFT)[0])
    return ffts

def file_to_images(fin,outdir,filetype='png',shape=(640,640),framerate=25,sym=2,rot=0,inv=1):
    framerate/=2.0
    import time
    signal,fs=get_audio(fin)
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    print 'got audio'
    NFFT=int(fs/framerate)
    signal=signal[:(signal.shape[0]*2/NFFT)*NFFT/2,:]
    #ffts=get_spectrum_slices(signal,fs,framerate)
    print signal.shape[0]*2/NFFT,' spectrum slices'
    gauss=gaussian(21,2)
    gauss/=gauss.sum()
    t0=time.time()
    t1=0
    t2=0
    t3=time.time()
    for i in range(signal.shape[0]/NFFT*2):
        if os.path.isfile('%sconv%04d.%s'%(outdir,i,filetype)):
            continue
        psd=get_spectrum(signal,i,fs,NFFT)[0]
        psd[:,0]=np.convolve(psd[:,0],gauss,'same')
        psd[:,1]=np.convolve(psd[:,1],gauss,'same')
        spectrum=10*np.log10(psd+1e-20)
        if signal.shape[1]==2:
            #spectrum = 10*np.log10(np.convolve(s[0].psd,gauss,'valid'))
            im1=draw_spectrum(psd[:,0],spectrum[:,0],shape,name='%s%04d_1.%s'%(outdir,i,filetype),sym=sym,inv=inv)
            im2=draw_spectrum(psd[:,1],spectrum[:,1],shape,name='%s%04d_2.%s'%(outdir,i,filetype),sym=sym,inv=inv)
            t1+=time.time()-t3
            t3=time.time()
            convolve_image(im1,im2,name='%sconv%04d.%s'%(outdir,i,filetype),rot=rot)
            t2+=time.time()-t3
            t3=time.time()
        else:
            #spectrum = 10*np.log10(np.convolve(s.psd,gauss,'valid'))
            im=draw_spectrum(spectrum,shape,name='%s%04d_1.%s'%(outdir,i,filetype),sym=sym,inv=inv)
            convolve_image(im,im,name='%sconv%04d.%s'%(outdir,i,filetype),rot=rot)
        if i%100==0:
            print i,time.time()-t0,t1,t2
    return

if __name__=='__main__':
    import sys
    fin=sys.argv[1]
    outdir=sys.argv[2]
    shape=int(sys.argv[3])
    sym=int(sys.argv[4])
    inv=int(sys.argv[5])
    file_to_images(fin,outdir,shape=(shape,shape),sym=sym,inv=inv)
