import numpy as np
from scipy import fftpack

def fftshift_color(img):
    fftshift = fftpack.fftshift
    if len(img.shape)!=3:
        return fftshift(img)
    else:
        for i in range(img.shape[2]):
            img[:,:,i] = fftshift(img[:,:,i])
    return img

def create_image(r,i,j,k):
    img = np.zeros((r.shape[0],r.shape[1],4))
    img[:,:,0] = i
    img[:,:,1] = j
    img[:,:,2] = k
    img[:,:,3] = np.abs(r)
    img = fftshift_color(img)
    return img

def normalize(img):
    img -= img.min()
    img /= img.max()
    return img

def log_normalize(img):
    img = np.multiply(np.log(1+np.abs(img)),np.sign(img))
    img[:,:,:3] = normalize(img[:,:,:3])
    img[:,:,3] = normalize(img[:,:,3])
    return img

def decompose_h(r,i,j,k):
    if not type(r)==np.ndarray:
        r=np.zeros(i.shape,dtype=np.float32)
    ha = np.zeros(i.shape,dtype=np.complex64)
    ha.real += r
    ha.imag += i
    hb = np.zeros(i.shape,dtype=np.complex64)
    hb.real += j
    hb.imag += k
    return ha,hb

def qmult(r0,i0,j0,k0,r1,i1,j1,k1):
    r = np.multiply(r0,r1) - np.multiply(i0,i1) - np.multiply(j0,j1) - np.multiply(k0,k1)
    i = np.multiply(r0,i1) + np.multiply(i0,r1) + np.multiply(j0,k1) - np.multiply(k0,j1)
    j = np.multiply(r0,j1) + np.multiply(j0,r1) + np.multiply(k0,i1) - np.multiply(i0,k1)
    k = np.multiply(r0,k1) + np.multiply(k0,r1) + np.multiply(i0,j1) - np.multiply(j0,i1)
    return r,i,j,k

def QFT1(r,i,j,k,inv=0):
    ha,hb = decompose_h(r,i,j,k)
    if not inv:
        fha = fftpack.fft2(ha)
        fhb = fftpack.fft2(hb[:,::-1])
    else:
        fha = fftpack.ifft2(ha)
        fhb = fftpack.ifft2(hb[:,::-1])
    hcr = fha.real
    hci = fha.imag
    hcj = fhb.real
    hck = fhb.imag
    hq1r = (hcr + hcr[:,::-1]+hck-hck[:,::-1])/2
    hq1i = (hci - hcj + hci[:,::-1] + hcj[:,::-1])/2
    hq1j = (hcj + hci + hcj[::-1] - hci[:,::-1])/2
    hq1k = (hck - hcr + hck[:,::-1] + hcr[:,::-1])/2
    return hq1r,hq1i,hq1j,hq1k

def QFT2(r,i,j,k,inv=0):
    ha,hb = decompose_h(r,i,j,k)
    if not inv:
        fha = fftpack.fft2(ha)
        fhb = fftpack.fft2(hb)
    else:
        fha = fftpack.ifft2(ha)
        fhb = fftpack.ifft2(hb)
    hr = fha.real
    hi = fha.imag
    hj = fhb.real
    hk = fhb.imag
    return hr,hi,hj,hk

def QFT3(r,i,j,k,inv=0):
    ha,hb = decompose_h(r,i,j,k)
    if not inv:
        fha = fftpack.fft2(ha)
        fhb = fftpack.ifft2(hb)
    else:
        fha = fftpack.ifft2(ha)
        fhb = fftpack.fft2(hb)
    hr = fha.real
    hi = fha.imag
    hj = fhb.real
    hk = fhb.imag
    return hr,hi,hj,hk

def QCV2(fr,fi,fj,fk,hr,hi,hj,hk):
    fa,fb = decompose_h(fr,fi,fj,fk)
    Hr,Hi,Hj,Hk = QFT2(hr,hi,hj,hk)
    fa = fftpack.fft2(fa)
    fb = fftpack.fft2(fb)
    fhar,fhai,fhaj,fhak = qmult(fa.real,fa.imag,0,0,Hr,Hi,Hj,Hk)
    fhbr,fhbi,fhbj,fhbk = qmult(0,0,fb.real,fb.imag,Hr[::-1],Hi[::-1],Hj[::-1],Hk[::-1])
    return QFT2(fhar+fhbr,fhai+fhbi,fhaj+fhbj,fhak+fhbk,inv=1)
