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

def pad(im):
    m,n = im.shape[:2]
    if len(im.shape) == 2:
        newim = np.zeros((m*2,n*2),dtype=im.dtype)
        newim[m/2:m/2+m,n/2:n/2+n] = im
    else:
        newim = np.zeros((m*2,n*2,im.shape[2]),dtype=im.dtype)
        newim[m/2:m/2+m,n/2:n/2+n,:] = im
    return newim

def normalize(img):
    if len(img.shape)>2:
        img[:,:,:3] -= img[:,:,:3].min()
        img[:,:,:3] /= img[:,:,:3].max()
        if img.shape[2] == 4:
            img[:,:,3] -= img[:,:,3].min()
            img[:,:,3] /= img[:,:,3].max()
    else:
        img -= img.min()
        img /= img.max()
    return img

def tanh_normalize(img):
    img = np.multiply(np.sqrt(np.abs(np.tanh(img/img.std()))),np.sign(img))
    return normalize(img)

def log_normalize(img):
    img = np.multiply(np.log(1+np.abs(img)),np.sign(img))
    return normalize(img)

def sqrt_normalize(img):
    img = np.multiply(np.sqrt(np.abs(img)),np.sign(img))
    return normalize(img)

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

def decompose_lum(r,i,j,k):
    ha = np.zeros(i.shape,dtype=np.complex64)
    ha.real = r
    lum = (i+j+k)/np.sqrt(3.0)
    ha.imag = lum
    hb = np.zeros(i.shape,dtype=np.complex64)
    cr1 = (j-k)/np.sqrt(2.0)
    cr2 = (-2*i+j+k)/2.0
    hb.real = cr1
    hb.imag = cr2
    return ha,hb

def recompose_lum(r,i,j,k):
    lum = i / np.sqrt(3)
    cr1 = j / np.sqrt(2)
    cr2 = k / 3
    j = lum + cr1 + cr2
    i = lum - 2 * cr2
    k = lum + cr2 - cr1
    return r,i,j,k

def qmult(r0,i0,j0,k0,r1,i1,j1,k1):
    r = np.multiply(r0,r1) - np.multiply(i0,i1) - np.multiply(j0,j1) - np.multiply(k0,k1)
    i = np.multiply(r0,i1) + np.multiply(i0,r1) + np.multiply(j0,k1) - np.multiply(k0,j1)
    j = np.multiply(r0,j1) + np.multiply(j0,r1) + np.multiply(k0,i1) - np.multiply(i0,k1)
    k = np.multiply(r0,k1) + np.multiply(k0,r1) + np.multiply(i0,j1) - np.multiply(j0,i1)
    return r,i,j,k

def QFT1(r,i,j,k,inv=0):
    ha,hb = decompose_h(r,i,j,k)
    fha = fftpack.fft2(ha)
    fhb = fftpack.fft2(hb[:,::-1])
    if inv:
        fha /= fha.size
        fha = fha[::-1]
        fhb /= fhb.size
        fhb = fhb[::-1]
    hcr = fha.real
    hci = fha.imag
    hcj = fhb.real
    hck = fhb.imag
    hq1r = (hcr + hcr[:,::-1]+hck-hck[:,::-1])/2
    hq1i = (hci - hcj + hci[:,::-1] + hcj[:,::-1])/2
    hq1j = (hcj + hci + hcj[::-1] - hci[:,::-1])/2
    hq1k = (hck - hcr + hck[:,::-1] + hcr[:,::-1])/2
    return hq1r,hq1i,hq1j,hq1k

def QFT3(r,i,j,k,inv=0):
    ha,hb = decompose_lum(r,i,j,k)
    fha = fftpack.fft2(ha)
    fhb = fftpack.ifft2(hb)
    if inv:
        fha /= fha.size
        fha = fha[::-1]
        fhb /= fhb.size
        fhb = fhb[::-1]
    hr = fha.real
    hi = fha.imag
    hj = fhb.real
    hk = fhb.imag
    return hr,hi,hj,hk

def SPQCV(fr,fi,fj,fk,hr,hi,hj,hk,mode=1):
    QFTfunc=QFT1
    if mode == 2:
        QFTfunc = QFT2
    if mode == 3:
        QFTfunc = QFT3
    fr,fi,fj,fk = QFTfunc(fr,fi,fj,fk)
    hr,hi,hj,hk = QFTfunc(hr,hi,hj,hk)
    fhr,fhi,fhj,fhk = qmult(fr,fi,fj,fk,hr,hi,hj,hk)
    return QFTfunc(fhr,fhi,fhj,fhk,inv=1)

def QFT2(r,i,j,k,inv=0,lum=1):
    if lum:
        ha,hb = decompose_lum(r,i,j,k)
    else:
        ha,hb = decompose_h(r,i,j,k)
    fha = fftpack.fft2(ha)
    fhb = fftpack.fft2(hb)
    if inv:
        fha /= fha.size
        fha = fha[::-1]
        fhb /= fhb.size
        fhb = fhb[::-1]
    hr = fha.real
    hi = fha.imag
    hj = fhb.real
    hk = fhb.imag
    return hr,hi,hj,hk

def QCV2(fr,fi,fj,fk,hr,hi,hj,hk,mode=None):
    fa,fb = decompose_lum(fr,fi,fj,fk)
    Hr,Hi,Hj,Hk = QFT2(hr,hi,hj,hk)
    fa = fftpack.fft2(fa)
    fb = fftpack.fft2(fb)
    fhar,fhai,fhaj,fhak = qmult(fa.real,fa.imag,0,0,Hr,Hi,Hj,Hk)
    fhbr,fhbi,fhbj,fhbk = qmult(0,0,fb.real,fb.imag,Hr[::-1],Hi[::-1],Hj[::-1],Hk[::-1])
    r,i,j,k = QFT2(fhar+fhbr,fhai+fhbi,fhaj+fhbj,fhak+fhbk,inv=1,lum=0)
    return recompose_lum(r,i,j,k)

def AQCV2(r,i,j,k,mode=None):
    fa,fb = decompose_lum(r,i,j,k)
    ffa = fftpack.fft2(fa)
    ffb = fftpack.fft2(fb)
    ffar,ffai,ffaj,ffak = qmult(ffa.real,ffa.imag,0,0,ffa.real,ffa.imag,ffb.real,ffb.imag)
    ffbr,ffbi,ffbj,ffbk = qmult(0,0,ffb.real,ffb.imag,ffa.real[::-1],ffa.imag[::-1],ffb.real[::-1],ffb.imag[::-1])
    r,i,j,k = QFT2(ffar+ffbr,ffai+ffbi,ffaj+ffbj,ffak+ffbk,inv=1,lum=0)
    return recompose_lum(r,i,j,k)

def QCV3(fr,fi,fj,fk,hr,hi,hj,hk,mode=None):
    ha,hd = decompose_h(hr,hi,hj,hk)
    hd.imag *= -1
    fr,fi,fj,fk = QFT3(fr,fi,fj,fk)
    ha = fftpack.fft2(ha)
    hd = fftpack.fft2(hd)
    fhar,fhai,fhaj,fhak = qmult(fr,fi,fj,fk,ha.real,ha.imag,0,0)
    fhdr,fhdi,fhdj,fhdk = qmult(fr[::-1],fi[::-1],fj[::-1],fk[::-1],0,0,hd.real,-1*hd.imag)
    return QFT3(fhar+fhdr,fhai+fhdi,fhaj+fhdj,fhak+fhdk,inv=1)

def QCR3(fr,fi,fj,fk,hr,hi,hj,hk,mode=None):
    return QCV3(fr,fi,fj,fk,hr[::-1],-1*hi[::-1],-1*hj[::-1],-1*hk[::-1])

def QCV1(fr,fi,fj,fk,hr,hi,hj,hk,mode=None):
    far,fai,faj,fak = QFT1(fr,fi,0,0)
    fbr,fbi,fbj,fbk = QFT1(fj,fk,0,0)
    hr,hi,hj,hk = QFT1(hr,hi,hj,hk)
    fh1r,fh1i,fh1j,fh1k = qmult(far,fai,faj,fak,hr,0,hj,0)
    fh2r,fh2i,fh2j,fh2k = qmult(far[:,::-1],fai[:,::-1],faj[:,::-1],fak[:,::-1],0,hi,0,hk)
    fbr,fbi,fbj,fbk = qmult(fbr,fbi,fbj,fbk,0,0,1,0)
    fh3r,fh3i,fh3j,fh3k = qmult(fbr[:,::-1],fbi[:,::-1],fbj[:,::-1],fbk[:,::-1],hr[::-1,:],0,hj[::-1,:],0)
    fh4r,fh4i,fh4j,fh4k = qmult(fbr,fbi,fbj,fbk,0,hi[::-1,:],0,hk[::-1,:])
    return QFT1(fh1r+fh2r+fh3r+fh4r,fh1i+fh2i+fh3i+fh4i,fh1j+fh2j+fh3j+fh4j,fh1k+fh2k+fh3k+fh4k,inv=1)
   


