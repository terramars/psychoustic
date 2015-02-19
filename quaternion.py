import numpy as np
from scipy import fftpack
from pyfft.cl import Plan
#import pyopencl as cl
#import pyopencl.array as cl_array
#from pyopencl import clmath
from scipy import sparse
import time

from pyfft.cuda import Plan
import pycuda.driver as cuda
from pycuda.tools import make_default_context
import pycuda.gpuarray as gpuarray
import pycuda.cumath as cumath

#ctx = cl.create_some_context(interactive=False)
#queue = cl.CommandQueue(ctx)

cuda.init()
ctx = make_default_context()
stream = cuda.Stream()
plan = None

posr = None
negr = None
posa = None
nega = None

index_color_matrix = None

def init_index_colors(colors):
    global index_color_matrix
    n = len(colors)
    index_color_matrix = np.zeros((n,n,4),dtype=np.float32)
    for i in range(n):
        ci = colors[i]
        for j in range(i,n):
            cj = colors[j]
            newcolor = np.array([0,0,0,0],dtype=np.float32)
            newcolor[0] = ci[0]*cj[0]-ci[1]*cj[1]-ci[2]*cj[2]-ci[3]*cj[3]
            newcolor[1] = ci[0]*cj[1]+ci[1]*cj[0]+ci[2]*cj[3]-ci[3]*cj[2]
            newcolor[2] = ci[0]*cj[2]+c2[2]*cj[0]+ci[3]*cj[1]-ci[1]*cj[3]
            newcolor[3] = ci[0]*cj[3]+c3[3]*cj[0]+ci[1]*cj[2]-ci[2]*cj[1]
            index_color_matrix[i,j,:]=newcolor
            index_color_matrix[j,i,:]=newcolor

def sparse_quaternion_convolution(im,tmp=None):
    global index_color_matrix
    if tmp:
        index_color_matrix = tmp
    print im.shape
    t0=time.time()
    sim = sparse.dok_matrix(im.sum(axis=2))
    print 'sim',time.time()-t0
    t0=time.time()
    m,n,_ = im.shape
    keys = sim.keys()
    r = [i[0] for i in keys]
    c = [i[1] for i in keys]
    print 'nkeys',len(r)
    I,K = np.meshgrid(r,r)
    J,L = np.meshgrid(c,c)
    print I.shape
    #C = index_color_matrix[im[I,J],im[K,L]]
    i1 = im[I,J]
    i2 = im[K,L]
    print 'grids',time.time()-t0
    t0=time.time()
    C = np.concatenate([i[:,:,np.newaxis] for i in qmult(0,i1[:,:,0],i1[:,:,1],i1[:,:,2],0,i2[:,:,0],i2[:,:,1],i2[:,:,2])],axis = 2)
    print 'C',time.time()-t0
    t0=time.time()
    rows = (I+K).flatten()
    cols = (J+L).flatten()
    out = np.zeros((m*2-1,n*2-1,4),dtype=np.float32)
    for i in range(im.shape[2]):
        out[:,:,i] = sparse.coo_matrix((C[:,:,i].flatten(),(rows,cols)),shape=out.shape[:2]).toarray()
    print 'done',time.time()-t0
    return out

def makeGaussian(size, fwhm = 3, center=None):
    """ Make a square gaussian kernel.

    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """
    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)

gauss = 1-makeGaussian(24,8)*.8

def gpu_fft(data,inverse=False):
    #global plan, ctx, queue
    global plan, ctx, stream ##cuda
    if not plan:
        print 'building plan',data.shape
        plan = Plan(data.shape, stream = stream, wait_for_finish = True)
        #plan = Plan(data.shape, queue=queue,wait_for_finish = True)
    #gpu_data = cl_array.to_device(ctx,queue,data.astype(np.complex64)) OLD
   
    #result = cl_array.zeros_like(data)
    result = gpuarray.zeros_like(data)

    plan.execute(data, data_out = result, inverse=inverse)
    
    #result = gpu_data.get() OLD

    return result

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
    global gauss
    if len(img.shape)>2:
        for i in range(img.shape[2]):
            img[img.shape[0]/2-12:img.shape[0]/2+12,img.shape[1]/2-12:img.shape[1]/2+12,i] *= gauss
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
    norm = np.sqrt(np.sqrt(np.sum(np.multiply(img[:,:,:3],img[:,:,:3]),axis=-1)))
    norm += 1e-10
    img[:,:,0] /= norm
    img[:,:,1] /= norm
    img[:,:,2] /= norm
    img[:,:,3] = np.sqrt(img[:,:,3])
    
    return normalize(img)

def normalize_gpu(rgb,a):
    arr = rgb.get()
    rgb -= arr.min()
    rgb /= arr.max()
    arr = a.get()
    a -= arr.min()
    a /= arr.max()
    rgb = rgb.get()
    a = a.get()
    img = np.zeros((a.shape[0],a.shape[1],4),dtype=np.float32)
    img[:,:,:3] = rgb
    img[:,:,3] = a
    return img

def sqrt_normalize_gpu(img):
    global posr,negr,posa,nega,stream
    #rgb = cl_array.to_device(queue,img[:,:,:3].copy())
    #a = cl_array.to_device(queue,img[:,:,3].copy())
    rgb = gpuarray.to_gpu(img[:,:,:3].copy())
    a = gpuarray.to_gpu(img[:,:,3].copy())

    if not posr:
        posr = gpuarray.zeros_like(rgb) + 1 
        negr = gpuarray.zeros_like(rgb) - 1
        posa = gpuarray.zeros_like(a) + 1
        nega = gpuarray.zeros_like(a) - 1
    rgb = cumath.sqrt(abs(rgb),stream=stream) * gpuarray.if_positive(rgb,posr,negr,stream=stream)
    a = cumath.sqrt(abs(a),stream=stream) * gpuarray.if_positive(a,posa,nega,stream=stream)
    return normalize_gpu(rgb,a)

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

def decompose_h_gpu(r,i,j,k):
    ha = gpuarray.zeros(i.shape,dtype=np.complex64)
    ha += r
    ha += i*1j
    hb = gpuarray.zeros(i.shape,dtype=np.complex64)
    hb += j
    hb += k*1j
    return ha,hb

def decompose_lum_gpu(r,i,j,k):
    ha = gpuarray.zeros(i.shape,dtype=np.complex64)
    ha += r
    lum = (i+j+k)/float(np.sqrt(3.0))
    ha += lum*1j
    hb = gpuarray.zeros(i.shape,dtype=np.complex64)
    cr1 = (j-k)/float(np.sqrt(2.0))
    cr2 = (-2*i+j+k)/float(2.0)
    hb += cr1
    hb += cr2*1j
    return ha,hb

def decompose_lum(r,i,j,k):
    ha = np.zeros(i.shape,dtype=np.complex64)
    ha += r
    lum = (i+j+k)/np.sqrt(3.0)
    ha += lum*1j
    hb = np.zeros(i.shape,dtype=np.complex64)
    cr1 = (j-k)/np.sqrt(2.0)
    cr2 = (-2*i+j+k)/2.0
    hb += cr1
    hb += cr2*1j
    return ha,hb

def recompose_lum(r,i,j,k):
    lum = i / np.sqrt(3)
    cr1 = j / np.sqrt(2)
    cr2 = k / 3
    j = lum + cr1 + cr2
    i = lum - 2 * cr2
    k = lum + cr2 - cr1
    return r,i,j,k

def recompose_lum_gpu(r,i,j,k):
    lum = i / np.sqrt(3)
    cr1 = j / np.sqrt(2)
    cr2 = k / 3
    j = lum + cr1 + cr2
    i = lum - 2 * cr2
    k = lum + cr2 - cr1
    return r.get(),i.get(),j.get(),k.get()

def qmult(r0,i0,j0,k0,r1,i1,j1,k1):
    r = np.multiply(r0,r1) - np.multiply(i0,i1) - np.multiply(j0,j1) - np.multiply(k0,k1)
    i = np.multiply(r0,i1) + np.multiply(i0,r1) + np.multiply(j0,k1) - np.multiply(k0,j1)
    j = np.multiply(r0,j1) + np.multiply(j0,r1) + np.multiply(k0,i1) - np.multiply(i0,k1)
    k = np.multiply(r0,k1) + np.multiply(k0,r1) + np.multiply(i0,j1) - np.multiply(j0,i1)
    return r,i,j,k

def qmult_gpu(r0,i0,j0,k0,r1,i1,j1,k1):
    r = r0*r1 - i0*i1 - j0*j1 - k0*k1
    i = r0*i1 + i0*r1 + j0*k1 - k0*j1
    j = r0*j1 + j0*r1 + k0*i1 - i0*k1
    k = r0*k1 + k0*r1 + i0*j1 - j0*i1
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
        ha,hb = decompose_lum_gpu(r,i,j,k)
    else:
        ha,hb = decompose_h_gpu(r,i,j,k)
    #fha = fftpack.fft2(ha)
    #fhb = fftpack.fft2(hb)
    fha = gpu_fft(ha)
    fhb = gpu_fft(hb)
    if inv:
        fha /= fha.size
        fha = reverse_gpu(fha)
        fhb /= fhb.size
        fhb = reverse_gpu(fhb)
    hr = fha.real
    hi = fha.imag
    hj = fhb.real
    hk = fhb.imag
    return hr,hi,hj,hk

def QCV2(fr,fi,fj,fk,hr,hi,hj,hk,mode=None):
    print 'ERROR : BROKEN FOR NOW'
    raise Exception
    r = cl_array.to_device(ctx,queue,r.astype(np.float32))
    i = cl_array.to_device(ctx,queue,i.astype(np.float32))
    j = cl_array.to_device(ctx,queue,j.astype(np.float32))
    k = cl_array.to_device(ctx,queue,k.astype(np.float32))
    fa,fb = decompose_lum_gpu(fr,fi,fj,fk)
    Hr,Hi,Hj,Hk = QFT2(hr,hi,hj,hk)
    fa = gpu_fft(fa)
    fb = gpu_fft(fb)
    fhar,fhai,fhaj,fhak = qmult_gpu(fa.real,fa.imag,0,0,Hr,Hi,Hj,Hk)
    fhbr,fhbi,fhbj,fhbk = qmult_gpu(0,0,fb.real,fb.imag,Hr[::-1],Hi[::-1],Hj[::-1],Hk[::-1])
    r,i,j,k = QFT2(fhar+fhbr,fhai+fhbi,fhaj+fhbj,fhak+fhbk,inv=1,lum=0)
    return recompose_lum_gpu(r,i,j,k)

def reverse_gpu(arr):
    return gpuarray.to_gpu(arr.get()[::-1].copy())

def AQCV2(r,i,j,k,mode=None):
    #t0=time.time()
    if r == 0:
        r = np.zeros(i.shape)
    r = gpuarray.to_gpu(r.astype(np.complex64))
    i = gpuarray.to_gpu(i.astype(np.complex64))
    j = gpuarray.to_gpu(j.astype(np.complex64))
    k = gpuarray.to_gpu(k.astype(np.complex64))
    fa,fb = decompose_lum_gpu(r,i,j,k)
    #print 'load and decompose',time.time()-t0
    #ffa = fftpack.fft2(fa)
    #ffb = fftpack.fft2(fb)
    #t0=time.time()
    ffa = gpu_fft(fa)
    ffb = gpu_fft(fb)
    #print 'first ffts',time.time()-t0
    #t0=time.time()
    ffar,ffai,ffaj,ffak = qmult_gpu(ffa.real,ffa.imag,0,0,ffa.real,ffa.imag,ffb.real,ffb.imag)
    ffbr,ffbi,ffbj,ffbk = qmult_gpu(0,0,ffb.real,ffb.imag,reverse_gpu(ffa.real),reverse_gpu(ffa.imag),reverse_gpu(ffb.real),reverse_gpu(ffb.imag))
    #print 'qmult',time.time()-t0
    #t0=time.time()
    r,i,j,k = QFT2(ffar+ffbr,ffai+ffbi,ffaj+ffbj,ffak+ffbk,inv=1,lum=0)
    #print 'qft2',time.time()-t0
    #t0=time.time()
    out = recompose_lum_gpu(r,i,j,k)
    #print 'recompose',time.time()-t0
    return out

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
   
