from dear import io
from dear.spectrum import dft, cqt, auditory
from matplotlib import cm
import numpy as np
from PIL import ImageDraw, Image
import quaternion
from scipy import signal
from scipy.misc import imsave
import os
import time
from iso226 import *

iso = iso226_spl_itpl(1,True)

colormap = [(np.array(cm.hsv(i)[:3])) for i in range(256)]
colormap = [(255 * i / np.sqrt(np.dot(i,i))).astype(np.uint8) for i in colormap]

def init_palette(n = 64):
    i = np.arange(0,256)
    div = np.linspace(0,255,n+1)[1]
    quant = np.int0(np.linspace(0,255,n))
    color_levels = np.clip(np.int0(i/div),0,n-1)
    palette = quant[color_levels]
    return palette

# this function converts the spectrum into dimensionless magnitude units that will be drawn in the circle projection
def normalize_spectrum(spectrum,min_offset = -25,max_offset = -5,follow_distance = 20,inv = 1,scales = (10.0,5.0,50.0),n_octaves=7,mode='cnt'):
    maxp = spectrum.max()
    offset = maxp - follow_distance
    if offset < min_offset:
        offset = min_offset
    if offset > max_offset:
        follow_distance = maxp - max_offset
        offset = max_offset
    sparsity_constraint = 0.2
    while (spectrum > offset).sum() > len(spectrum) * sparsity_constraint:
        follow_distance /= 1.05
        offset = maxp - follow_distance
    spectrum_p = np.choose(spectrum > offset,(0,spectrum-offset))
    spectrum_n = np.choose(spectrum <= offset,(0,offset-spectrum))
    spectrum = 10**(spectrum/10)
    power = np.log10(np.sum(spectrum))+2
    dpmp = 10*(power-2)-maxp
    data = np.ones(spectrum.shape) + scales[2] 
    spectrum_p = scales[0]*power*np.tanh(spectrum_p/follow_distance) #(1-1/np.log10(spectrum_p+10))
    spectrum_n = scales[1]*np.tanh(spectrum_n/50) #(1-1/np.log10(spectrum_n+10))
    
    if inv:
        data -= spectrum_p
        data += spectrum_n
    else:
        data += spectrum_p
        data -= spectrum_n
   
    print offset, power, maxp, dpmp, data.max(), data.min()
    data /= data.max()
    return data

def get_radians(n, final_angle = np.pi, log_index = False):
    rads = np.array(range(n))*1.0
    if log_index:
        rads = np.log2(rads+2.0)
    rads /= rads.max()
    rads *= final_angle
    return rads

def project_to_circle(data,rads):
    avgrad = (rads[:-1]+rads[1:])/2
    avgradcos = np.cos(avgrad)
    avgradsin = np.sin(avgrad)
    n = len(data)-1
    startx = np.multiply(avgradcos,data[:n])
    endx = np.multiply(avgradcos,data[1:n+1])
    starty = np.multiply(avgradsin,data[:n])
    endy = np.multiply(avgradsin,data[1:n+1])
    return startx,endx,starty,endy

def data_to_circle(data,sym = 6,log_index = False):
    final_angle = 2*np.pi / sym
    n = len(data)-1
    rads = get_radians(n+1,final_angle,log_index)
    offset = 0
    direction = 0
    present_angles = rads.copy()
    finalx = np.zeros((n*sym,2))
    finaly = np.zeros((n*sym,2))
    for i in range(sym):
        x0s,xfs,y0s,yfs = project_to_circle(data,present_angles)
        finalx[i*n:(i+1)*n,0] = x0s
        finalx[i*n:(i+1)*n,1] = xfs
        finaly[i*n:(i+1)*n,0] = y0s
        finaly[i*n:(i+1)*n,1] = yfs
        direction+=1
        offset += final_angle
        if direction%2:
            present_angles = offset + final_angle - rads
        else:
            present_angles = rads + offset
    return finalx, finaly

def center_circle(xs,ys,shape):
    radius = min(shape)/2
    centerx = shape[0]/2
    centery = shape[1]/2
    xs *= radius
    ys *= radius
    xs += centerx
    ys += centery
    return xs, ys

def color_and_data_to_value(color,data0,data1,normcolor):
    opdata = max((data0,data1))
    data2 = 1-opdata
    color = (color * opdata + normcolor * data2)
    return color.astype(np.uint8)

def get_colors(data, spectrum, log_index = False, n_octaves = 7, mode='cnt', outerval = 4.0, sparsity_constraint = 0.2):
    colordata = data.copy()
    n = data.shape[0]-1
    colordata = np.sqrt((1.0 - colordata))
    if mode=='cnt':
        spectrum = 10**(spectrum/10)
        binsize = 2**(np.linspace(0,n_octaves,len(spectrum)+1))*cqt.A0*2
        binsize = binsize[1:]-binsize[:-1]
        spectrum=10*np.log10(np.multiply(spectrum,np.log10(binsize+1)))
    spmax = spectrum.max()
    if spmax>20:
        offset = 0
    else:
        offset = spmax-20
    #print offset
    nonsparse = (spectrum > offset).sum()
    while nonsparse > len(spectrum) * sparsity_constraint:
        offset = spmax - (spmax - offset) / 1.05
        nonsparse = (spectrum > offset).sum()
    #    print nonsparse, offset
    colordata = np.choose(spectrum < offset, (colordata,colordata/10))
    colordata -= colordata.min()
    colordata /= colordata.max()
    final_angle = 2*np.pi
    if log_index:
        final_angle *= np.log2(n)
    else:
        final_angle *= n_octaves
    colorads = get_radians(n, final_angle, log_index)
    colorads = (colorads%(2*np.pi)*(255/(2.0*np.pi))).astype(np.int32)
    normcolor = outerval * np.ones(3)
    return [color_and_data_to_value(colormap[colorads[i]],colordata[i],colordata[i+1],normcolor) for i in range(n)]

def draw_sparse(xs,ys,colors,shape):
    im = Image.new('RGB',shape)
    draw = ImageDraw.Draw(im)
    n=len(colors)
    interpxs = xs[:,1]-(xs[:,1]-xs[:,0])/10
    interpys = ys[:,1]-(ys[:,1]-ys[:,0])/10
    for i in range(xs.shape[0]):
        draw.line( ( (interpxs[i],interpys[i]), (xs[i,1],ys[i,1]) ),fill = tuple(colors[i%n]) )
    return np.array(im).astype(np.float32)/255.0

def draw_data(xs, ys, colors, shape, no_color=False):
    if not no_color:
        im = Image.new('RGB',shape)
    else:
        im = Image.new('L', shape)
    draw = ImageDraw.Draw(im)
    n = len(colors)
    for i in range(xs.shape[0]):
        draw.line( ( (xs[i,0],ys[i,0]), (xs[i,1],ys[i,1]) ),fill = tuple(colors[i%n]) )
    return np.array(im).astype(np.float32)/255.0

def edge_filter(im, no_color=False):
    filt = np.ones((3,3))
    filt[1,1]=0
    if not no_color:
        imchoose = im.sum(axis=2)>0
    else:
        imchoose = im>0
    edge = signal.convolve2d( np.choose(imchoose,(0,1)), filt, 'same')
    edge = edge==8
    if not no_color:
        for i in range(im.shape[2]):
            im[:,:,i] = np.choose(edge,(im[:,:,i],0))
    else:
        im = np.choose(edge, (im,0))
    return im

def draw_spectrum(spectrum,shape,sym=6,inv=1,log_index=False,n_octaves=7,edge=True,mode='cnt',no_color=False):
    spectrum = 10*np.log10(spectrum+1e-160)
    if spectrum.max() < -159:
        if not no_color:
            im = Image.new('RGB',shape)
        else:
            im = Image.new('L', shape)
        return np.array(im).astype(np.float32),(0,0,0,0)
    t0=time.time()
    data = normalize_spectrum(spectrum,inv=inv,n_octaves=n_octaves,mode=mode)
    t0 = time.time()-t0 
    t1 = time.time()
    xs, ys = data_to_circle(data, sym, log_index)
    xs, ys = center_circle(xs, ys, shape)
    t1 = time.time()-t1
    t2 = time.time()
    colors = get_colors(data, spectrum, log_index, n_octaves, mode)
    if no_color:
        colors = [(np.sqrt(sum(j**2 for j in i)).astype(np.uint8),) for i in colors]
    t2 = time.time()-t2
    t3 = time.time()
    im = draw_data(xs,ys,colors,shape,no_color)
    if edge:
        im = edge_filter(im, no_color)
    t3 = time.time()-t3
    return im, (t0,t1,t2,t3)

def convolve_quaternion(im, pad=True, preserve_alpha=False, no_color=False):
    if pad:
        im = quaternion.pad(im)
    #print 'pad',time.time()-t0
    #t0=time.time()
    if no_color:
        im = quaternion.ACV(im)
        im.flatten()[im.argmax()]=0
        im = quaternion.fftshift_color(im)
        im = quaternion.sqrt_normalize(im)
        return im * 255.1
    r,i,j,k = quaternion.AQCV2(0, im[:,:,0], im[:,:,1], im[:,:,2])
    #print 'aqcv',time.time()-t0
    #t0=time.time()
    maxval = r.argmax()
    for n in (r,i,j,k):
        n.flatten()[maxval] = 0
    #print 'zero',time.time()-t0
    #t0=time.time()
    im = quaternion.create_image(r,i,j,k)
    #print 'image',time.time()-t0
    #t0=time.time()
    im = quaternion.sqrt_normalize(im)
    #print 'normalize',time.time()-t0
    #t0=time.time()
    if preserve_alpha:
        return im * 255.1
    for i in range(3):
        im[:,:,i] = np.multiply(im[:,:,i],im[:,:,3])
    im *= 255.1
    #print 'alpha',time.time()-t0
    return im[:,:,:3]

# this is the actual function that steps through the audio file and generates each frame 
def render_file(fin, outdir, shape = (512,512), framerate = 25, sym = 6, inv = 1, pad = True, mode = 'dft', preserve_alpha=False, params = {}, no_color=False):
    decoder = io.get_decoder(name = 'audioread')
    audio = decoder.Audio(fin)
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    fs = audio.samplerate
    nframes = audio.duration * framerate
    print 'fs: %d Hz, Duration: %d sec, Frames: %d'%(audio.samplerate,audio.duration,nframes)
    gram = None
    log_index = True
    n_octaves = None
    gram_args = []
    gram_kwargs = {'start':0,'end':None}
    if mode == 'dft':
        win = 2 * fs / framerate
        hop = fs / framerate
        nfft = None
        if 'w' in params:
            win = params['w']
        if 'n' in params:
            nfft = params['n']
        gram = dft.PowerSpectrum(audio,nfft=nfft)
        gram_args = [win,hop]
    elif mode == 'cnt':
        n = 60
        hop = 1.0 / framerate
        n_octaves = 8
        log_index = False
        if 'n' in params:
            n = params['n']
        if 'o' in params:
            n_octaves = params['o']
        gram_args=[n]
        gram_kwargs['hop'] = hop
        gram_kwargs['freq_base'] = cqt.A0 * 4
        gram_kwargs['freq_max'] = cqt.A0 * 2**n_octaves
        gram = cqt.CNTPowerSpectrum(audio)
    elif mode == 'gmt':
        n = 60
        hop = 1.0 / framerate
        n_octaves = 9
        log_index = False
        if 'n' in params:
            n = params['n']
        if 'o' in params:
            n_octaves = params['o']
        n = n * (n_octaves-1)
        gram_args = [n]
        gram_kwargs['thop'] = hop
        gram_kwargs['twin'] = 2*hop
        gram_kwargs['freq_base'] = cqt.A0 * 4
        gram_kwargs['freq_max'] = cqt.A0 * 2**n_octaves
        gram_kwargs['combine'] = True
        gram = auditory.GammatoneSpectrum(audio)
    i = 0
    j = 0
    #print gram_args
    #print gram_kwargs
    tqcv = 0
    iso226_factors = None
    timesums=np.zeros(4)
    for spectrum in gram.walk(*gram_args,**gram_kwargs):
        iso_init = isinstance(iso226_factors, np.ndarray)
        i += 1
        if hasattr(gram, 'fqs') and not iso_init:
            iso226_factors = np.array(map(iso,gram.fqs))
            iso226_factors = 10 ** (iso226_factors / 10)
            iso226_factors = 1 / iso226_factors
        if os.path.isfile(outdir+'conv%05d.png'%i):
            continue
        if iso_init:
            spectrum = np.multiply(spectrum,iso226_factors)
        j += 1
        im,times = draw_spectrum(spectrum,shape,sym,inv,log_index,n_octaves,mode=mode,no_color=no_color)
        timesums += np.array(times)
        imsave(outdir+'img%05d.png'%i,(im*255).astype(np.uint8)) # this is for saving the kernel images
        t0=time.time()
        im = convolve_quaternion(im, pad, preserve_alpha, no_color=no_color)
        tqcv += time.time()-t0
        imsave(outdir+'conv%05d.png'%i,im.astype(np.uint8))
        if i%100==0:
            ttot = tqcv + timesums.sum()
            actual_nframes = nframes - (i - j)
            pct = i / nframes
            pct_diff = j / actual_nframes
            eta = ttot / pct_diff - ttot
            print '%d %4.4f%% %6.2fs elapsed %6.2fs eta %6.2f convolution %s timing' % (i, 100.0 * pct, ttot, eta, tqcv, str(timesums))
    print 'done rendering',i,tqcv+timesums.sum(),tqcv,timesums
    return 




