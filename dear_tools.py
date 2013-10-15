from dear import io
from dear.spectrum import dft, cqt
from matplotlib import cm
import numpy as np
from PIL import ImageDraw, Image
import quaternion
from scipy import signal
from scipy.misc import imsave
import os
import time

colormap = [(np.array(cm.hsv(i)[:3])*255).astype(np.uint8) for i in range(256)]

def normalize_spectrum(spectrum,offset = -10,inv = 1,scales = (24.0,12)):
    maxp = spectrum.max()
    spectrum_p = np.choose(spectrum > offset,(0,spectrum-offset))
    spectrum_n = np.choose(spectrum <= offset,(0,offset-spectrum))
    data = np.ones(spectrum.shape) + scales[1]
    spectrum_p = scales[0]*(1-1/np.log10(spectrum_p+10)) #np.tanh(spectrum_p/20.0)
    spectrum_n = np.log2(np.exp(spectrum).sum()+2)*2*(1-1/np.log10(spectrum_n+10)) #np.tanh(spectrum_n/50.0)
    if inv:
        data -= spectrum_p
        data += spectrum_n
    else:
        data += spectrum_p
        data -= spectrum_n
    data /= max(data.max(),10.0)
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

def get_colors(data, spectrum, log_index = False, n_octaves = 7, outerval = 4.0):
    colordata = data.copy()
    n = data.shape[0]-1
    colordata = np.sqrt((1.0 - colordata))
    if spectrum.max()>-20:
        offset = -30
    else:
        offset = spectrum.max()-10
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

def draw_data(xs, ys, colors, shape):
    im = Image.new('RGB',shape)
    draw = ImageDraw.Draw(im)
    n = len(colors)
    for i in range(xs.shape[0]):
        draw.line( ( (xs[i,0],ys[i,0]), (xs[i,1],ys[i,1]) ),fill = tuple(colors[i%n]) )
    return np.array(im).astype(np.float32)/255.0

def edge_filter(im):
    filt = np.ones((3,3))
    filt[1,1]=0
    edge = signal.convolve2d( np.choose(im.sum(axis=2)>0,(0,1)), filt, 'same')
    edge = edge==8
    for i in range(im.shape[2]):
        im[:,:,i] = np.choose(edge,(im[:,:,i],0))
    return im

def draw_spectrum(spectrum,shape,sym=6,inv=1,log_index=False,n_octaves=7,edge=True):
    spectrum = 10*np.log10(spectrum+1e-160)
    if spectrum.max() < -159:
        im = Image.new('RGB',shape)
        return np.array(im).astype(np.float32),(0,0,0,0)
    t0=time.time()
    data = normalize_spectrum(spectrum,inv=inv)
    t0 = time.time()-t0 
    t1 = time.time()
    xs, ys = data_to_circle(data, sym, log_index)
    xs, ys = center_circle(xs, ys, shape)
    t1 = time.time()-t1
    t2 = time.time()
    colors = get_colors(data, spectrum, log_index, n_octaves)
    t2 = time.time()-t2
    t3 = time.time()
    im = draw_data(xs,ys,colors,shape)
    if edge:
        im = edge_filter(im)
    t3 = time.time()-t3
    return im, (t0,t1,t2,t3)

def convolve_quaternion(im,pad=True):
    if pad:
        im = quaternion.pad(im)
    r,i,j,k = quaternion.AQCV2(0, im[:,:,0], im[:,:,1], im[:,:,2])
    maxval = r.argmax()
    for n in (r,i,j,k):
        n.flatten()[maxval] = 0
    im = quaternion.create_image(r,i,j,k)
    im = quaternion.sqrt_normalize(im)
    for i in range(3):
        im[:,:,i] = np.multiply(im[:,:,i],im[:,:,3])
    im *= 255.1
    return im[:,:,:3]

def render_file(fin, outdir, shape = (512,512), framerate = 25, sym = 6, inv = 1, pad = True, mode = 'dft', params = {}):
    decoder = io.get_decoder(name = 'audioread')
    audio = decoder.Audio(fin)
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    fs = audio.samplerate
    print 'fs: %d Hz, Duration: %d sec, Frames: %d'%(audio.samplerate,audio.duration,audio.duration*framerate)
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
        n_octaves = 9
        log_index = False
        if 'n' in params:
            n = params['n']
        if 'o' in params:
            n_octaves = params['o']
        gram_args=[n]
        gram_kwargs['hop'] = hop
        gram_kwargs['freq_base'] = cqt.A0 * 2
        gram_kwargs['freq_max'] = cqt.A0 * 2**n_octaves
        gram = cqt.CNTPowerSpectrum(audio)
    i=0
    print gram_args
    print gram_kwargs
    tqcv=0
    timesums=np.zeros(4)
    for spectrum in gram.walk(*gram_args,**gram_kwargs):
        if os.path.isfile(outdir+'conv%05d.png'%i):
            i+=1
            continue
        im,times = draw_spectrum(spectrum,shape,sym,inv,log_index,n_octaves)
        timesums += np.array(times)
        imsave(outdir+'img%05d.png'%i,(im*255).astype(np.uint8))
        t0=time.time()
        im = convolve_quaternion(im,pad)
        tqcv += time.time()-t0
        imsave(outdir+'conv%05d.png'%i,im.astype(np.uint8))
        if i%100==0:
            print i,tqcv+timesums.sum(),tqcv,timesums
        i+=1
    print 'done rendering',i,tqcv+timesums.sum(),tqcv,timesums
    return 




