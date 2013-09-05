from viz_tools import *
from pylab import cm
import quaternion
import time

def draw_points(xs,ys,colors,im):
    for i in range(len(xs)):
        im[xs[i],ys[i],:]+=np.array(colors[i])

def interpolate_segment(im,x1,y1,x2,y2,color1,color2,n=8,min_dist=32):
    dx=x2-x1
    dy=y2-y1
    diff=(dx*dx+dy*dy)
    if diff<=min_dist*min_dist:
        im[x1,y1,:]=color1
        im[x2,y2,:]=color2
        return
    n*=1.0
    dx/=n
    dy/=n
    for i in range(int(n+1)):
        #color=color1*(1-i/n)+color2*(i/n)
        im[int(x1+i*dx),int(y1+i*dy),:]=color1

def draw_points_interp(xs,ys,colors,im):
    colors=[np.array(c) for c in colors]
    for i in range(1,len(xs)):
        interpolate_segment(im,xs[i-1],ys[i-1],xs[i],ys[i],colors[i-1],colors[i])

def draw_points_draw(xs,ys,colors,draw):
    for i in range(1,len(xs)):
        draw.line(((xs[i-1],ys[i-1]),(xs[i],ys[i])),fill=tuple(colors[i-1]))

def rad_and_data_to_value(colors,rad,data,normcolor):
    index=int(rad%(2*pi)/2/pi*255)
    data2=1-data
    color=colors[index]
    color=(color*(data)+normcolor*(data2))
    #if data<0.3:
    #    color=normcolor
    return color.astype(np.uint8)

def get_colors(colors,colorads,colordata,outerval=4.0):
    normcolor=outerval*np.ones(3)
    return [rad_and_data_to_value(colors,colorads[i],colordata[i],normcolor) for i in range(len(colorads))]

def draw_spectrum(power,psd,spectrum,shape,name,sym=2,inv=1):
    #colors=[np.array(cm.jet(i)[:3])*255 for i in range(256)]
    t0=time.time()
    psdmax=psd.max()
    if psdmax <= 0.0:
        im=Image.new('RGB',shape)
        return np.array(im).astype(np.float32),(0,0,0,0,0)
    data=normalize_spectrum(spectrum,psdmax,inv=inv)
    colors=[np.array(cm.hsv(i)[:3])*255 for i in range(256)]
    colordata=data.copy()
    colordata=(1.0-colordata)**(1/3.0)
    psdmax=10*np.log10(psdmax)
    if psdmax>-20:
        colordata=np.choose(spectrum<-30,(colordata,colordata/10))
    else:
        colordata=np.choose(spectrum<psdmax-10,(colordata,colordata/10))
    colordata-=colordata.min()
    colordata/=colordata.max()
    t0=time.time()-t0
    t1=time.time()
    #colordata*=255
    xs,ys=data_to_circle(data,psd,sym=sym)
    xs,ys=center_circle(xs,ys,shape)
    t1=time.time()-t1
    t2=time.time()
    nsemi=xs.shape[0]/sym 
    colorads=get_radians(nsemi,2*pi*np.log2(nsemi))
    t2=time.time()-t2
    t3=time.time()
    #colors=[colors[i] for i in colordata]
    colors=get_colors(colors,colorads,colordata)
    t3=time.time()-t3
    #print colordata.max(),colordata.min(),name
    t4=time.time()
    im=Image.new('RGB',shape)
    #draw=ImageDraw.Draw(im)
    im=np.array(im).astype(np.float32)
    for i in range(sym):
        draw_points_interp(xs[i*nsemi:(i+1)*nsemi],ys[i*nsemi:(i+1)*nsemi],colors,im)
    t4=time.time()-t4
    #imsave(name,im.astype(np.uint8))
    #im.save(name)
    return np.array(im).astype(np.float32),(t0,t1,t2,t3,t4)#.astype(np.float32)

def convolve_quaternion(im1,im2,name,mode=0):
    im1/=255.0
    im1 = quaternion.pad(im1)
    im2/=255.0
    im2 = quaternion.pad(im2)
    qfunc = quaternion.QCV2
    if mode in (1,2,3):
        qfunc = quaternion.SPQCV
    r,i,j,k = qfunc(np.zeros(im1.shape[:2]),im1[:,:,0],im1[:,:,1],im1[:,:,2],np.zeros(im2.shape[:2]),im2[:,:,0],im2[:,:,1],im2[:,:,2],mode)
    img = quaternion.create_image(r,i,j,k)
    img = quaternion.sqrt_normalize(img)
    for i in range(3):
        img[:,:,i] = np.multiply(img[:,:,i],img[:,:,3])
    img = quaternion.normalize(img[:,:,:3])
    img *= 255.1
    imsave(name,img.astype(np.uint8))
    return img


def convolve_image(im1,im2,name,rot=0,filt=1,same=0,mode=0):
    im=np.zeros((im1.shape[0]*2-1,im1.shape[0]*2-1,3))
    tmp1s=[]
    #print name
    tmp2s=[]
    for channel in range(3):
        im1tmp=im1[:,:,channel]
        im1tmp=edge_filter(im1tmp)
        #fft1=np.zeros(im.shape[:2])
        #fft1[im1.shape[0]/2:3*im1.shape[0]/2,im1.shape[1]/2:im1.shape[1]*3/2]=im1tmp
        #fft1=fftshift(fftn(fft1))
        tmp1s.append(im1tmp)
        if same:
            tmp2s.append(im1tmp)
        else:    
            im2tmp=im2[:,:,channel]
            im2tmp=edge_filter(im2tmp)
            #fft2=np.zeros(im.shape[:2])
            #fft2[im1.shape[0]/2:3*im1.shape[0]/2,im1.shape[1]/2:im1.shape[1]*3/2]=im2tmp
            #fft2=fftshift(fftn(fft2))
            tmp2s.append(im2tmp)
    for channel in range(3):
        im1tmp=tmp1s[channel]
        for channel2 in range(channel,3):
            im2tmp=tmp2s[channel2]
            imtmp=fftconvolve(im1tmp,im2tmp) #fftshift(ifftn(im1tmp*im2tmp))
            im[:,:,channel]+=imtmp
            im[:,:,channel2]+=imtmp
    im=np.choose(im>0,(0,im))
    im+=1.0000001
    im=np.log(im)
    im*=255.0/(im.max())
    imsave(name,im.astype(np.uint8)[:-1,:-1])
    return im

def file_to_images(fin,outdir,filetype='png',shape=(640,640),framerate=25,sym=2,rot=0,inv=1):
    framerate/=2.0
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
    t3=0
    t4=time.time()
    ts=np.zeros(5)
    for i in range(signal.shape[0]/NFFT*2):
        if os.path.isfile('%sconv%05d.%s'%(outdir,i,filetype)):
            continue
        psd,power=get_spectrum(signal,i,fs,NFFT,channel=1)
        #psd[:,0]=np.convolve(psd[:,0],gauss,'same')
        #psd[:,1]=np.convolve(psd[:,1],gauss,'same')
        spectrum=10*np.log10(psd+1e-20)
        t3+=time.time()-t4
        t4=time.time()
        if signal.shape[1]==2:
            #spectrum = 10*np.log10(np.convolve(s[0].psd,gauss,'valid'))
            im1,ts_tmp=draw_spectrum(power,psd[:,0],spectrum[:,0],shape,name='%s%05d_1.%s'%(outdir,i,filetype),sym=sym,inv=inv)
            ts+=np.array(ts_tmp)
            #im2=draw_spectrum(power,psd[:,1],spectrum[:,1],shape,name='%s%05d_2.%s'%(outdir,i,filetype),sym=sym,inv=inv)
            t1+=time.time()-t4
            t4=time.time()
            imsave('%simg%05d.%s'%(outdir,i,filetype),im1)
            convolve_quaternion(im1,im1,name='%sconv%05d.%s'%(outdir,i,filetype),mode=0)
            t2+=time.time()-t4
            t4=time.time()
        else:
            #spectrum = 10*np.log10(np.convolve(s.psd,gauss,'valid'))
            im=draw_spectrum(spectrum,shape,name='%s%05d_1.%s'%(outdir,i,filetype),sym=sym,inv=inv)
            convolve_quaternion(im,im,name='%sconv%05d.%s'%(outdir,i,filetype),mode=0)
        if i%100==0:
            print i,time.time()-t0,t3,t1,t2,[(i,ts[i]) for i in range(5)]
    return
