from viz_tools import *
from pylab import cm

def draw_points(xs,ys,colors,im):
    for i in range(len(xs)):
        im[xs[i],ys[i],:]+=np.array(colors[i])

def interpolate_segment(im,x1,y1,x2,y2,color1,color2,n=8,min_dist=8):
    dx=x2-x1
    dy=y2-y1
    diff=np.sqrt((dx*dx+dy*dy))
    if diff/n<=min_dist:
        im[x1,y1,:]=color1
        im[x2,y2,:]=color2
        return
    curx=x1
    cury=y1
    dx/=1.0*n
    dy/=1.0*n
    n*=1.0
    for i in range(int(n+1)):
        color=color1*(1-i/n)+color2*(i/n)
        im[int(x1+i*dx),int(y1+i*dy),:]=color

def draw_points_interp(xs,ys,colors,im):
    for i in range(1,len(xs)):
        interpolate_segment(im,xs[i-1],ys[i-1],xs[i],ys[i],np.array(colors[i-1]),np.array(colors[i]))

def draw_points_draw(xs,ys,colors,draw):
    for i in range(1,len(xs)):
        draw.line(((xs[i-1],ys[i-1]),(xs[i],ys[i])),fill=tuple(colors[i-1]))

def rad_and_data_to_value(colors,rad,data,outerval=16.0,drawdown=8.0):
    index=int(rad%(2*pi)/2/pi*255)
    color=colors[index]
    color=(color*(data)+outerval*np.ones(3)*(1-data))/((1-data)*(1-data)*drawdown+1)
    return color

def get_colors(colors,colorads,colordata):
    return [rad_and_data_to_value(colors,colorads[i],colordata[i]) for i in range(len(colorads))]

def draw_spectrum(psd,spectrum,shape,name,sym=2,inv=1):
    #colors=[np.array(cm.jet(i)[:3])*255 for i in range(256)]
    colors=[np.array(cm.hsv(i)[:3])*255 for i in range(256)]
    data=normalize_spectrum(spectrum,inv=inv)
    colordata=data.copy()
    colordata=(1.0-colordata)**(1/3.0)
    colordata-=colordata.min()
    colordata/=colordata.max()
    #colordata*=255
    xs,ys=data_to_circle(data,psd,sym=sym)
    xs,ys=center_circle(xs,ys,shape)
    im=Image.new('RGB',shape)
    #draw=ImageDraw.Draw(im)
    im=np.array(im)
    nsemi=xs.shape[0]/sym 
    colorads=get_radians(nsemi,2*pi*np.log2(nsemi))
    #colors=[colors[i] for i in colordata]
    colors=get_colors(colors,colorads,colordata)
    #print colordata.max(),colordata.min(),name
    for i in range(sym):
        draw_points_interp(xs[i*nsemi:(i+1)*nsemi],ys[i*nsemi:(i+1)*nsemi],colors,im)
    imsave(name,im.astype(np.uint8))
    #im.save(name)
    return im#.astype(np.float32)

def convolve_image(im1,im2,name,rot=0,filt=1):
    im=np.zeros((im1.shape[0]*2-1,im1.shape[0]*2-1,3))
    tmp1s=[]
    #print name
    tmp2s=[]
    for channel in range(3):
        im1tmp=im1[:,:,channel]
        im1tmp=edge_filter(im1tmp)
        tmp1s.append(im1tmp)
        im2tmp=im2[:,:,channel]
        im2tmp=edge_filter(im2tmp)
        tmp2s.append(im2tmp)
    for channel in range(3):
        im1tmp=tmp1s[channel]
        for channel2 in range(channel,3):
            im2tmp=tmp2s[channel]
            imtmp=fftconvolve(im1tmp,im2tmp)
            im[:,:,channel]+=imtmp
            im[:,:,channel2]+=imtmp
    im=np.choose(im>0,(0,im))
    im+=1.0000001
    im=np.log(im)**2
    im*=255.0/(im.max())
    imsave(name,im.astype(np.uint8)[:-1,:-1])
    return im

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
        psd=get_spectrum(signal,i,fs,NFFT)
        #psd[:,0]=np.convolve(psd[:,0],gauss,'same')
        #psd[:,1]=np.convolve(psd[:,1],gauss,'same')
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
