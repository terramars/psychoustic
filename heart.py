import numpy as np
from scipy.signal import fftconvolve
from PIL import Image,ImageDraw

def heart(t):
    x=16*np.sin(t)**3
    y=13*np.cos(t)-5*np.cos(2*t)-2*np.cos(3*t)-np.cos(4*t)
    return x/17,-1*y/17

def shift_point(p,size,inv):
    if inv:
        x,y=p
        #y=y*-1
        y=y*5/17
        x=x*5/17
        p=x,y
    center=min([i/2 for i in size])
    p=tuple([int(i*center+center) for i in p])
    return p

def draw_heart(im,color,n=1000,inv=0,width=1):
    pi=np.pi
    draw=ImageDraw.Draw(im)
    size=im.size
    p1=heart(0)
    p1=shift_point(p1,size,inv)
    for i in range(1,n):
        t=i*2*pi/n
        p2=heart(t)
        p2=shift_point(p2,size,inv)
        draw.line((p1,p2),fill=color,width=width)
        p1=p2
    
def make_pretty_picture(shape,colors,width=1):
    im=Image.new('RGB',shape)
    draw_heart(im,colors[0],width=width)
    draw_heart(im,colors[1],inv=1,width=width)
    im2=np.zeros([i*2-1 for i in shape]+[3])
    im=np.array(im)
    imt=Image.fromarray(im.astype(np.uint8))
    imt.save('tmp.png')
    for channel in range(3):
        for channel2 in range(channel,3):
            imtmp=fftconvolve(im[:,:,channel],im[:,:,channel2])
            imtmp=np.choose(imtmp>0,(0,imtmp))
            im2[:,:,channel]+=imtmp
            im2[:,:,channel2]+=imtmp
    im2=np.sqrt(im2)
    im2/=im2.max()/255.0
    return Image.fromarray(im2.astype(np.uint8))

if __name__=='__main__':
    import sys
    fout=sys.argv[1]
    im=make_pretty_picture((1024,1024),((255,0,127),(0,127,255)),8) ## TODO set from command line
    im.save(fout)
