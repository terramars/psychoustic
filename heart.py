import numpy as np
from scipy.signal import fftconvolve
from PIL import Image,ImageDraw
from dear_tools import convolve_quaternion


def heart(t, orientation = 1):
    x=16*np.sin(t)**3
    y=13*np.cos(t)-5*np.cos(2*t)-2*np.cos(3*t)-np.cos(4*t)
    return x/17,-1 * orientation * y/17

def apply_offset(p,size, offsetx = 0, offsety = 0):
    x,y = p
    size = min([i/2 for i in size])
    x = x*size + size
    y = y*size + size
    x += offsetx
    y += offsety
    p = tuple(int(i) for i in (x,y))
    return p

def shift_point(p,size,inv, nest = 1):
    if inv:
        x,y=p
        y=y*-1
        y=y*5/17
        x=x*5/17
        p=x,y
    center=min([i/2 for i in size])
    p=tuple([int(i*center+center) for i in p])
    return p

def calculate_new_offset(orientation, currentsize, shrink, currentoffsetx, currentoffsety):
    size = [int(i*shrink) for i in currentsize]
    offsetx = currentoffsetx + (currentsize[0] - size[0]) / 2
    p0y = apply_offset(heart(0,1),currentsize,0, 0)[1]
    print p0y
    if orientation == 1:
        offsety = currentoffsety - p0y + size[1]
    else:
        offsety = currentoffsety + p0y

    return size, offsetx, offsety
    

def red_blue_circle(t, orientation):
    mult = 3
    theta = (np.pi / 2)
    g = 0
    t = np.pi - t
    g = 0.25 * (np.cos(4*t) + 1)
    r = 0.5 * (np.cos(mult*t) + 1)
    b = 0.5 * (np.sin(mult*t - theta) + 1)
    if orientation == 1:
        g /= 2
    else:
        b /= 2
    return (int(r*255),int(255*g),int(255*b))

def draw_two_level(im, n, size, width, color, scale = 0.5):
    orientation = 1
    offsetx = 0
    offsety = 0
    draw_with_offset(im,size,color,n,orientation,width,offsetx,offsety)
    orientation= -1
    size,offsetx,offsety = calculate_new_offset(orientation,size,scale,offsetx,offsety)
    draw_with_offset(im,size,color,n,orientation,width,offsetx,offsety)
    return im


def draw_with_offset(im, size, color, n=1000, orientation = 1, width = 1, offsetx = 0, offsety = 0):
    pi = np.pi
    draw = ImageDraw.Draw(im)
    p1 = heart(0,orientation)
    p1 = apply_offset(p1,size, offsetx, offsety)
    for i in range(1,n):
        t = i*2*pi / n
        p2 = heart(t,orientation)
        p2 = apply_offset(p2,size,offsetx, offsety)
        draw.line((p1,p2),fill=color(t, orientation),width=width)
        p1=p2

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

def do_quaternion_convolution(im):
    return convolve_quaternion(im)
    
def do_simple_color_convolution(im, power = 0.5):
    im2=np.zeros([i*2-1 for i in im.shape[:2]]+[3])
    #imnorm = None
    #imnorm = np.ones(im.shape[:2])
    #imnorm = fftconvolve(imnorm,imnorm)
    #imnorm /= imnorm.max()
    for channel in range(3):
        for channel2 in range(channel,3):
            imtmp=fftconvolve(im[:,:,channel],im[:,:,channel2])
            imtmp=np.choose(imtmp>0,(0,imtmp))
            #imtmp = np.divide(imtmp,imnorm)
            im2[:,:,channel]+=imtmp
            im2[:,:,channel2]+=imtmp
    im2=np.power(im2,power)
    im2/=im2.max()/255.0
    return im2
 

def make_pretty_picture(shape,colors,width=1):
    im=Image.new('RGB',shape)
    draw_heart(im,colors[0],width=width)
    draw_heart(im,colors[1],inv=1,width=width)
    im=np.array(im)
    imt=Image.fromarray(im.astype(np.uint8))
    imt.save('tmp.png')
    im2 = do_simple_color_convolution(im)
    return Image.fromarray(im2.astype(np.uint8))

if __name__=='__main__':
    import sys
    fout=sys.argv[1]
    im=make_pretty_picture((1024,1024),((255,0,127),(0,127,255)),8) ## TODO set from command line
    im.save(fout)
