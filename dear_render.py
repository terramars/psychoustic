import sys
import os
import shutil
from dear_tools import *


def cleanup():
  quaternion.ctx.pop()

import atexit

atexit.register(cleanup)


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--frequency-mode', type=str, choices=('dft', 'cnt', 'gmt'), default='cnt', help='algorithm to calculate frequency power, default cnt')
parser.add_argument('--sym', type=int, default=6, help='number of symmetries in the kernel, default 6')
parser.add_argument('files', type=str, nargs='*', help='input files to render')
parser.add_argument('--resolution', '-r', type=int, default=512, help='kernel resolution (half final), default 512')
parser.add_argument('--framerate', '-fr', type=int, default=30, help='framerate for video, default 30')
parser.add_argument('--no-inv', action='store_true', default=False, help='use the outside facing kernel')
parser.add_argument('--keep-frames', action='store_true', default=False, help='save image directory after run')
parser.add_argument('--preserve-alpha', action='store_true', default=False, help='save the images with alpha')
args = parser.parse_args()


mode = args.frequency_mode
sym = args.sym
files=args.files
files.sort()
resolution = args.resolution
framerate = args.framerate
inv = not args.no_inv

replace_chars=[' ','(',')','&',';',"'",'"']

if mode not in ['dft','cnt','gmt']:
    print 'invalid mode',mode
    sys.exit(1)

for fin in files:
    base=fin.rsplit('.',1)[0]
    clean_fin=fin
    for char in replace_chars:
        clean_fin=clean_fin.replace(char,'\\'+char)
    clean_base=clean_fin.rsplit('.',1)[0]
    fout=base  # +'_color_'+mode+'_'+str(sym)
    clean_fout=clean_base  # +'_color_'+mode+'_'+str(sym)
    outdir=fout+'_pics/'
    clean_outdir=clean_fout+'_pics/'
    wav=fout+'.wav'
    clean_wav=clean_fout+'.wav'
    fout+='.avi'
    clean_fout+='.avi'
    if os.path.isfile(fout):
        print 'already have ',fout,'skipping'
        continue

    if not os.path.isfile(wav):
        p=os.popen('ffmpeg -y -i %s %s'%(clean_fin,clean_wav))
        p.close()
        print '\n'
        print 'made wav'

    render_file(wav,outdir,shape=(resolution, resolution),sym=sym,framerate=framerate,inv=inv,pad=True,mode=mode,preserve_alpha=args.preserve_alpha)
    print 'rendered images'

    #p=os.popen('ffmpeg -y -r %d -sameq -i %simg%%05d.png -i %s %s'%(framerate,clean_outdir,clean_fin,clean_fout.rsplit('.',1)[0]+'_img.avi'))
    p=os.popen('ffmpeg -y -framerate %d -i %sconv%%05d.png -i %s -vcodec libx264 -crf 18 -preset slow -vf "transpose=1" -r %d %s'%(framerate,clean_outdir,clean_fin,framerate,clean_fout))
    p.close()
    print '\n'
    print 'rendered video'

    if not args.keep_frames:
        shutil.rmtree(outdir)
        os.remove(wav)
    	print 'cleaned temp stuff'


