import sys
import os
import shutil
from dear_tools import *

mode = sys.argv[1]
sym = int(sys.argv[2])
files=sys.argv[3:]
files.sort()
#fout=sys.argv[2]

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
    fout=base+'_color_'+mode+'_'+str(sym)
    clean_fout=clean_base+'_color_'+mode+'_'+str(sym)
    outdir=fout+'_pics/'
    clean_outdir=clean_fout+'_pics/'
    wav=fout+'.wav'
    clean_wav=clean_fout+'.wav'
    fout+='.avi'
    clean_fout+='.avi'
    if os.path.isfile(fout):
        print 'already have ',fout,'skipping'
        continue
    framerate=25

    if not os.path.isfile(wav):
        p=os.popen('ffmpeg -y -i %s %s'%(clean_fin,clean_wav))
        p.close()
        print '\n'
        print 'made wav'

    render_file(wav,outdir,shape=(512,512),sym=sym,framerate=framerate,inv=1,pad=True,mode=mode)
    print 'rendered images'

    #p=os.popen('ffmpeg -y -r %d -sameq -i %simg%%05d.png -i %s %s'%(framerate,clean_outdir,clean_fin,clean_fout.rsplit('.',1)[0]+'_img.avi'))
    p=os.popen('ffmpeg -y -r %d -sameq -i %sconv%%05d.png -i %s %s'%(framerate,clean_outdir,clean_fin,clean_fout))
    p.close()
    print '\n'
    print 'rendered video'

    shutil.rmtree(outdir)
    os.remove(wav)
    print 'cleaned temp stuff'


