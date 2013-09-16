import sys
import os
import shutil
from dear_tools import *


files=sys.argv[1:]
files.sort()
#fout=sys.argv[2]

replace_chars=[' ','(',')','&',';',"'",'"']

for fin in files:
    base=fin.rsplit('.',1)[0]
    clean_fin=fin
    for char in replace_chars:
        clean_fin=clean_fin.replace(char,'\\'+char)
    clean_base=clean_fin.rsplit('.',1)[0]
    outdir=base+'_pics/'
    clean_outdir=clean_base+'_pics/'
    wav=base+'.wav'
    clean_wav=clean_base+'.wav'
    fout=base+'_color.avi'
    clean_fout=clean_base+'_color.avi'
    if os.path.isfile(fout):
        print 'already have ',fout,'skipping'
        continue
    framerate=25

    p=os.popen('ffmpeg -y -i %s %s'%(clean_fin,clean_wav))
    p.close()
    print '\n'
    print 'made wav'

    render_file(wav,outdir,shape=(241,241),sym=4,framerate=framerate,inv=1,mode='cnt')
    print 'rendered images'

    p=os.popen('ffmpeg -y -r %d -sameq -i %sconv%%05d.png -i %s %s_tmp.avi'%(framerate,clean_outdir,clean_fin,clean_base))
    p.close()
    print '\n'
    print 'rendered video'

    p=os.popen('ffmpeg -y -i %s_tmp.avi -b:v 8000k %s'%(clean_base,clean_fout))
    p.close()
    print '\n'
    print 'downsampled video'

    os.remove('%s_tmp.avi'%base)
    shutil.rmtree(outdir)
    os.remove(wav)
    print 'cleaned temp stuff'


