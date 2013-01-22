import sys
import os
import shutil
from viz_tools import file_to_images

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
    fout=base+'.avi'
    clean_fout=clean_base+'.avi'
    framerate=25

    p=os.popen('ffmpeg -y -i %s %s'%(clean_fin,clean_wav))
    p.close()
    print '\n'
    print 'made wav'

    file_to_images(wav,outdir,shape=(321,321),sym=6,framerate=framerate,inv=1)
    print 'rendered images'

    p=os.popen('ffmpeg -y -r %d -sameq -i %sconv%%04d.png -i %s %s_tmp.avi'%(framerate,clean_outdir,clean_fin,clean_base))
    p.close()
    print '\n'
    print 'rendered video'

    p=os.popen('ffmpeg -y -i %s_tmp.avi -b:v 4000k %s'%(clean_base,clean_fout))
    p.close()
    print '\n'
    print 'downsampled video'

    os.remove('%s_tmp.avi'%base)
    shutil.rmtree(outdir)
    os.remove(wav)
    print 'cleaned temp stuff'


