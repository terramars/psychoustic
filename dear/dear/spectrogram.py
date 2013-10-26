#-*- coding: utf-8 -*-

import numpy as np
import matplotlib
#matplotlib.use('GTKAgg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.image import NonUniformImage
import matplotlib.colors as colo

from dear.spectrum import cqt, dft, auditory, SpectrogramFile
from dear.analysis import MFCCs


def plot_spectrogram(spec, Xd=(0,1), Yd=(0,1), norm=colo.LogNorm(vmin=0.000001), figname=None):
    #
    x_min, x_max = Xd
    y_min, y_max = Yd
    #
    fig = plt.figure(num=figname)
    nf = len(spec)
    for ch, data in enumerate(spec):
        #print ch, data.shape
        x = np.linspace(x_min, x_max, data.shape[0])
        y = np.linspace(y_min, y_max, data.shape[1])
        #print x[0],x[-1],y[0],y[-1]
        ax = fig.add_subplot(nf*100+11+ch)
        im = NonUniformImage(ax, interpolation='bilinear', cmap=cm.gray_r,
                norm=norm)
        im.set_data(x, y, data.T)
        ax.images.append(im)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_title('Channel %d' % ch)
        #ax.set_xlabel('timeline')
        ax.set_ylabel('frequency')
        print 'Statistics: max<%.3f> min<%.3f> mean<%.3f> median<%.3f>' % (data.max(), data.min(), data.mean(), np.median(data))
    #
    plt.show()


if __name__ == '__main__':
    import getopt, sys

    def exit_with_usage():
        print """Usage: $ python -m dear.spectrogram <options> /path/to/song

Options:
    [-s]    start time in second, default 0
    [-t]    end time, default is duration of song
    [-o]    output file
    [-g]    type of spectrogram, default dft:

        dft --- Discrete Fourier Transform
            [-w]    window size, default 1024
            [-p]    step size, default 512

        cqt --- Constant-Q Transform
            [-q]    Q, default 17
            [-p]    hop in second, default 0.02

        cnt --- Constant-N Transform
            [-n]    N, default 12
            [-p]    hop in second, default 0.02
            [-r]    resize window size of each of the bands if specified.

        gmt --- Gammatone Wavelet Transform
            [-n]    N, default 84
            [-w]    combine length in second, default 0.02
            [-p]    hop in second, default 0.02
            [-f]    frequency boundary, default (55, 7040)

        Y1...Y5 --- Auditory Spectrograms
            [-n]    N, default 84
            [-f]    frequency boundary, default (55, 7040)
            [-c]    Combine frames by -w and -h.
            [-w]    combine length in second, default 0.020
            [-p]    hop in second, default 0.020

        mfcc --- MFCCs Spectrogram
            [-n]    number of bands, default 20
            [-w]    window size, default 2048
            [-p]    step size, default 1024
            [-f]    frequency boundary, default (0, 7040)
"""
        exit()

    try:
        opts, args = getopt.getopt(sys.argv[1:], "g:s:t:o:p:w:q:n:f:b:rc")
    except getopt.GetoptError as ex:
        print ex
        exit_with_usage()
    if len(args) != 1:
        #print args
        exit_with_usage()

    import dear.io as io
    decoder = io.get_decoder(name='audioread')
    audio = decoder.Audio(args[0])
    print "SampleRate: %d Hz\nChannel(s): %d\nDuration: %d sec"\
            % (audio.samplerate, audio.channels, audio.duration)

    graph = 'dft'
    st = 0
    to = None
    outfile = None
    norm=colo.LogNorm(vmin=0.000001)

    for o, a in opts:
        if o == '-s':
            st = float(a)
        elif o == '-t':
            to = float(a)
        elif o == '-g':
            graph = a
            assert graph in ('dft','cqt','cnt','gmt','Y1','Y2','Y3','Y4','Y5','mfcc')
        elif o == '-o':
            outfile = a

    if to is None or to > audio.duration:
        r_to = audio.duration
    else:
        r_to = to

    if graph == 'dft':
        win = 1024
        hop = 512
        for o, a in opts:
            if o == '-w':
                win = int(a)
            elif o == '-p':
                hop = int(a)
        spec = [[]]
        gram = dft.PowerSpectrum(audio)
        for freqs in gram.walk(win, hop, start=st, end=to, join_channels=True):
            spec[0].append(freqs)
    #
    elif graph == 'mfcc':
        N = 20
        fmin, fmax = 0., 7040.
        win = 2048
        hop = 1024
        for o, a in opts:
            if o == '-w':
                win = int(a)
            elif o == '-p':
                hop = int(a)
            elif o == '-n':
                N = int(a)
            elif o == '-f':
                fmin, fmax = [float(f) for f in a.split(',',1)]
        spec = [[]]
        gram = MFCCs(audio)
        for freqs in gram.walk(N, fmin, fmax, win, hop, st, to):
            spec[0].append(freqs)
        norm = colo.Normalize()
    #
    elif graph == 'cqt':
        Q = 17
        hop = 0.02
        for o, a in opts:
            if o == '-q':
                Q = int(a)
            elif o == '-p':
                hop = float(a)
        spec = [[]]
        gram = cqt.CQTPowerSpectrum(audio)
        print 'total:', int((r_to-st)/hop)
        for t,freqs in enumerate(gram.walk(Q=Q, freq_base=55., freq_max=7040, hop=hop, start=st, end=to, join_channels=True)):
            if t%100 == 0:
                sys.stdout.write('%d...' % t)
                sys.stdout.flush()
            spec[0].append(freqs)
    #
    elif graph == 'cnt':
        N = 12
        hop = 0.02
        rw = False
        for o, a in opts:
            if o == '-n':
                N = int(a)
            elif o == '-p':
                hop = float(a)
            elif o == '-r':
                rw = True
        spec = [[]]
        gram = cqt.CNTPowerSpectrum(audio)
        print 'total:', int((r_to-st)/hop)
        for t, freqs in enumerate(
                gram.walk(N=N, freq_base=55., freq_max=7040, hop=hop, start=st, end=to, resize_win=rw)):
            if t%100==0:
                sys.stdout.write('%d...' % t)
                sys.stdout.flush()
            spec[0].append(freqs)
        print ""
    #
    elif graph == 'gmt':
        N = 84
        win = 0.02
        hop = 0.02
        freqs = [55., 7040.]
        combine=False
        for o, a in opts:
            if o == '-n':
                N = int(a)
            elif o == '-p':
                hop = float(a)
            elif o == '-w':
                win = float(a)
            elif o == '-f':
                freqs = [float(f) for f in a.split(',',1)]
            elif o == '-c':
                combine=True
        spec = [[]]
        gram = auditory.GammatoneSpectrum(audio)
        print 'total:', int((r_to-st)/hop)
        for t, freqs in enumerate(
                gram.walk(N=N, freq_base=freqs[0], freq_max=freqs[1], 
                    start=st, end=to, combine=combine, twin=win, thop=hop)):
            if t%100==0:
                sys.stdout.write('%d...' % t)
                sys.stdout.flush()
            spec[0].append(freqs)
        print ""
    #
    elif graph in ('Y1','Y2','Y3','Y4','Y5'):
        N = 84
        win = 0.020
        hop = 0.020
        freqs = [55.,7040.]
        combine = False
        for o, a in opts:
            if o == '-n':
                N = int(a)
            elif o == '-p':
                hop = float(a)
            elif o == '-w':
                win = float(a)
            elif o == '-f':
                freqs = [float(f) for f in a.split(',',1)]
            elif o == '-c':
                combine=True
        spec = [[]]
        gram = getattr(auditory,graph)
        gram = gram(audio)
        print 'total:', int((r_to-st)/hop)
        for t, freqs in enumerate(
                gram.walk(N=N, freq_base=freqs[0], freq_max=freqs[1], 
                    start=st, end=to, combine=combine, twin=win, thop=hop)):
            if t%100==0:
                sys.stdout.write('%d...' % t)
                sys.stdout.flush()
            spec[0].append(freqs)
        print ""

    # to dB scale
    dBmax, dBmin = -1., -160.
    if graph in ('dft','cqt','cnt'):
        magmin = 10**(dBmin/20)
        for g in spec:
            for i,frame in enumerate(g):
                g[i] = 20*np.log10(np.maximum(frame/20.,magmin))
        norm = colo.Normalize(vmin=dBmin, vmax=dBmax) 
    elif graph in ('gmt','Y1','Y2','Y3','Y4','Y5'):
        magmin = 10**(dBmin/20)
        for g in spec:
            for i,frame in enumerate(g):
                g[i] = 20*np.log10(np.maximum(frame,magmin))
        norm = colo.Normalize(vmin=dBmin, vmax=dBmax) 

    figname = "%s - %s" % (graph, audio.path)
    plot_spectrogram(np.array(spec), (0,len(spec[0])), (0,len(spec[0][0])), norm=norm, figname=figname)
    if outfile:
        out = SpectrogramFile(outfile, 'w')
        out.dump(spec[0])
        out.close()


