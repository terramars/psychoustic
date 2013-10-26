#-*- coding: utf-8 -*-

from _io_base import AudioBase
import audioread
import struct, numpy, itertools


support_formats = ['mp3','wav','aiff']


class Audio(AudioBase):
    ''''''

    def __init__(self, path):
        super(Audio, self).__init__(path)
        fh = audioread.audio_open(path)
        self._samplerate = fh.samplerate
        self._channels = fh.channels
        self._duration = fh.duration
        fh.close()

    @property
    def samplerate(self):
        return self._samplerate

    @property
    def channels(self):
        return self._channels

    @property
    def duration(self):
        return self._duration

    def __len__(self):
        return self._duration

    _max_pulse_value = 2**15

    def _pcm_bin2num(self, bins, channels, join=False):
        pcms = numpy.array(
                struct.unpack('<'+'h'*(len(bins)/2), bins), 
                float)
        pcms = pcms / self._max_pulse_value
        if channels <= 1:
            if join: return pcms
            return [pcms]
        chs = []
        for i in range(channels):
            chs.append(
                numpy.array(
                    [pcms[i] for i in xrange(i, len(pcms), channels)]))
        if not join:
            return chs
        return sum(chs) / channels

    def walk(self, win=1024, step=None, start=0, end=None,
            join_channels=True):
        if not step:
            step = win
        if not end:
            end = self._duration
        assert win >= step > 0
        assert start >= 0
        if start >= self._duration or end <= start:
            raise StopIteration
        #
        fh = audioread.audio_open(self._path)
        total = fh.samplerate * fh.duration / 1024.0
        start_point = int(float(start) / fh.duration * total)
        end_point = int(float(end) / fh.duration * total)
        win_len = int(win) * fh.channels * 2
        step_len = int(step) * fh.channels * 2
        #
        #if True:
        if win_len <= step * 1.5:
            win_buf = ''
            for p, buf in enumerate(fh):
                if p < start_point:
                    continue
                if p >= end_point:
                    break
                win_buf += buf
                while len(win_buf) > win_len:
                    yield self._pcm_bin2num(win_buf[:win_len], fh.channels,
                            join_channels)
                    win_buf = win_buf[step_len:]
            #
            if len(win_buf) > 0:
                win_buf += '\0' * (win_len - len(win_buf))
                yield self._pcm_bin2num(win_buf, fh.channels, join_channels)
        else:
            if join_channels:
                win_buf = numpy.array([], dtype=numpy.double)
                for p, buf in enumerate(fh):
                    if p < start_point:
                        continue
                    if p >= end_point:
                        break
                    newbuf = self._pcm_bin2num(buf, fh.channels, join_channels)
                    win_buf = numpy.append(win_buf, newbuf)
                    while len(win_buf) > win:
                        yield win_buf[:win]
                        win_buf = win_buf[step:]
                if len(win_buf) > 0:
                    yield numpy.append(win_buf, 
                            numpy.zeros([win-len(win_buf)], numpy.double))
            else:
                win_buf = [numpy.array([], dtype=numpy.double) \
                        for i in range(fh.channels)]
                for p, buf in enumerate(fh):
                    if p < start_point:
                        continue
                    if p >= end_point:
                        break
                    newbuf = self._pcm_bin2num(buf, fh.channels, join_channels)
                    for i in range(fh.channels):
                        win_buf[i] = numpy.append(win_buf[i], newbuf[i])
                    while len(win_buf[0]) > win:
                        yield [ win_buf[i][:win] \
                                for i in range(fh.channels)]
                        win_buf = [ win_buf[i][step:] \
                                for i in range(fh.channels)]
                if len(win_buf[0]) > 0:
                    yield [ numpy.append(win_buf[i], 
                                numpy.zeros([win-len(win_buf[i])], 
                                    numpy.double)) \
                            for i in range(fh.channels)]
        fh.close()

