#-*- coding: utf-8 -*-

import dear.spectrum.dft as dft
import dear.spectrum.cqt as cqt
import dear.spectrum.auditory as auditory

DFTSpectrum = dft.Spectrum
DFTPowerSpectrum = dft.PowerSpectrum
CQTSpectrum = cqt.Spectrum
CQTPowerSpectrum = cqt.CQTPowerSpectrum
CNTSpectrum = cqt.CNTSpectrum
CNTPowerSpectrum = cqt.CNTPowerSpectrum
from dear.spectrum.auditory import GammatoneSpectrum, Y1, Y2, Y3, Y4, Y5
from dear.spectrum._base import SpectrogramFile
