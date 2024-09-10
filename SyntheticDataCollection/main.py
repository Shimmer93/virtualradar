import numpy as np
import mmwave.dsp as dsp
from mmwave.dsp.utils import Window
import matplotlib.pyplot as plt
from scipy.signal import hilbert

from tcpip import TCPServer

def real2IQ(real_data):
    iq_data = hilbert(real_data, axis=-1)
    return iq_data


