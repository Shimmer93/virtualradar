import numpy as np
from scipy.signal import hilbert
import yaml
from argparse import Namespace

def real2IQ(real_data, axis=-1):
    iq_data = hilbert(real_data, axis=axis)
    return iq_data

def dtype_det_2d_int(num_tx):
    np.dtype({'names': ['range_idx', 'doppler_idx', 'peak_val', 'location', 'snr'], 
              'formats': ['<i4', '<i4', '<f4', '(' + str(num_tx) + ',)<f4', '<f4']})
    
def dtype_det_2d_float(num_tx):
    np.dtype({'names': ['range_idx', 'doppler_idx', 'peak_val', 'location', 'snr'], 
              'formats': ['<f4', '<f4', '<f4', '(' + str(num_tx) + ',)<f4', '<f4']})
    
def read_cfg(cfg_path, mode='dict'):
    assert mode in ['dict', 'namespace'], 'Unsupported read mode'
    with open(cfg_path, 'r', errors='ignore') as f:
        cfg = yaml.safe_load(f)
    if mode == 'namespace':
        cfg = Namespace(**cfg)
    return cfg