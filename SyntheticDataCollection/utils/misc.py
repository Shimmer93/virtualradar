import numpy as np
from scipy.signal import hilbert

def real2IQ(real_data):
    iq_data = hilbert(real_data, axis=-1)
    return iq_data

def dtype_det_2d_int(num_tx):
    np.dtype({'names': ['range_idx', 'doppler_idx', 'peak_val', 'location', 'snr'], 
              'formats': ['<i4', '<i4', '<f4', '(' + str(num_tx) + ',)<f4', '<f4']})
    
def dtype_det_2d_float(num_tx):
    np.dtype({'names': ['range_idx', 'doppler_idx', 'peak_val', 'location', 'snr'], 
              'formats': ['<f4', '<f4', '<f4', '(' + str(num_tx) + ',)<f4', '<f4']})