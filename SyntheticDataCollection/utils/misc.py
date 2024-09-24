import numpy as np
from scipy.signal import hilbert
import yaml
from argparse import Namespace

def real2IQ(real_data, axis=-1):
    iq_data = hilbert(real_data, axis=axis)
    return iq_data

def convert2ADC(signal, resolution=16, is_complex=False):
    max_adc_value = 2**resolution - 1

    if is_complex:
        # Separate real and imaginary parts
        real_part = signal.real
        imag_part = signal.imag

        # Normalize real part to 0 to 1 range
        real_min = np.min(real_part)
        real_max = np.max(real_part)
        normalized_real = (real_part - real_min) / (real_max - real_min)

        # Normalize imaginary part to 0 to 1 range
        imag_min = np.min(imag_part)
        imag_max = np.max(imag_part)
        normalized_imag = (imag_part - imag_min) / (imag_max - imag_min)

        # Scale to ADC range
        adc_real = np.round(normalized_real * max_adc_value)
        adc_imag = np.round(normalized_imag * max_adc_value)

        # Combine real and imaginary parts into complex ADC data
        adc_data = adc_real.astype(np.uint16) + 1j * adc_imag.astype(np.uint16)
    else:
        min_value = np.min(signal)
        max_value = np.max(signal)

        # Normalize signal to 0 to 1 range
        normalized_signal = (signal - min_value) / (max_value - min_value)

        # Scale to ADC range
        adc_data = np.round(normalized_signal * max_adc_value).astype(np.uint16)

    return adc_data

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