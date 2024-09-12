import numpy as np
import mmwave.dsp as dsp
from mmwave.dsp.utils import Window
from scipy.constants import c

from utils.misc import dtype_det_2d_int
from radar.angle_estimation import beamforming_naive_mixed_xyz_modified

class RadarSignalProcessor:
    def __init__(self, 
                 num_chirps=48, 
                 num_samples=300, 
                 num_tx=4,
                 num_rx=3,
                 antenna_type='ti_xwr1843',
                 dist_antenna=0.0025,
                 center_freq=60.5e9, 
                 sample_freq=2e6, 
                 bandwidth=7e9,
                 cfar_num_train=8,
                 cfar_num_guard=4,
                 cfar_false_alarm_rate=1e-6,
                 cfar_l_bound_range=2.5,
                 cfar_l_bound_doppler=2.5,
                 use_window=True
                ):
        self.num_chirps = num_chirps
        self.num_samples = num_samples
        self.num_tx = num_tx
        self.num_rx = num_rx
        self.antenna_type = antenna_type
        self.center_freq = center_freq
        self.sample_freq = sample_freq
        self.bandwidth = bandwidth
        self.dist_antenna = dist_antenna if dist_antenna > 0 else c / (2 * center_freq)
        self.wavelength = c / self.center_freq
        self.dur_chirp = self.num_samples / self.sample_freq
        self.framerate = self.num_chirps / self.dur_chirp
        self.lower_freq = self.center_freq - self.bandwidth / 2
        self.range_res = c / (2 * self.bandwidth) # Range resolution, unit: m
        self.doppler_res = c / (2 * num_samples * num_tx * center_freq * self.dur_chirp)

        self.cfar_num_train = cfar_num_train
        self.cfar_num_guard = cfar_num_guard
        self.cfar_threshold = cfar_num_train * (cfar_false_alarm_rate ** (-1 / cfar_num_train) - 1)
        self.cfar_l_bound_range = cfar_l_bound_range
        self.cfar_l_bound_doppler = cfar_l_bound_doppler

        self.use_window = use_window
        self.window_range = None if use_window else Window.BLACKMAN
        self.window_doppler = None if use_window else Window.HAMMING

        self.snr_threshold = np.array([[2, 23], [10, 11.5], [35, 16.0]])
        self.peak_val_threshold = np.array([[4, 275], [1, 400], [500, 0]])

    def process_frame(self, frame):
        # signal: (num_chirps, num_rx, num_samples)

        # Range processing
        doppler_input = dsp.range_processing(frame, window_type_1d=self.window_range)

        # Doppler processing
        range_doppler_map, angle_input = dsp.doppler_processing(doppler_input, num_tx_antennas=self.num_tx, interleaved=True,
                                                                clutter_removal_enabled=True, window_type_2d=self.window_doppler)
        range_doppler_map = range_doppler_map.astype(np.int64)

        # CFAR
        threshold_range, noise_level_range = np.apply_along_axis(dsp.ca_, 0, range_doppler_map, l_bound=self.cfar_l_bound_range, 
                                                                 guard_len=self.cfar_num_guard, noise_len=self.cfar_num_train)
        thres_doppler, noise_level_doppler = np.apply_along_axis(dsp.ca_, 0, range_doppler_map.T, l_bound=self.cfar_l_bound_doppler, 
                                                                 guard_len=self.cfar_num_guard, noise_len=self.cfar_num_train)
        thres_doppler, noise_level_doppler = thres_doppler.T, noise_level_doppler.T

        # Peak Processing
        det_range_mask = (range_doppler_map > threshold_range)
        det_doppler_mask = (range_doppler_map > thres_doppler)
        det_mask_full = (det_range_mask & det_doppler_mask)
        det_peak_indices = np.argwhere(det_mask_full == True)
        det_peak_values = range_doppler_map[det_peak_indices[:, 0], det_peak_indices[:, 1]]
        det_snr = det_peak_values - noise_level_range[det_peak_indices[:, 0], det_peak_indices[:, 1]]

        dtype_location = '(' + str(self.num_tx) + ',)<f4'
        dtype_detObj2D = np.dtype({'names': ['rangeIdx', 'dopplerIdx', 'peakVal', 'location', 'SNR'],
                                   'formats': ['<i4', '<i4', '<f4', dtype_location, '<f4']})
        det_2d_raw = np.zeros((det_peak_indices.shape[0],), dtype=dtype_detObj2D)
        # det_2d_raw = {}
        det_2d_raw['rangeIdx'] = det_peak_indices[:, 0].squeeze()
        det_2d_raw['dopplerIdx'] = det_peak_indices[:, 1].squeeze()
        det_2d_raw['peakVal'] = det_peak_values.flatten()
        det_2d_raw['SNR'] = det_snr.flatten()

        # Peak Pruning
        det_2d_raw = dsp.prune_to_peaks(det_2d_raw, range_doppler_map, self.num_chirps, reserve_neighbor=True)

        # Peak Grouping
        det_2d = dsp.peak_grouping_along_doppler(det_2d_raw, range_doppler_map, self.num_chirps)
        det_2d = dsp.range_based_pruning(det_2d, self.snr_threshold, self.peak_val_threshold, self.num_samples, 0.5, self.range_res)

        # Angle Processing
        angle_input = angle_input[det_2d['rangeIdx'], :, det_2d['dopplerIdx']]
        _, _, ranges, xyz_coords, nums_out = beamforming_naive_mixed_xyz_modified(angle_input, det_2d['rangeIdx'], self.range_res, 
                                                                        num_vrx=self.num_rx * self.num_tx, method='Bartlett')
        
        # Add doppler speeds and intensities to pointcloud
        doppler_speeds = []
        intensities = []
        for n, d, p in zip(nums_out, det_2d['dopplerIdx'], det_2d['peakVal']):
            for _ in range(n):
                doppler_speeds.append(self.doppler_res * d)
                intensities.append(p)
        doppler_speeds = np.array(doppler_speeds)
        intensities = np.array(intensities)

        # doppler_speeds = self.doppler_res * det_2d['dopplerIdx']
        # intensities = det_2d['peakVal']
        pointcloud = np.concatenate([xyz_coords.T, doppler_speeds[:, None], intensities[:, None]], axis=1)

        return range_doppler_map, pointcloud
    

