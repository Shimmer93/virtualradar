import numpy as np
from mmwave.dsp.utils import *
from mmwave.dsp.angle_estimation import gen_steering_vec, aoa_capon, aoa_bartlett, peak_search_full_variance, variance_estimation, aoa_estimation_bf_one_point
from mmwave.dsp import compensation
from scipy.signal import find_peaks
import warnings

def beamforming_naive_mixed_xyz_modified(azimuth_input, input_ranges, range_resolution, method='Capon', num_vrx=12, est_range=90,
                                est_resolution=1):
    """ This function estimates the XYZ location of a series of input detections by performing beamforming on the
    azimuth axis and naive AOA on the vertical axis.
        
    TI xWR1843 virtual antenna map
    Row 1               8  9  10 11
    Row 2         0  1  2  3  4  5  6  7

    phi (ndarray):
    theta (ndarray):
    ranges (ndarray):
    xyz_vec (ndarray):

    Args:
        azimuth_input (ndarray): Must be a numpy array of shape (numDetections, numVrx)
        input_ranges (ndarray): Numpy array containing the rangeBins that have detections (will determine x, y, z for
        each detection)
        range_resolution (float): The range_resolution in meters per rangeBin for rangeBin->meter conversion
        method (string): Determines which beamforming method to use for azimuth aoa estimation.
        num_vrx (int): Number of virtual antennas in the radar platform. Default set to 12 for 1843
        est_range (int): The desired span of thetas for the angle spectrum. Used for gen_steering_vec
        est_resolution (float): The desired angular resolution for gen_steering_vec

    Raises:
        ValueError: If method is not one of two AOA implementations ('Capon', 'Bartlett')
        ValueError: azimuthInput's second axis should have same shape as the number of Vrx

    Returns:
        tuple [ndarray, ndarray, ndarray, ndarray, list]:
            1. A numpy array of shape (numDetections, ) where each element represents the elevation angle in degrees
            #. A numpy array of shape (numDetections, ) where each element represents the azimuth in degrees
            #. A numpy array of shape (numDetections, ) where each element represents the polar range in rangeBins
            #. A numpy array of shape (3, numDetections) and format: [x, y, z] where x, y, z are 1D arrays. x, y, z \
            should be in meters

    """
    if method not in ('Capon', 'Bartlett'):
        raise ValueError("Method argument must be 'Capon' or 'Bartlett'")

    if azimuth_input.shape[1] != num_vrx:
        raise ValueError("azimuthInput is the wrong shape. Change num_vrx if not using TI 1843 platform")

    doa_var_thr = 10
    num_vec, steering_vec = gen_steering_vec(est_range, est_resolution, 8)

    output_e_angles = []
    output_a_angles = []
    output_ranges = []
    nums_out = []

    for i, inputSignal in enumerate(azimuth_input):
        if method == 'Capon':
            doa_spectrum, _ = aoa_capon(np.reshape(inputSignal[:8], (8, 1)).T, steering_vec)
            doa_spectrum = np.abs(doa_spectrum)
        elif method == 'Bartlett':
            doa_spectrum = aoa_bartlett(steering_vec, np.reshape(inputSignal[:8], (8, 1)), axis=0)
            doa_spectrum = np.abs(doa_spectrum).squeeze()
        else:
            doa_spectrum = None

        # Find Max Values and Max Indices

        #    num_out, max_theta, total_power = peak_search(doa_spectrum)
        obj_dict, total_power = peak_search_full_variance(doa_spectrum, steering_vec.shape[0], sidelobe_level=0.9)
        num_out = len(obj_dict)
        nums_out.append(num_out)
        
        max_theta = [obj['peakLoc'] for obj in obj_dict]

        estimated_variance = variance_estimation(num_out, est_resolution, obj_dict, total_power)

        higher_rung = inputSignal[8:12]
        lower_rung = inputSignal[2:6]
        for j in range(num_out):
            ele_out = aoa_estimation_bf_one_point(4, higher_rung, steering_vec[max_theta[j]])
            azi_out = aoa_estimation_bf_one_point(4, lower_rung, steering_vec[max_theta[j]])
            num = azi_out * np.conj(ele_out)
            wz = np.arctan2(num.imag, num.real) / np.pi

            temp_angle = -est_range + max_theta[
                j] * est_resolution  # Converts to degrees, centered at boresight (0 degrees)
            # Make sure the temp angle generated is within bounds
            if np.abs(temp_angle) <= est_range and estimated_variance[j] < doa_var_thr:
                e_angle = np.arcsin(wz)
                a_angle = -1 * (np.pi / 180) * temp_angle  # Degrees to radians
                output_e_angles.append((180 / np.pi) * e_angle)  # Convert radians to degrees

                # print(e_angle)
                # if (np.sin(a_angle)/np.cos(e_angle)) > 1 or (np.sin(a_angle)/np.cos(e_angle)) < -1:
                # print("Found you", (np.sin(a_angle)/np.cos(e_angle)))
                # assert np.cos(e_angle) == np.nan, "Found you"

                # TODO: Not sure how to deal with arg of arcsin >1 or <-1
#                if np.sin(a_angle)/np.cos(e_angle) > 1:
#                    output_a_angles.append((180 / np.pi) * np.arcsin(1))
#                    print("Found a pesky nan")
#                elif np.sin(a_angle)/np.cos(e_angle) < -1:
#                    output_a_angles.append((180 / np.pi) * np.arcsin(-1))
#                    print("Found a pesky nan")
#                else:
#                    output_a_angles.append((180 / np.pi) * np.arcsin(np.sin(a_angle)/np.cos(e_angle))) # Why

                output_a_angles.append((180 / np.pi) * np.arcsin(np.sin(a_angle) * np.cos(e_angle)))  # Why

                output_ranges.append(input_ranges[i])

    phi = np.array(output_e_angles)
    theta = np.array(output_a_angles)
    ranges = np.array(output_ranges)

    # points could be calculated by trigonometry,
    x = np.sin(np.pi / 180 * theta) * ranges * range_resolution     # x = np.sin(azi) * range
    y = np.cos(np.pi / 180 * theta) * ranges * range_resolution     # y = np.cos(azi) * range
    z = np.tan(np.pi / 180 * phi) * ranges * range_resolution       # z = np.tan(ele) * range

    xyz_vec = np.array([x, y, z])

    # return phi, theta, ranges
    return phi, theta, ranges, xyz_vec, nums_out

def beamforming_naive_mixed_xyz_6843(azimuth_input, input_ranges, range_resolution, method='Capon', num_vrx=12, est_range=90,
                                est_resolution=1):
    """ This function estimates the XYZ location of a series of input detections by performing beamforming on the
    azimuth axis and naive AOA on the vertical axis.

    TI xWR6843 virtual antenna map
    Row 1               10 11
    Row 2               8  9 
    Row 3         4  5  6  7 
    Row 4         0  1  2  3 

    phi (ndarray):
    theta (ndarray):
    ranges (ndarray):
    xyz_vec (ndarray):

    Args:
        azimuth_input (ndarray): Must be a numpy array of shape (numDetections, numVrx)
        input_ranges (ndarray): Numpy array containing the rangeBins that have detections (will determine x, y, z for
        each detection)
        range_resolution (float): The range_resolution in meters per rangeBin for rangeBin->meter conversion
        method (string): Determines which beamforming method to use for azimuth aoa estimation.
        num_vrx (int): Number of virtual antennas in the radar platform. Default set to 12 for 1843
        est_range (int): The desired span of thetas for the angle spectrum. Used for gen_steering_vec
        est_resolution (float): The desired angular resolution for gen_steering_vec

    Raises:
        ValueError: If method is not one of two AOA implementations ('Capon', 'Bartlett')
        ValueError: azimuthInput's second axis should have same shape as the number of Vrx

    Returns:
        tuple [ndarray, ndarray, ndarray, ndarray, list]:
            1. A numpy array of shape (numDetections, ) where each element represents the elevation angle in degrees
            #. A numpy array of shape (numDetections, ) where each element represents the azimuth in degrees
            #. A numpy array of shape (numDetections, ) where each element represents the polar range in rangeBins
            #. A numpy array of shape (3, numDetections) and format: [x, y, z] where x, y, z are 1D arrays. x, y, z \
            should be in meters

    """
    if method not in ('Capon', 'Bartlett'):
        raise ValueError("Method argument must be 'Capon' or 'Bartlett'")

    if azimuth_input.shape[1] != num_vrx:
        raise ValueError("azimuthInput is the wrong shape. Change num_vrx if not using TI 6843 platform")

    doa_var_thr = 10
    num_vec, steering_vec = gen_steering_vec(est_range, est_resolution, 4)

    output_e_angles = []
    output_a_angles = []
    output_ranges = []

    signal_ele1 = inputSignal[[2, 6, 8, 10]]
    signal_ele2 = inputSignal[[3, 7, 9, 11]]

    for i, inputSignal in enumerate(azimuth_input):
        if method == 'Capon':
            doa_spectrum_azi1, _ = aoa_capon(np.reshape(inputSignal[:4], (4, 1)).T, steering_vec)
            doa_spectrum_azi2, _ = aoa_capon(np.reshape(inputSignal[4:8], (4, 1)).T, steering_vec)
            doa_spectrum_azi = np.abs(doa_spectrum_azi1 + doa_spectrum_azi2)
            doa_spectrum_ele1, _ = aoa_capon(np.reshape(signal_ele1, (4, 1)).T, steering_vec)
            doa_spectrum_ele2, _ = aoa_capon(np.reshape(signal_ele2, (4, 1)).T, steering_vec)
            doa_spectrum_ele = np.abs(doa_spectrum_ele1 + doa_spectrum_ele2)
        elif method == 'Bartlett':
            doa_spectrum_azi1 = aoa_bartlett(steering_vec, np.reshape(inputSignal[:4], (4, 1)), axis=0)
            doa_spectrum_azi2 = aoa_bartlett(steering_vec, np.reshape(inputSignal[4:8], (4, 1)), axis=0)
            doa_spectrum_azi = np.abs(doa_spectrum_azi1 + doa_spectrum_azi2)
            doa_spectrum_ele1 = aoa_bartlett(steering_vec, np.reshape(signal_ele1, (4, 1)), axis=0)
            doa_spectrum_ele2 = aoa_bartlett(steering_vec, np.reshape(signal_ele2, (4, 1)), axis=0)
            doa_spectrum_ele = np.abs(doa_spectrum_ele1 + doa_spectrum_ele2)
        else:
            doa_spectrum_azi = None
            doa_spectrum_ele = None

        # Find Max Values and Max Indices

        #    num_out, max_theta, total_power = peak_search(doa_spectrum)
        obj_dict_azi, total_power_azi = peak_search_full_variance(doa_spectrum_azi, steering_vec.shape[0], sidelobe_level=0.9)
        num_out_azi = len(obj_dict_azi)
        max_theta = [obj['peakLoc'] for obj in obj_dict_azi]

        #    num_out, max_theta, total_power = peak_search(doa_spectrum)
        obj_dict_ele, total_power_ele = peak_search_full_variance(doa_spectrum_ele, steering_vec.shape[0], sidelobe_level=0.9)
        num_out_ele = len(obj_dict_ele)
        max_phi = [obj['peakLoc'] for obj in obj_dict_ele]

        estimated_variance_azi = variance_estimation(num_out_azi, est_resolution, obj_dict_azi, total_power_azi)
        estimated_variance_ele = variance_estimation(num_out_ele, est_resolution, obj_dict_ele, total_power_ele)

        # Finish this part       

    phi = np.array(output_e_angles)
    theta = np.array(output_a_angles)
    ranges = np.array(output_ranges)

    # points could be calculated by trigonometry,
    x = np.sin(np.pi / 180 * theta) * ranges * range_resolution     # x = np.sin(azi) * range
    y = np.cos(np.pi / 180 * theta) * ranges * range_resolution     # y = np.cos(azi) * range
    z = np.tan(np.pi / 180 * phi) * ranges * range_resolution       # z = np.tan(ele) * range

    xyz_vec = np.array([x, y, z])

    # return phi, theta, ranges
    return phi, theta, ranges, xyz_vec