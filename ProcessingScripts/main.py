import socket
import numpy as np
from scipy.signal.windows import hann, chebwin
from scipy.signal import butter, find_peaks, filtfilt
from scipy.fftpack import fft, fftshift
from scipy.constants import c
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import argparse
import datetime
import os

class TCPServer:
    def __init__(self, host, port, timeout=120):
        self.host = host
        self.port = port
        self.tcp_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.timeout = timeout
        self.conn = None
        self.addr = None

    def start_server(self):
        self.tcp_server.bind((self.host, self.port))
        self.tcp_server.settimeout(self.timeout)
        self.tcp_server.listen(1)
        self.conn, self.addr = self.tcp_server.accept()
        print('Connected by', self.addr)
        
    def receive_data(self, datalengthbyte):
        data = self.conn.recv(datalengthbyte)
        self.conn.settimeout(self.timeout)
        return data
    
    def close_server(self):
        self.conn.close()
        self.tcp_server.close()
    
class RadarSignalProcessor:
    def __init__(self, 
                 num_chirps=48, 
                 num_samples=300, 
                 num_antennas=4,
                 center_freq=60.5e9, 
                 sample_freq=2e6, 
                 bandwidth=7e9,
                 dist_antenna=-1,
                 cfar_num_train=8,
                 cfar_num_guard=4,
                 cfar_false_alarm_rate=1e-6,
                 use_window=True
                ):
        self.num_chirps = num_chirps
        self.num_samples = num_samples
        self.num_antennas = num_antennas
        self.center_freq = center_freq
        self.sample_freq = sample_freq
        self.bandwidth = bandwidth
        self.dist_antenna = dist_antenna if dist_antenna > 0 else c / (2 * bandwidth)
        self.wavelength = c / self.center_freq
        self.dur_chirp = self.num_samples / self.sample_freq
        self.framerate = self.num_chirps / self.dur_chirp
        self.lower_freq = self.center_freq - self.bandwidth / 2
        self.range_res = c / (2 * self.bandwidth) # Range resolution, unit: m
        self.doppler_res = c * (1 / (self.num_chirps * self.dur_chirp)) / self.center_freq # Doppler speed resolution, unit: m/s

        self.cfar_num_train = cfar_num_train
        self.cfar_num_guard = cfar_num_guard
        self.cfar_threshold = cfar_num_train * (cfar_false_alarm_rate ** (-1 / cfar_num_train) - 1)

        self.win_range = hann(self.num_samples)
        self.win_doppler = hann(self.num_chirps)
        self.use_window = use_window

    def range_fft(self, data):
        if self.use_window:
            data = data * self.win_range[np.newaxis, :, np.newaxis]
        range_fft = np.fft.fft(data, axis=1)
        return range_fft

    def doppler_fft(self, data):
        if self.use_window:
            data = data * self.win_doppler[:, np.newaxis, np.newaxis]
        doppler_fft = np.fft.fftshift(np.abs(np.fft.fft(data, axis=0)), axes=0)
        return doppler_fft

    def cfar(self, data):
        # data: num_chirps x num_samples
        assert data.shape[1] > self.cfar_num_train + self.cfar_num_guard, 'Signal length is too short'
        num_train_half = self.cfar_num_train // 2
        num_guard_half = self.cfar_num_guard // 2
        num_all_half = num_train_half + num_guard_half
        cfar_map = np.zeros_like(data)
        for i in range(data.shape[0]):
            for j in range(num_all_half, data.shape[1] - num_all_half):
                training_cells = np.concatenate([data[i, j - num_all_half:j - num_guard_half], data[i, j + num_guard_half + 1:j + num_all_half + 1]])
                noise_level = np.mean(training_cells)
                threshold = self.cfar_threshold * noise_level
                if data[i, j] > threshold:
                    cfar_map[i, j] = data[i, j]

        return cfar_map

    # def cfar_algorithm(self, data):
    #     assert data.shape[1] > self.cfar_num_train + self.cfar_num_guard, 'Signal length is too short'

    #     cfar_map = np.zeros_like(data)
    #     for chirp in range(data.shape[0]):
    #         for antenna in range(data.shape[2]):
    #             for i in range(self.cfar_num_train + self.cfar_num_guard, data.shape[1] - self.cfar_num_train - self.cfar_num_guard):
    #                 training_cells = np.concatenate([data[chirp, i - self.cfar_num_train - self.cfar_num_guard:i - self.cfar_num_guard, antenna], data[chirp, i + self.cfar_num_guard + 1:i + self.cfar_num_guard + self.cfar_num_train + 1, antenna]])
    #                 noise_level = np.mean(training_cells)
    #                 threshold = self.cfar_threshold * noise_level
    #                 if data[chirp, i, antenna] > threshold:
    #                     cfar_map[chirp, i, antenna] = data[chirp, i, antenna]

    #     return cfar_map
    
    def process_signal(self, data):
        # TODOs:
        # 1. more antennas (12 should be enough)
        # 2. beamforming
        # 3. clutter removal
        # 4. stap (space-time adaptive processing)
        # 5. multiple targets
        # data: num_chirps x num_samples x num_antennas
        # assert data.shape == (self.num_chirps, self.num_samples, self.num_antennas), f'Data shape is incorrect: {data.shape}'

        # Apply FFT to convert time domain to frequency domain
        print(data)
        range_fft = self.range_fft(data)
        doppler_fft = self.doppler_fft(range_fft)

        filename = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S.png')
        filename = f'range_fft_{filename}'
        # abs_doppler_fft = np.abs(doppler_fft)
        # abs_doppler_fft = abs_doppler_fft / np.max(abs_doppler_fft)

        # Apply CFAR to range-doppler map
        range_doppler_map = doppler_fft.sum(axis=2)
        positive_range_doppler_map = range_doppler_map[:, range_doppler_map.shape[1]//2:]
        plot_range_doppler(positive_range_doppler_map, filename=filename)
        cfar_map = self.cfar(positive_range_doppler_map)

        # Detect peaks in CFAR map
        detections = np.argwhere(cfar_map > 0)

        # Calculate range, velocity, and angle
        points = []
        for detection in detections:
            doppler_bin, range_bin = detection

            # Calculate range and doppler speed
            range = self.range_res * range_bin
            doppler_speed = self.doppler_res * doppler_bin

            # Calculate angle
            phase_diffs = np.angle(doppler_fft[doppler_bin, range_bin, :])
            azimuth_angle = np.arcsin(phase_diffs[1] / (2 * np.pi * self.dist_antenna))
            elevation_angle = np.arcsin(phase_diffs[2] / (2 * np.pi * self.dist_antenna)) if self.num_antennas > 2 else 0

            # Calculate intensity
            intensity = cfar_map[doppler_bin, range_bin] 
            intensity = 10 * np.log10(intensity) if intensity > 0 else 0 # Convert to dB

            # Convert to Cartesian coordinates
            x = range * np.cos(azimuth_angle) * np.cos(elevation_angle)
            y = range * np.sin(azimuth_angle) * np.cos(elevation_angle)
            z = range * np.sin(elevation_angle)

            points.append(np.array([x, y, z, doppler_speed, intensity]))

        # Convert to numpy array
        if len(points) > 0:
            points = np.stack(points)
            filename = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S.png')
            plot_points(points, filename=filename)
            print(points.shape)
        else:
            points = None
            print('No points detected')

        return points
    
def plot_range_doppler(range_doppler_map, cfar_map, filename='range_doppler.png'):
    if range_doppler_map is None:
        return
    fig = plt.figure()
    ax_rd = fig.add_subplot(121)
    ax_rd.imshow(range_doppler_map, aspect='auto', extent=[0, range_doppler_map.shape[1], 0, range_doppler_map.shape[0]])
    ax_rd.set_xlabel('Range Bins')
    ax_rd.set_ylabel('Doppler Bins')
    ax_rd.set_title('Range-Doppler Map')

    fig.savefig(filename)

def plot_points(points, filename='points.png'):
    if points is None:
        return
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    fig.savefig(filename)

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='localhost', help='Host IP address')
    parser.add_argument('--port', type=int, default=55000, help='Port number')
    parser.add_argument('--num_chirps_per_frame', type=int, default=1, help='Number of chirps per frame')
    parser.add_argument('--num_chirps', type=int, default=60, help='Number of chirps to process')
    parser.add_argument('--num_samples', type=int, default=500, help='Number of samples')
    parser.add_argument('--num_antennas', type=int, default=4, help='Number of antennas')
    parser.add_argument('--center_freq', type=float, default=60.5e9, help='Center frequency')
    parser.add_argument('--sample_freq', type=float, default=300000, help='Sampling frequency')
    parser.add_argument('--bandwidth', type=float, default=7e9, help='Bandwidth')
    parser.add_argument('--dist_antenna', type=float, default=-1, help='Distance between antennas')
    parser.add_argument('--cfar_num_train', type=int, default=16, help='Number of training cells')
    parser.add_argument('--cfar_num_guard', type=int, default=4, help='Number of guard cells')
    parser.add_argument('--cfar_false_alarm_rate', type=float, default=1e-2, help='False alarm rate')
    parser.add_argument('--use_window', type=bool, default=True, help='Use window function')
    parser.add_argument('--save_raw_data', type=bool, default=False, help='Save raw data')
    parser.add_argument('--save_pointcloud', type=bool, default=False, help='Save point cloud')
    parser.add_argument('--vis_range_doppler', type=bool, default=True, help='Visualize range-doppler map')
    parser.add_argument('--vis_pointcloud', type=bool, default=True, help='Visualize point cloud')
    parser.add_argument('--output_dir', type=str, default='ReceivedData', help='Save directory')
    args = parser.parse_args()

    # Initialize TCP server
    tcp_server = TCPServer(args.host, args.port)
    tcp_server.start_server()

    # Initialize radar signal processor
    radar_processor = RadarSignalProcessor(
        num_chirps=args.num_chirps,
        num_samples=args.num_samples,
        num_antennas=args.num_antennas,
        center_freq=args.center_freq,
        sample_freq=args.sample_freq,
        bandwidth=args.bandwidth,
        dist_antenna=args.dist_antenna,
        cfar_num_train=args.cfar_num_train,
        cfar_num_guard=args.cfar_num_guard,
        cfar_false_alarm_rate=args.cfar_false_alarm_rate,
        use_window=args.use_window
    )

    # Receive data
    data_to_process = np.zeros((args.num_chirps, args.num_samples, args.num_antennas))
    i = 0
    idx_frame = 0
    data_dir = f'{args.save_dir}/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(f'{data_dir}/raw', exist_ok=True)
    os.makedirs(f'{data_dir}/pointcloud', exist_ok=True)

    while True:
        data = tcp_server.receive_data(args.num_chirps_per_frame * args.num_samples * args.num_antennas * 4)
        if not data:
            break
        data = np.frombuffer(data, dtype=np.float32)
        data = data.reshape(args.num_chirps_per_frame, args.num_samples, args.num_antennas)
        data_to_process[i:i+args.num_chirps_per_frame, :, :] = data

        i += args.num_chirps_per_frame
        if i == args.num_chirps:
            # Process radar signal
            np.save(f'{data_dir}/{idx_frame:03d}.npy', data_to_process)
            points = radar_processor.process_signal(data_to_process)
            i = 0
            idx_frame += 1

    # Close TCP server
    tcp_server.close_server()

if __name__ == '__main__':
    main()