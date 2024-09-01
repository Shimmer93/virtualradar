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

class TCPServer:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.tcp_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.conn = None
        self.addr = None

    def start_server(self):
        self.tcp_server.bind((self.host, self.port))
        self.tcp_server.listen(1)
        self.conn, self.addr = self.tcp_server.accept()
        print('Connected by', self.addr)
        
    def receive_data(self, datalengthbyte):
        data = self.conn.recv(datalengthbyte)
        self.conn.settimeout(60)
        return data
    
    def close_server(self):
        self.conn.close()
        self.tcp_server.close()
    
class RadarSignalProcessor:
    def __init__(self, 
                 num_chirps=1, 
                 num_samples=300, 
                 num_antennas=4,
                 center_freq=60.5e9, 
                 sample_freq=2e6, 
                 bandwidth=7e9,
                 dist_antenna=-1,
                 cfar_num_train=16,
                 cfar_num_guard=4,
                 cfar_false_alarm_rate=1e-6,
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
        self.lower_freq = self.center_freq - self.bandwidth / 2
        self.range_res = c / (2 * self.bandwidth)
        self.doppler_res = 1 / (self.num_chirps * self.dur_chirp)

        self.cfar_num_train = cfar_num_train
        self.cfar_num_guard = cfar_num_guard
        self.cfar_threshold = cfar_num_train * (cfar_false_alarm_rate ** (-1 / cfar_num_train) - 1)

    def cfar_algorithm(self, data):
        assert data.shape[1] > self.cfar_num_train + self.cfar_num_guard, 'Signal length is too short'

        cfar_map = np.zeros_like(data)
        for chirp in range(data.shape[0]):
            for antenna in range(data.shape[2]):
                for i in range(self.cfar_num_train + self.cfar_num_guard, data.shape[1] - self.cfar_num_train - self.cfar_num_guard):
                    training_cells = np.concatenate([data[chirp, i - self.cfar_num_train - self.cfar_num_guard:i - self.cfar_num_guard, antenna], data[chirp, i + self.cfar_num_guard + 1:i + self.cfar_num_guard + self.cfar_num_train + 1, antenna]])
        noise_level = np.mean(training_cells)
        threshold = self.cfar_threshold * noise_level
        if data[i] > threshold:
            cfar_map[i] = data[i]

        return cfar_map
    
    def process_signal(self, data):
        # data: num_chirps x num_samples x num_antennas
        assert data.shape == (self.num_chirps, self.num_samples, self.num_antennas), 'Data shape is incorrect'

        # Apply FFT to convert time domain to frequency domain
        range_fft = np.fft.fft(data, axis=1)
        doppler_fft = np.fft.fft(range_fft, axis=0)

        # Apply CFAR to range-doppler map
        cfar_map = self.cfar_algorithm(np.abs(doppler_fft))

        # Detect peaks in CFAR map
        detections = np.argwhere(cfar_map > 0)

        # Calculate range, velocity, and angle
        points = []
        for detection in detections:
            doppler_bin, range_bin, antenna = detection

            # Calculate range and doppler speed
            range = self.range_res * range_bin
            doppler_speed = self.doppler_res * doppler_bin

            # Calculate angle
            phase_diffs = np.angle(doppler_fft[doppler_bin, range_bin, :])
            azimuth_angle = np.arcsin(phase_diffs[1] / (2 * np.pi * self.dist_antenna))
            elevation_angle = np.arcsin(phase_diffs[2] / (2 * np.pi * self.dist_antenna)) if self.num_antennas > 2 else 0

            # Calculate intensity
            intensity = cfar_map[doppler_bin, range_bin, antenna] 
            intensity = 10 * np.log10(intensity) if intensity > 0 else 0 # Convert to dB

            # Convert to Cartesian coordinates
            x = range * np.cos(azimuth_angle) * np.cos(elevation_angle)
            y = range * np.sin(azimuth_angle) * np.cos(elevation_angle)
            z = range * np.sin(elevation_angle)

            points.append([x, y, z, doppler_speed, intensity])

        # Convert to numpy array
        points = np.array(points)

        return points
    
def plot_points(points, filename='points.png'):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 4], cmap='viridis')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.savefig(filename)

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='localhost', help='Host IP address')
    parser.add_argument('--port', type=int, default=55000, help='Port number')
    parser.add_argument('--num_chirps', type=int, default=1, help='Number of chirps')
    parser.add_argument('--num_samples', type=int, default=300, help='Number of samples')
    parser.add_argument('--num_antennas', type=int, default=4, help='Number of antennas')
    parser.add_argument('--center_freq', type=float, default=60.5e9, help='Center frequency')
    parser.add_argument('--sample_freq', type=float, default=2e6, help='Sampling frequency')
    parser.add_argument('--bandwidth', type=float, default=7e9, help='Bandwidth')
    parser.add_argument('--dist_antenna', type=float, default=-1, help='Distance between antennas')
    parser.add_argument('--cfar_num_train', type=int, default=16, help='Number of training cells')
    parser.add_argument('--cfar_num_guard', type=int, default=4, help='Number of guard cells')
    parser.add_argument('--cfar_false_alarm_rate', type=float, default=1e-6, help='False alarm rate')
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
    )

    # Receive data
    while True:
        data = tcp_server.receive_data(args.num_chirps * args.num_samples * args.num_antennas * 4)
        if not data:
            break
        data = np.frombuffer(data, dtype=np.float32)
        data = data.reshape(args.num_chirps, args.num_samples, args.num_antennas)

        # Process radar signal
        points = radar_processor.process_signal(data)
        filename = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S.png')
        plot_points(points, filename=filename)

    # Close TCP server
    tcp_server.close_server()

if __name__ == '__main__':
    main()