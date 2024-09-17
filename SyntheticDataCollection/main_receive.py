import numpy as np
import os
import datetime
import argparse
import struct

from utils.tcpip import TCPServer
from utils.misc import real2IQ, read_cfg

def main(args):
    # Read config file
    cfg = read_cfg(args.cfg_path, mode='namespace')

    # Initialize TCP server
    tcp_server = TCPServer(args.host, args.port, verbose=args.verbose)
    tcp_server.start_server()

    # Receive save directory, run name, and transmitter index from the client
    save_dir = tcp_server.receive_data_flexible().decode('utf-8')
    run_name = tcp_server.receive_data_flexible().decode('utf-8')
    idx_tx = int.from_bytes(tcp_server.receive_data(4), byteorder='little')
    # idx_tx is a 4-byte integer

    print(idx_tx)
    print(save_dir)
    print(run_name)

    # Create directory to save data
    data_dir = os.path.join(save_dir.replace('/', os.sep), run_name, 'raw', str(idx_tx))
    if args.verbose:
        print(f'Saving data to {data_dir}')
    os.makedirs(data_dir, exist_ok=True)

    # Initialize data cube
    data_cube = np.zeros((cfg.num_chirps, cfg.num_rx, cfg.num_samples), dtype=np.complex64)

    # Receive data
    cur_length = 0
    idx_frame = 0
    while True:
        data_slice = tcp_server.receive_data(args.num_chirps_per_chunk * cfg.num_samples * cfg.num_rx * 4)
        if data_slice is None:
            break

        data_slice = np.frombuffer(data_slice, dtype=np.float32)
        data_slice = data_slice.reshape(args.num_chirps_per_chunk, cfg.num_samples, cfg.num_rx).transpose(0, 2, 1)
        data_slice = real2IQ(data_slice)
        data_cube[cur_length:cur_length + args.num_chirps_per_chunk] = data_slice

        cur_length += args.num_chirps_per_chunk
        if cur_length >= cfg.num_chirps:
            np.save(os.path.join(data_dir, f'data_{idx_frame:04d}.npy'), data_cube)
            cur_length = 0
            idx_frame += 1

            if args.total_frames > 0 and idx_frame >= args.total_frames:
                break

    # Close TCP server
    tcp_server.stop_server()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Receive radar data')
    parser.add_argument('--host', type=str, default='localhost', help='Host IP address')
    parser.add_argument('--port', type=int, default=55000, help='Port number')
    parser.add_argument('--cfg_path', type=str, default='cfg/ti_xwr1843.yml', help='Path to configuration file')
    parser.add_argument('--num_chirps_per_chunk', type=int, default=1, help='Number of chirps per chunk')
    parser.add_argument('--total_frames', type=int, default=-1, help='Total number of frames to receive')
    parser.add_argument('--idx_tx', type=int, default=0, help='Index of the transmitter')
    parser.add_argument('--save_dir', type=str, default='data', help='Directory to save data')
    parser.add_argument('--run_name', type=str, default='test', help='Name of this run')
    parser.add_argument('--verbose', action='store_true', help='Print debug messages')
    args = parser.parse_args()

    main(args)

