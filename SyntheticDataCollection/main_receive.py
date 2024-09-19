import numpy as np
import os
import argparse

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
    num_total_chunks = int.from_bytes(tcp_server.receive_data(4), byteorder='little')
    num_chirps_per_chunk = int.from_bytes(tcp_server.receive_data(4), byteorder='little')
    num_total_frames = num_total_chunks // (cfg.num_chirps // num_chirps_per_chunk)

    # Create directory to save data
    data_dir = os.path.join(save_dir, run_name, 'raw_data', str(idx_tx))
    if args.verbose:
        print(f'Saving data to {data_dir}')
    os.makedirs(data_dir, exist_ok=True)

    # Initialize data cube
    data_cube = np.zeros((cfg.num_chirps, cfg.num_rx, cfg.num_samples), dtype=np.complex64)

    # Receive data
    cur_length = 0
    idx_frame = 0
    while True:
        data_slice = tcp_server.receive_data(num_chirps_per_chunk * cfg.num_samples * cfg.num_rx * 4)

        if data_slice is None or len(data_slice) == 0:
            if args.verbose:
                print('No more data received')
            break
        assert len(data_slice) == num_chirps_per_chunk * cfg.num_samples * cfg.num_rx * 4, 'Data length mismatch'

        data_slice = np.frombuffer(data_slice, dtype=np.float32)
        data_slice = data_slice.reshape(num_chirps_per_chunk, cfg.num_samples, cfg.num_rx).transpose(0, 2, 1)
        data_slice = real2IQ(data_slice)
        data_cube[cur_length:cur_length + num_chirps_per_chunk] = data_slice

        cur_length += num_chirps_per_chunk
        if cur_length >= cfg.num_chirps:
            np.save(os.path.join(data_dir, f'data_{idx_frame:04d}.npy'), data_cube)
            cur_length = 0
            idx_frame += 1

            if idx_frame >= num_total_frames:
                if args.verbose:
                    print(f'Received all {num_total_frames} frames')
                break

    # Close TCP server
    tcp_server.stop_server()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Receive radar data')
    parser.add_argument('-h', '--host', type=str, default='localhost', help='Host IP address')
    parser.add_argument('-p', '--port', type=int, default=55000, help='Port number')
    parser.add_argument('-c', '--cfg_path', type=str, default='cfg/ti_xwr1843.yml', help='Path to configuration file')
    parser.add_argument('-v', '--verbose', action='store_true', help='Print debug messages')
    args = parser.parse_args()

    main(args)

