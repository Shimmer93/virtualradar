import numpy as np
import os
import shutil
import argparse

from radar.data_processing import RadarSignalProcessor
from utils.vis import vis_range_doppler_map, vis_point_cloud, make_video, calc_global_range_doppler_bounds, calc_global_point_cloud_bounds
from utils.misc import read_cfg, real2IQ, convert2ADC

def main(args):
    # Read data from input directory
    input_dir = args.input_dir
    txs = os.listdir(input_dir)
    txs.sort()
    data_list = []
    for tx in txs:
        data_tx = []
        data_tx_list = os.listdir(os.path.join(input_dir, tx))
        data_tx_list.sort()
        for fn in data_tx_list:
            data = np.load(os.path.join(input_dir, tx, fn))
            data_tx.append(data)
        data_tx = np.stack(data_tx, axis=0)
        data_list.append(data_tx)
    data = np.concatenate(data_list, axis=2)

    data = convert2ADC(data, resolution=16, is_complex=args.complex)

    # Create save directory
    save_dir = input_dir.replace('raw_data', 'point_clouds')
    os.makedirs(save_dir, exist_ok=True)

    # Create temporary directory for visualization
    temp_dir_rd = f'{save_dir}/temp_rd'
    temp_dir_pc = f'{save_dir}/temp_pc'
    if os.path.exists(temp_dir_rd):
        shutil.rmtree(temp_dir_rd)
    if os.path.exists(temp_dir_pc):
        shutil.rmtree(temp_dir_pc)
    os.makedirs(temp_dir_rd)
    os.makedirs(temp_dir_pc)

    # Initialize radar signal processor
    cfg = read_cfg(args.cfg_path)
    radar_processor = RadarSignalProcessor(**cfg)
    point_clouds = []
    range_doppler_maps = []
    
    # Process data frame by frame
    for frame in data:
        range_doppler_map, point_cloud = radar_processor.process_frame(frame)
        range_doppler_maps.append(range_doppler_map)
        point_clouds.append(point_cloud)

    # Visualize range-doppler maps and point clouds
    vmin, vmax = calc_global_range_doppler_bounds(range_doppler_maps)
    xlim, ylim, zlim = calc_global_point_cloud_bounds(point_clouds)

    for idx, (range_doppler_map, point_cloud) in enumerate(zip(range_doppler_maps, point_clouds)):
        vis_range_doppler_map(range_doppler_map, temp_dir_rd, idx, vmin=vmin, vmax=vmax)
        vis_point_cloud(point_cloud, temp_dir_pc, idx, xlim=xlim, ylim=ylim, zlim=zlim)

    # Make video
    make_video(temp_dir_rd, f'{save_dir}/range_doppler.mp4', verbose=args.verbose)
    make_video(temp_dir_pc, f'{save_dir}/point_clouds.mp4', verbose=args.verbose)

    # Save point clouds
    np.save(f'{save_dir}/point_clouds.npy', point_clouds)

    # Remove temporary directories
    shutil.rmtree(temp_dir_rd)
    shutil.rmtree(temp_dir_pc)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process radar data')
    parser.add_argument('-i', '--input_dir', type=str, default='data/test/raw_data', help='Directory containing raw data')
    parser.add_argument('-c', '--cfg_path', type=str, default='cfg/ti_xwr1843.yml', help='Path to configuration file')
    parser.add_argument('-v', '--verbose', action='store_true', help='Print debug messages')
    parser.add_argument('--complex', action='store_true', help='Use complex data')
    args = parser.parse_args()

    main(args)
