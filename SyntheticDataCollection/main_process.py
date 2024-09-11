import numpy as np
import os
import datetime
import argparse

from radar.data_processing import RadarSignalProcessor
from utils.vis import vis_range_doppler_map, vis_point_cloud, make_video
from utils.misc import read_cfg

def main(args):
    # Read data from input directory
    input_dir = args.input_dir
    txs = os.listdir(input_dir)
    data_list = []
    for tx in txs:
        data_tx_list = os.listdir(os.path.join(input_dir, tx))
        data_tx_list = [np.load(data_tx) for data_tx in data_tx_list]
        data_tx = np.stack(data_tx_list, axis=0)
        data_list.append(data_tx)
    data = np.sum(np.stack(data_list), axis=0)

    # Create save directory
    save_dir = input_dir.replace('raw', 'point_cloud')
    os.makedirs(save_dir, exist_ok=True)

    # Create temporary directory for visualization
    temp_dir_rd = f'{input_dir}/temp_rd'
    temp_dir_pc = f'{input_dir}/temp_pc'
    if os.exists(temp_dir_rd):
        os.removedirs(temp_dir_rd)
    if os.exists(temp_dir_pc):
        os.removedirs(temp_dir_pc)
    os.makedirs(temp_dir_rd)
    os.makedirs(temp_dir_pc)

    # Initialize radar signal processor
    cfg = read_cfg(args.cfg_path)
    radar_processor = RadarSignalProcessor(**cfg)
    point_clouds = []
    range_doppler_maps = []
    idx_frame = 0
    
    for frame in data:
        # Process data frame by frame
        range_doppler_map, point_cloud = radar_processor.process_frame(frame)
        range_doppler_maps.append(range_doppler_map)
        point_clouds.append(point_cloud)

        # Visualize range-doppler map
        vis_range_doppler_map(range_doppler_map, temp_dir_rd, idx_frame)
        vis_point_cloud(point_cloud, temp_dir_pc, idx_frame)
        
        idx_frame += 1

    # Make video
    make_video(temp_dir_rd, f'{save_dir}/range_doppler.mp4')
    make_video(temp_dir_pc, f'{save_dir}/point_cloud.mp4')

    # Save point clouds
    np.save(f'{save_dir}/point_cloud.npy', point_clouds)

    # Remove temporary directories
    os.removedirs(temp_dir_rd)
    os.removedirs(temp_dir_pc)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process radar data')
    parser.add_argument('--input_dir', type=str, default='data/raw', help='Directory containing raw data')
    parser.add_argument('--cfg_path', type=str, default='cfg/ti_xwr1843.yml', help='Path to configuration file')
