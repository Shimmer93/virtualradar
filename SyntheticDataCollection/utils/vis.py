import numpy as np
import os
from moviepy.editor import ImageSequenceClip
import matplotlib.pyplot as plt

def vis_range_doppler_map(range_doppler_map, output_dir, frame_idx):
    # range_doppler_map: (num_range_bins, num_doppler_bins)
    map_vis = np.fft.fftshift(range_doppler_map, axes=1)
    plt.imshow(map_vis, aspect='auto')
    plt.colorbar()
    plt.xlabel('Doppler bins')
    plt.ylabel('Range bins')
    plt.title(f'Range-Doppler map, frame {frame_idx:04d}')
    plt.savefig(os.path.join(output_dir, f'frame_{frame_idx:04d}.png'))
    plt.clf()

def vis_point_cloud(point_cloud, output_dir, frame_idx):
    # point_cloud: (num_points, 5)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], cmap='viridis')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Point cloud, frame {frame_idx:04d}')
    fig.savefig(os.path.join(output_dir, f'frame_{frame_idx:04d}.png'))
    plt.clf()
    
def make_video(images_dir, output_path, fps=30):
    images = [img for img in os.listdir(images_dir) if img.endswith(".png")]
    images.sort()
    clip = ImageSequenceClip([os.path.join(images_dir, img) for img in images], fps=fps)
    clip.write_videofile(output_path)