import numpy as np
import os
from moviepy.editor import ImageSequenceClip
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

def calc_global_range_doppler_bounds(range_doppler_maps):
    vmin, vmax = float('inf'), float('-inf')
    for map in range_doppler_maps:
        vmin = min(vmin, np.min(map))
        vmax = max(vmax, np.max(map))
    return vmin, vmax

def calc_global_point_cloud_bounds(point_clouds):
    xlim, ylim, zlim = [float('inf'), float('-inf')], [float('inf'), float('-inf')], [float('inf'), float('-inf')]
    for pc in point_clouds:
        xlim[0] = min(xlim[0], np.min(pc[:, 0]))
        xlim[1] = max(xlim[1], np.max(pc[:, 0]))
        ylim[0] = min(ylim[0], np.min(pc[:, 1]))
        ylim[1] = max(ylim[1], np.max(pc[:, 1]))
        zlim[0] = min(zlim[0], np.min(pc[:, 2]))
        zlim[1] = max(zlim[1], np.max(pc[:, 2]))
    return xlim, ylim, zlim

def vis_range_doppler_map(range_doppler_map, output_dir, frame_idx, vmin=None, vmax=None):
    # range_doppler_map: (num_range_bins, num_doppler_bins)
    map_vis = np.fft.fftshift(range_doppler_map, axes=1)
    norm = Normalize(vmin=vmin, vmax=vmax)
    plt.imshow(map_vis, aspect='auto', norm=norm)
    plt.colorbar()
    plt.xlabel('Doppler bins')
    plt.ylabel('Range bins')
    plt.title(f'Range-Doppler map, frame {frame_idx:04d}')
    plt.savefig(os.path.join(output_dir, f'frame_{frame_idx:04d}.png'))
    plt.clf()

def vis_point_cloud(point_cloud, output_dir, frame_idx, xlim=None, ylim=None, zlim=None):
    # point_cloud: (num_points, 5)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], cmap='viridis')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Point cloud, frame {frame_idx:04d}')
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if zlim is not None:
        ax.set_zlim(zlim)
    fig.savefig(os.path.join(output_dir, f'frame_{frame_idx:04d}.png'))
    plt.clf()
    
def make_video(images_dir, output_path, fps=30, verbose=False):
    images = [img for img in os.listdir(images_dir) if img.endswith(".png")]
    images.sort()
    clip = ImageSequenceClip([os.path.join(images_dir, img) for img in images], fps=fps)
    if verbose:
        clip.write_videofile(output_path)
    else:
        clip.write_videofile(output_path, logger=None)