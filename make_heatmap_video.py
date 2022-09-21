import os
import pickle
import moviepy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import cm
import matplotlib.colors as mcolors
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage

config = {
    "font.family":'serif',
    "font.size": 25,
    "mathtext.fontset":'stix',
    "font.serif": ['SimSun'],
}
rcParams.update(config)

use_chinese = True
figsize = (8, 6)
fig = plt.figure(figsize=figsize)
ax = fig.subplots(1,1)
fps = 40 
stick_font_size = 20
font_legend = {'family': 'Times New Roman', 'weight': 'normal', 'size': stick_font_size}

root_folder = os.path.dirname(os.path.abspath(__file__))
figure_folder = os.path.join(root_folder, 'figs')

with open(os.path.join(root_folder, 'dynamics.pkl'), 'rb') as f:
    data_dict = pickle.load(f)
    
memory_shape = data_dict['memory_S'].shape
memory_S = data_dict['memory_S']
memory_I = data_dict['memory_I']
memory_H = data_dict['memory_H']
memory_R = data_dict['memory_R']

def make_frame(t):
    language = 'chinese' if use_chinese is True else 'english'
    ax.clear()
    X = np.arange(memory_shape[1])
    Y = np.arange(memory_shape[0])
    X, Y = np.meshgrid(X, Y)
    S = memory_S[:, :, int(t*fps)] * 500000 + 1
    H = memory_H[:, :, int(t*fps)] * 500000 + 1
    I = memory_I[:, :, int(t*fps)] * 500000 + 1
    R = memory_R[:, :, int(t*fps)] * 500000 + 1
    norm = mcolors.LogNorm(vmin=1, vmax=1000)
    # ax.pcolormesh(X, Y, S, cmap=cm.coolwarm, alpha = 1.0,
    #             norm=norm, 
    #             shading='nearest', 
    #             rasterized=True)
    # ax.pcolormesh(X, Y, H, cmap=cm.coolwarm, alpha=1.0,
    #             norm=norm, 
    #             shading='nearest', 
    #             rasterized=True)
    # ax.pcolormesh(X, Y, I, cmap=cm.coolwarm, alpha=1.0,
    #             norm=norm, 
    #             shading='nearest', 
    #             rasterized=True)
    ax.pcolormesh(X, Y, R, cmap=cm.coolwarm, alpha=1.0,
                norm=norm, 
                shading='nearest', 
                rasterized=True)
    ax.spines['bottom'].set_linewidth(1.0)
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['right'].set_linewidth(1.0)
    ax.spines['top'].set_linewidth(1.0)
    plt.yticks(fontproperties = 'Times New Roman', size = stick_font_size)
    plt.xticks(fontproperties = 'Times New Roman', size = stick_font_size)
    plt.ylim([0.0, S.shape[0]-1])
    plt.xlim([0.0, S.shape[1]-1])
    if language == 'chinese':
        ax.set_xlabel('入度数')
        ax.set_ylabel('出度数')
        # ax.set_title('仿真步长：{}'.format(int(t*fps)))
    else:
        ax.set_xlabel('In-degree')
        ax.set_ylabel('Out-degree')   
        # ax.set_title('Time step {}'.format(int(t*fps)))
    plt.tight_layout()
    return mplfig_to_npimage(fig)
    
animation = VideoClip(make_frame, duration=10)
animation.write_gif(os.path.join(figure_folder, 'video', 'R_heatmap.gif'), fps=fps)
