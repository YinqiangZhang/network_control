import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage


config = {
    "font.family":'serif',
    "font.size": 25,
    "mathtext.fontset":'stix',
    "font.serif": ['SimSun'],
}
rcParams.update(config)

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
colors = [u'#ee854a', u'#4878d0', u'#d65f5f', u'#6acc64']

reduced_axis = 0

def make_frame(t):
    ax.clear()
    X = np.arange(memory_shape[1])
    Y = np.arange(memory_shape[0])
    X, Y = np.meshgrid(X, Y)
    S = np.sum(memory_S[:, :, int(t*fps)], axis=reduced_axis) * 500000
    H = np.sum(memory_H[:, :, int(t*fps)], axis=reduced_axis) * 500000
    I = np.sum(memory_I[:, :, int(t*fps)], axis=reduced_axis) * 500000
    R = np.sum(memory_S[:, :, 0], axis=reduced_axis) * 500000 - S - H - I + 1
    data = np.vstack((R, H, I, S)).transpose()
    data_table = pd.DataFrame(data=data, columns=['R', 'H', 'I', 'S'])
    data_dict = data_table.to_dict('list')
    ax.stackplot(range(S.shape[0]), data_dict.values(), 
             labels=data_dict.keys(), alpha=0.65, colors=colors, edgecolor = 'k', linewidth=0.3)
    plt.yticks(fontproperties = 'Times New Roman', size = stick_font_size)
    plt.xticks(fontproperties = 'Times New Roman', size = stick_font_size)
    plt.xlim([6, S.shape[0]])
    plt.ylim([1.0, 500000])
    ax.set_yscale('log')
    ax.set_yticks([1, 10, 100, 1000, 10000, 100000, 500000])
    ax.spines['bottom'].set_linewidth(1.0)
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['right'].set_linewidth(1.0)
    ax.spines['top'].set_linewidth(1.0)
    ax.grid(linestyle='-.')
    ax.legend(loc='upper right', prop=font_legend)
    if reduced_axis == 1:
        ax.set_xlabel('入度数')
    else:
        ax.set_xlabel('出度数')
    ax.set_ylabel(r'$\log{n}$')
    plt.tight_layout(pad=1.1)
    return mplfig_to_npimage(fig)

degree_type = 'in' if reduced_axis == 1 else 'out'
video_name = 'node_count_{}_degree.gif'.format(degree_type)
animation = VideoClip(make_frame, duration=10)
animation.write_gif(os.path.join(figure_folder, 'video', video_name), fps=fps)