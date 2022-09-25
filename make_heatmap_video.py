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
import seaborn as sns

config = {
    "font.family":'serif',
    "font.size": 25,
    "mathtext.fontset":'stix',
    "font.serif": ['SimSun'],
}
rcParams.update(config)

# figsize = (8, 6)
# fig = plt.figure(figsize=figsize)
# ax = fig.subplots(1,1)
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

def make_frame(t, ax, state):
    ax.clear()
    use_all_degree = False
    clip_range = 100
    if use_all_degree is not True:
        X = np.arange(clip_range)
        Y = np.arange(clip_range)
        X, Y = np.meshgrid(X, Y)
        base = memory_S[:clip_range, :clip_range, 0]
        node_num = 500000
        S = memory_S[:clip_range, :clip_range, int(t*fps)] 
        H = memory_H[:clip_range, :clip_range, int(t*fps)] 
        I = memory_I[:clip_range, :clip_range, int(t*fps)] 
        R = base - S - H - I
        R[R<0] = 0.0
        R = R[:clip_range, :clip_range]
    else:
        X = np.arange(memory_S.shape[1])
        Y = np.arange(memory_S.shape[0])
        X, Y = np.meshgrid(X, Y)
        base = memory_S[:, :, 0]
        node_num = 500000
        S = memory_S[:, :, int(t*fps)] 
        H = memory_H[:, :, int(t*fps)] 
        I = memory_I[:, :, int(t*fps)] 
        R = base - S - H - I
        R[R<0] = 0.0
    # levels = np.arange(0, 1.3, step=0.1)
    norm = mcolors.LogNorm(vmin=1, vmax=1000)
    if state == 0:
        # S_ratio = np.divide(S, base, out=np.zeros_like(base), where=np.not_equal(base, 0))
        # ax.contourf(X, Y, S_ratio, levels=levels, cmap=cm.Greens, alpha=1.0, edgecolor='k', linewidth=0.1)
        ax.pcolormesh(X, Y, S*node_num+1, cmap=cm.jet, alpha = 1.0, norm=norm, rasterized=True)
    elif state == 1:
        # H_ratio = np.divide(H, base, out=np.zeros_like(base), where=np.not_equal(base, 0))
        # ax.contourf(X, Y, H_ratio, levels=levels, cmap=cm.Blues, alpha=1.0, edgecolor='k', linewidth=0.1)
        ax.pcolormesh(X, Y, H*node_num+1, cmap=cm.jet, alpha=1.0, norm=norm, rasterized=True)
    elif state == 2:
        # I_ratio = np.divide(I, base, out=np.zeros_like(base), where=np.not_equal(base, 0))
        # ax.contourf(X, Y, I_ratio, levels=levels, cmap=cm.Reds, alpha=1.0, edgecolor='k', linewidth=0.1)
        ax.pcolormesh(X, Y, I*node_num+1, cmap=cm.jet, alpha=1.0, norm=norm, rasterized=True)
    else:
        # R_ratio = np.divide(R, base, out=np.zeros_like(base), where=np.not_equal(base, 0))
        # ax.contourf(X, Y, R_ratio, levels=levels, cmap=cm.Oranges, alpha=1.0, edgecolor='k', linewidth=0.1)
        ax.pcolormesh(X, Y, R*node_num+1, cmap=cm.jet, alpha=1.0, norm=norm, rasterized=True)

    # ax.legend(loc='upper right')
    ax.spines['bottom'].set_linewidth(1.0)
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['right'].set_linewidth(1.0)
    ax.spines['top'].set_linewidth(1.0)
    plt.yticks(fontproperties = 'Times New Roman', size = stick_font_size)
    plt.xticks(fontproperties = 'Times New Roman', size = stick_font_size)
    plt.ylim([0.0, S.shape[0]-1])
    plt.xlim([0.0, S.shape[1]-1])
    ax.set_xlabel('出度数')
    ax.set_ylabel('入度数')
    plt.tight_layout(pad=1.1)
    return mplfig_to_npimage(fig)
    
# animation = VideoClip(make_frame, duration=10)
# animation.write_gif(os.path.join(figure_folder, 'video', 'S_heatmap.gif'), fps=fps)


if __name__ == "__main__":
    figsize = (8, 6)
    fig = plt.figure(figsize=figsize)
    state_indices = ['S', 'H', 'I', 'R']
    state_types = {'S': [0, 0.5, 1, 2, 8], 
                  'H': [0, 0.025, 0.5, 2, 4, 6],
                  'I': [0, 1, 4, 10, 14],
                  'R': [0, 1, 4, 10]}
    for state_type, time_steps in state_types.items():
        for time_step in time_steps:
            fig = plt.figure(figsize=(8, 6))
            ax = fig.subplots(1, 1)
            make_frame(time_step, ax=ax, state=state_indices.index(state_type))
            plt.savefig(os.path.join(figure_folder, 'pdf', 
                                     'heatmap_{}_{}.pdf'.format(state_type,time_step)), dpi=600, 
                            format='pdf', bbox_inches='tight', pad_inches=0.05)
            plt.savefig(os.path.join(figure_folder, 'png', 
                                     'heatmap_{}_{}.png'.format(state_type,time_step)), dpi=600, 
                            format='png', bbox_inches='tight', pad_inches=0.05)
            plt.close()
        


