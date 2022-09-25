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

# moving average kernel
kernel_size = 20
kernel = np.ones(kernel_size) / kernel_size

def make_frame(t, ax, reduced_axis):
    ax.clear()
    X = np.arange(memory_shape[1])
    Y = np.arange(memory_shape[0])
    X, Y = np.meshgrid(X, Y)
    base_ratio = np.sum(memory_S[:, :, 0], axis=reduced_axis)
    S_ratio = np.sum(memory_S[:, :, int(t*fps)], axis=reduced_axis)
    H_ratio = np.sum(memory_H[:, :, int(t*fps)], axis=reduced_axis)
    I_ratio = np.sum(memory_I[:, :, int(t*fps)], axis=reduced_axis)
    R_ratio = base_ratio - S_ratio - H_ratio - I_ratio
    
    base_ratio = np.convolve(base_ratio, kernel, mode='same')
    S_ratio = np.convolve(S_ratio, kernel, mode='same')
    H_ratio = np.convolve(H_ratio, kernel, mode='same')
    I_ratio = np.convolve(I_ratio, kernel, mode='same')
    R_ratio = np.convolve(R_ratio, kernel, mode='same')
    
    R_ratio[R_ratio<=0.0] = 0.0
    
    S = np.divide(S_ratio, base_ratio, out=np.zeros_like(base_ratio), where=np.not_equal(base_ratio, 0))
    H = np.divide(H_ratio, base_ratio, out=np.zeros_like(base_ratio), where=np.not_equal(base_ratio, 0))
    I = np.divide(I_ratio, base_ratio, out=np.zeros_like(base_ratio), where=np.not_equal(base_ratio, 0))
    R = np.divide(R_ratio, base_ratio, out=np.zeros_like(base_ratio), where=np.not_equal(base_ratio, 0))
    data = np.vstack((R, H, I, S)).transpose()
    data_table = pd.DataFrame(data=data, columns=['R', 'H', 'I', 'S'])
    data_dict = data_table.to_dict('list')
    ax.stackplot(range(S.shape[0]), data_dict.values(), 
             labels=data_dict.keys(), alpha=0.65, colors=colors, edgecolor = 'k', linewidth=0.4)
    plt.yticks(fontproperties = 'Times New Roman', size = stick_font_size)
    plt.xticks(fontproperties = 'Times New Roman', size = stick_font_size)
    plt.xlim([kernel_size-15, S.shape[0]])
    plt.ylim([0, 1])
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
    ax.set_ylabel('比率')
    plt.tight_layout(pad=1.1)
    return mplfig_to_npimage(fig)

# degree_type = 'in' if reduced_axis == 1 else 'out'
# video_name = 'node_ratio_{}_degree.gif'.format(degree_type)
# animation = VideoClip(make_frame, duration=10)
# animation.write_gif(os.path.join(figure_folder, 'video', video_name), fps=fps)


if __name__ == "__main__":
    figsize = (8, 6)
    reduced_axis = 0
    degree_type = 'in' if reduced_axis == 1 else 'out'
    fig = plt.figure(figsize=figsize)
    time_steps = [0, 0.5, 1, 2, 4, 8]
    for time_step in time_steps:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.subplots(1, 1)
        make_frame(time_step, ax, reduced_axis)
        plt.savefig(os.path.join(figure_folder, 'pdf', 
                                    'degree_ratio_{}_{}.pdf'.format(degree_type, time_step)), dpi=600, 
                        format='pdf', bbox_inches='tight', pad_inches=0.05)
        plt.savefig(os.path.join(figure_folder, 'png', 
                                    'degree_ratio_{}_{}.png'.format(degree_type, time_step)), dpi=600, 
                        format='png', bbox_inches='tight', pad_inches=0.05)
        plt.close()