import os 
from tqdm import tqdm
import numpy as np
import pandas as pd 
import seaborn as sns
import time
import matplotlib.pyplot as plt
from matplotlib import rcParams

config = {
    "font.family":'serif',
    "font.size": 25,
    "mathtext.fontset":'stix',
    "font.serif": ['SimSun'],
}
rcParams.update(config)
stick_font_size = 20
font_legend = {'family': 'Times New Roman', 'weight': 'normal', 'size': stick_font_size}

# sns.set_theme()
root_folder = os.path.dirname(os.path.abspath(__file__))
data_folder = os.path.join(root_folder, 'data', 'twitter_network')
figure_folder = os.path.join(root_folder, 'figs')

t1 = time.time()
df = pd.read_csv(os.path.join(data_folder, 'edges.csv'), 
                 header=None, names=['Follower','Target'])
t2 = time.time()
print('Elapsed time [s]: ', np.round(t2-t1, 2))

out_counts = df.Follower.value_counts().rename_axis('Follower').reset_index(name='Frequency')
in_counts = df.Target.value_counts().rename_axis('Target').reset_index(name='Frequency')

fig = plt.figure(figsize=(8, 6))
ax = fig.subplots(1, 1)
ax.hist(np.log10(1 + out_counts.Frequency), 100, alpha=0.7, label='入度', edgecolor='white', linewidth=0.5)
ax.hist(np.log10(1 + in_counts.Frequency), 100, alpha=0.5, label='出度', edgecolor='white', linewidth=0.5)
ax.set_yscale('log')
ax.set_yticks([1, 10, 100, 1000, 10000, 100000, 1000000, 10000000])
ax.spines['bottom'].set_linewidth(1.0)
ax.spines['left'].set_linewidth(1.0)
ax.spines['right'].set_linewidth(1.0)
ax.spines['top'].set_linewidth(1.0)
ax.grid(linestyle='-.')
plt.yticks(fontproperties = 'Times New Roman', size = stick_font_size)
plt.xticks(fontproperties = 'Times New Roman', size = stick_font_size)
ax.legend(loc='upper right', fontsize=stick_font_size)
ax.set_xlabel(r'$\log_{10}(1+$度$)$')
plt.tight_layout(pad=1.1)
plt.xlim([0, 6])
plt.savefig(os.path.join(figure_folder, 'pdf', 'twitter_degree.pdf'), dpi=600, 
                        format='pdf', bbox_inches='tight', pad_inches=0.05)
plt.savefig(os.path.join(figure_folder, 'png', 'twitter_degree.png'), dpi=600, 
                        format='png', bbox_inches='tight', pad_inches=0.05)
plt.close()