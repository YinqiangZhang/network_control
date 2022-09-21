import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
import seaborn as sns
from matplotlib import rcParams

config = {
    "font.family":'serif',
    "font.size": 25,
    "mathtext.fontset":'stix',
    "font.serif": ['SimSun'],
}
rcParams.update(config)
stick_font_size = 20
figsize=(8, 6)
font_legend = {'family': 'Times New Roman', 'weight': 'normal', 'size': stick_font_size}

root_folder = os.path.dirname(os.path.abspath(__file__))
figure_folder = os.path.join(root_folder, 'figs')

with open('dynamics.pkl', 'rb') as f:
    data_dict = pickle.load(f)

language = 'chinese' # 'english' 
memory_shape = data_dict['memory_S'].shape
memory_S = data_dict['memory_S']
memory_I = data_dict['memory_I']
memory_H = data_dict['memory_H']
memory_R = data_dict['memory_R']

theta_S = np.array(data_dict['memory_theta_S'])
theta_I = np.array(data_dict['memory_theta_I'])
theta_H = np.array(data_dict['memory_theta_H'])
theta_R = np.array(data_dict['memory_theta_R'])

S = np.sum(memory_S, axis=(0,1))
H =  np.sum(memory_H, axis=(0,1))
I =  np.sum(memory_I, axis=(0,1))
R =  1 - S - H - I

node_data = np.vstack((R, H, I, S)).transpose()
theta_data = np.vstack((theta_R, theta_H, theta_I, theta_S)).transpose()
node_data_table = pd.DataFrame(data=node_data, columns=['R', 'H', 'I', 'S'])
edge_data_table = pd.DataFrame(data=theta_data, columns=['R', 'H', 'I', 'S'])

node_data_dict = node_data_table.to_dict('list')
edge_data_dict = edge_data_table.to_dict('list')
colors = [u'#ee854a', u'#4878d0', u'#d65f5f', u'#6acc64']

# node state
fig, ax = plt.subplots(1, 1, figsize=figsize)
ax.stackplot(range(memory_shape[-1]), node_data_dict.values(), 
             labels=node_data_dict.keys(), alpha=0.8, colors=colors, edgecolor = 'k', linewidth=1.0)
plt.yticks(fontproperties = 'Times New Roman', size = stick_font_size)
plt.xticks(fontproperties = 'Times New Roman', size = stick_font_size)
ax.spines['bottom'].set_linewidth(1.0)
ax.spines['left'].set_linewidth(1.0)
ax.spines['right'].set_linewidth(1.0)
ax.spines['top'].set_linewidth(1.0)
plt.ylim([0.0, 1.0])
plt.xlim([0.0, 1000])
ax.legend(loc='upper right', prop=font_legend)
ax.set_xlabel(r'时间 $t$')
ax.set_ylabel('比率')
plt.savefig(os.path.join(figure_folder, 'pdf', 'stacked_node_state.pdf'), dpi=600, 
                        format='pdf', bbox_inches='tight', pad_inches=0.05)
plt.savefig(os.path.join(figure_folder, 'png', 'stacked_node_state.png'), dpi=600, 
                        format='png', bbox_inches='tight', pad_inches=0.05)
plt.close()

# edge state
fig, ax = plt.subplots(1, 1, figsize=figsize)
ax.stackplot(range(memory_shape[-1]), edge_data_dict.values(), 
             labels=[r'$\theta_R$',r"$\theta_H$",r"$\theta_I$",r"$\theta_S$"], 
             alpha=0.8, colors=colors, edgecolor = 'k', linewidth=1.0)
plt.yticks(fontproperties = 'Times New Roman', size = stick_font_size)
plt.xticks(fontproperties = 'Times New Roman', size = stick_font_size)
ax.spines['bottom'].set_linewidth(1.0)
ax.spines['left'].set_linewidth(1.0)
ax.spines['right'].set_linewidth(1.0)
ax.spines['top'].set_linewidth(1.0)
plt.ylim([0.0, 1.0])
plt.xlim([0.0, 1000])
ax.legend(loc='upper right', prop=font_legend)
ax.set_xlabel(r'时间 $t$')
ax.set_ylabel('比率')
plt.savefig(os.path.join(figure_folder, 'pdf', 'stacked_edge_state.pdf'), dpi=600, 
                        format='pdf', bbox_inches='tight', pad_inches=0.05)
plt.savefig(os.path.join(figure_folder, 'png', 'stacked_edge_state.png'), dpi=600, 
                        format='png', bbox_inches='tight', pad_inches=0.05)
plt.close()
