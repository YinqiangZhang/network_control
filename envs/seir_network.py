import os
import pickle
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd

config = {
    "font.family":'serif',
    "font.size": 25,
    "mathtext.fontset":'stix',
    "font.serif": ['SimSun'],
}
rcParams.update(config)
stick_font_size = 20
font_legend = {'family': 'Times New Roman', 'weight': 'normal', 'size': stick_font_size}
colors = [u'#ee854a', u'#4878d0', u'#d65f5f', u'#6acc64']


class SEIR_Network():
    def __init__(self, save_path):
        self.save_path = save_path
        # variables
        self.N = 1
        self.I = 0.001 * self.N
        self.E = 0
        self.R = 0
        self.S = self.N - self.I
        self.figsize = (8, 6)
        # parameters
        self.delta_t = 1
        self.T = 1000
        self.step_num = int(self.T/self.delta_t)
        self.r = 0.1
        self.beta = 0.02
        self.gamma = 0.01
        # variable memory
        self.memory_S = np.zeros((self.step_num, 1))
        self.memory_E = np.zeros((self.step_num, 1))
        self.memory_I = np.zeros((self.step_num, 1))
        self.memory_R = np.zeros((self.step_num, 1))
    
    def reset(self):
        self.time_step = 0
        self.memory_S[self.time_step] = self.S
        self.memory_E[self.time_step] = self.E
        self.memory_I[self.time_step] = self.I
        self.memory_R[self.time_step] = self.R
        
    def step(self):
        self.time_step += 1
        print('Step: {}'.format(self.time_step))
        delta_S = -self.r * self.I * self.S / self.N
        delta_E = self.r * self.I * self.S / self.N - self.beta * self.E
        delta_I = self.beta * self.E - self.gamma * self.I
        delta_R = self.gamma * self.I
        
        self.S += delta_S * self.delta_t
        self.E += delta_E * self.delta_t
        self.I += delta_I * self.delta_t
        self.R += delta_R * self.delta_t
        
        self.memory_S[self.time_step] = self.S
        self.memory_E[self.time_step] = self.E
        self.memory_I[self.time_step] = self.I
        self.memory_R[self.time_step] = self.R
        
        return True if self.time_step >= self.step_num - 1 else False
    
    def curve_render(self):
        fig = plt.figure(figsize=self.figsize)
        ax = fig.subplots(1, 1)
        t = np.arange(0, self.step_num)
        plt.plot(t, self.memory_S, color=colors[3], linewidth=2, label='S')
        plt.plot(t, self.memory_E, color=colors[1], linewidth=2, label='H')
        plt.plot(t, self.memory_I, color=colors[2], linewidth=2, label='I')
        plt.plot(t, self.memory_R, color=colors[0], linewidth=2, label='R')
        ax.spines['bottom'].set_linewidth(1.0)
        ax.spines['left'].set_linewidth(1.0)
        ax.spines['right'].set_linewidth(1.0)
        ax.spines['top'].set_linewidth(1.0)
        plt.legend(prop=font_legend, loc = 'upper right')
        plt.yticks(fontproperties = 'Times New Roman', size = stick_font_size)
        plt.xticks(fontproperties = 'Times New Roman', size = stick_font_size)
        plt.grid(linestyle='-.')
        plt.xlim([0.0, self.step_num])
        plt.ylabel("比率")
        plt.xlabel(r"时间 $t$")
        plt.ylim([0.0, 1.0])
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, 'png', 'curve_homo_state.png'), dpi=600, 
                    format='png', bbox_inches='tight', pad_inches=0.05)
        plt.savefig(os.path.join(self.save_path, 'pdf', 'curve_homo_state.pdf'), dpi=600, 
                    format='pdf', bbox_inches='tight', pad_inches=0.05)
        plt.close()
    
    def stack_render(self):
        fig = plt.figure(figsize=self.figsize)
        ax = fig.subplots(1, 1)
        
        data = np.hstack((self.memory_R, 
                          self.memory_E, 
                          self.memory_I, 
                          self.memory_S))
        data_table = pd.DataFrame(data=data, columns=['R', 'H', 'I', 'S'])
        data_dict = data_table.to_dict('list')
        ax.stackplot(range(self.memory_S.shape[0]), data_dict.values(), 
             labels=data_dict.keys(), alpha=0.8, colors=colors, edgecolor = 'k', linewidth=1.0)
        
        ax.spines['bottom'].set_linewidth(1.0)
        ax.spines['left'].set_linewidth(1.0)
        ax.spines['right'].set_linewidth(1.0)
        ax.spines['top'].set_linewidth(1.0)
        plt.yticks(fontproperties = 'Times New Roman', size = stick_font_size)
        plt.xticks(fontproperties = 'Times New Roman', size = stick_font_size)
        plt.xlim([0.0, self.step_num])
        plt.ylabel("比率")
        plt.xlabel(r"时间 $t$")
        plt.legend(prop=font_legend, loc = 'upper right')
        plt.ylim([0.0, 1.0])
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, 'png', 'stacked_homo_state.png'), dpi=600, 
                    format='png', bbox_inches='tight', pad_inches=0.05)
        plt.savefig(os.path.join(self.save_path, 'pdf', 'stacked_homo_state.pdf'), dpi=600, 
                    format='pdf', bbox_inches='tight', pad_inches=0.05)
        plt.close()

if __name__ == '__main__':
    root_folder = os.path.dirname(os.path.abspath(__file__))
    figure_folder = os.path.join(root_folder, '..', 'figs')
    
    net = SEIR_Network(figure_folder)
    net.reset()
    
    done = False
    while not done:
        done = net.step()
    
    net.curve_render()
    net.stack_render()