from cProfile import label
import os
import pickle
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import cm
import matplotlib.colors as mcolors

config = {
    "font.family":'serif',
    "font.size": 25,
    "mathtext.fontset":'stix',
    "font.serif": ['SimSun'],
}
rcParams.update(config)
stick_font_size = 20
# font_legend = {'family': 'Times New Roman', 'weight': 'normal', 'size': stick_font_size}

class TwitterNet():
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.node_df = pd.read_csv(os.path.join(data_folder, "nodes.csv"), header=None)
        self.node_num = self.node_df.shape[0]
        self.edge_df = pd.read_csv(os.path.join(data_folder, "edges.csv"), header=None, names=['Follower','Target'])
        self.join_degree_ditribution = None
        self.subgraph_sample(15000, 300, 500000, continue_sampling=False)
        # self.subgraph_sample(10000, 100, 10000, continue_sampling=False)
    
    def subgraph_sample(self, start_node_id, threshold, graph_size, continue_sampling=True):
        graph_name = 'subgraph_{}.pkl'.format(start_node_id)
        if os.path.exists(graph_name):
            with open(graph_name, 'rb') as f:
                self.G = pickle.load(f)
            included_set = set(self.G.nodes)
            open_set = included_set
            edge_links = set(self.G.edges)
        else:
            included_set = set()
            open_set = set()
            edge_links = set()
            open_set.add(start_node_id)
        
        if continue_sampling is True:   
            out_degree_dict = self.edge_df.Follower.value_counts().to_dict()
            in_degree_dict = self.edge_df.Target.value_counts().to_dict()
            raw_node_size = len(included_set)
            node_size = len(included_set)
            while node_size < raw_node_size+graph_size:
                print('Included node size: {}, open set: {}'.format(node_size, len(open_set)))
                source_id = open_set.pop()
                in_neighbors = self.edge_df[self.edge_df.Target==source_id]
                out_neighbors = self.edge_df[self.edge_df.Follower==source_id]
                if in_neighbors.shape[0] < threshold and out_neighbors.shape[0] < threshold:
                    for edge_link in in_neighbors.values:
                        neighbor = edge_link[0]
                        neighbor_in_degree = in_degree_dict.get(neighbor, threshold+1)
                        neighbor_out_degree = out_degree_dict.get(neighbor, threshold+1)
                        if neighbor_in_degree <= threshold and neighbor_out_degree <= threshold:
                            open_set.add(neighbor)
                            included_set.add(neighbor)
                            edge_links.add(tuple(edge_link.tolist()))
                    for edge_link in out_neighbors.values:
                        neighbor = edge_link[1]
                        neighbor_in_degree = in_degree_dict.get(neighbor, threshold+1)
                        neighbor_out_degree = out_degree_dict.get(neighbor, threshold+1)
                        if neighbor_in_degree <= threshold and neighbor_out_degree <= threshold:
                            open_set.add(neighbor)
                            included_set.add(neighbor)
                            edge_links.add(tuple(edge_link.tolist()))
                    included_set.add(source_id)
                node_size = len(included_set)
                
            self.G = nx.DiGraph()
            self.G.add_nodes_from(list(included_set))
            self.G.add_edges_from(edge_links)
        
            with open(graph_name, 'wb') as f:
                pickle.dump(self.G, f)
        
    def extract_distribution(self):
        distribution_dict = dict()
        max_in = 0
        max_out = 0
        print('node evaluation: ')
        self.node_num = len(self.G.nodes)
        for node_id in tqdm(self.G.nodes, total=self.node_num, leave=False):
            outdegree = self.G.out_degree[node_id]
            indegree = self.G.in_degree[node_id]
            if distribution_dict.get((indegree, outdegree), None) is None:
                distribution_dict[indegree, outdegree] = list()
            distribution_dict[indegree, outdegree].append(node_id)
            max_in = indegree if indegree > max_in else max_in
            max_out = outdegree if outdegree > max_out else max_out
        
        print('generate distribution: ')
        self.join_degree_ditribution = np.zeros((max_in+1, max_out+1))
        for in_out_degree, node_list in distribution_dict.items():
            self.join_degree_ditribution[in_out_degree[0], in_out_degree[1]] = len(node_list)/self.node_num
            
        return self.join_degree_ditribution
    
    def draw_in_out_distribution(self, save_path, language='chinese', figsize=(8,6)):
        in_degree_distribution = np.sum(self.join_degree_ditribution, axis=1)
        out_degree_distribution = np.sum(self.join_degree_ditribution, axis=0)
        fig = plt.figure(figsize=figsize)
        ax = fig.subplots(1, 1)
        if language == 'chinese':
            ax.semilogy(np.arange(len(in_degree_distribution)), 
                    in_degree_distribution*self.node_num + 1, 
                    linewidth=2.5,
                    color='orange', label="入度")
            ax.semilogy(np.arange(len(out_degree_distribution)), 
                    out_degree_distribution*self.node_num + 1, 
                    linewidth=2,
                    color="seagreen",
                    label="出度")
        else:
            ax.semilogy(np.arange(len(in_degree_distribution)), 
                    in_degree_distribution*self.node_num + 1, 
                    linewidth=2.5,
                    color="orange", label=r'In-degree $j$ distribution')
            ax.semilogy(np.arange(len(out_degree_distribution)), 
                    out_degree_distribution*self.node_num + 1, 
                    linewidth=2,
                    color="seagreen",
                    label=r'Out-degree $k$ distribution')
        ax.grid(linestyle='-.')
        ax.spines['bottom'].set_linewidth(1.0)
        ax.spines['left'].set_linewidth(1.0)
        ax.spines['right'].set_linewidth(1.0)
        ax.spines['top'].set_linewidth(1.0)
        ax.legend(fontsize=stick_font_size)
        ax.set_yticks([1, 10, 100, 1000, 10000, 100000])
        ax.set_xticks(np.arange(0, 301, step=50))
        plt.xlim([0.0, 300.0])
        if language == 'chinese':
            ax.set_xlabel("出入度数")
            ax.set_ylabel('节点数（个）')
        else:
            ax.set_xlabel('Number of in-out-degree')
            ax.set_ylabel('Number of nodes')
        plt.yticks(fontproperties = 'Times New Roman', size = stick_font_size)
        plt.xticks(fontproperties = 'Times New Roman', size = stick_font_size)
        plt.savefig(os.path.join(save_path, 'png', 'degree_distribution.png'), dpi=600, 
                    format='png', bbox_inches='tight', pad_inches=0.05)
        plt.savefig(os.path.join(save_path, 'pdf', 'degree_distribution.pdf'), dpi=600, 
                    format='pdf', bbox_inches='tight', pad_inches=0.05)
        plt.close()
    
    def draw_joint_distribution(self, save_path, language='chinese', figsize=(8,6)):
        fig = plt.figure(figsize=figsize)
        ax = fig.subplots(1, 1)
        X = np.arange(self.join_degree_ditribution.shape[1])
        Y = np.arange(self.join_degree_ditribution.shape[0])
        X, Y = np.meshgrid(X, Y)
        Z = np.log10(self.join_degree_ditribution*len(list(self.G.nodes))+1)
        norm = mcolors.Normalize(vmin=0, vmax=2.5, clip=True)
        mesh = ax.pcolormesh(X, Y, Z, cmap=cm.jet, norm=norm, shading='nearest', rasterized=True)
        cbar = plt.colorbar(mesh, extend='max')
        cbar_font_dict=dict(fontsize=10, family='serif')
        cbar.ax.set_yticks([0, 0.5, 1, 1.5, 2, 2.5], fontsize=10)
        # cbar.ax.tick_params(labelsize=20)
        cbar.set_label(r'$\log_{10}(\mathcal{n})$')
        ax.set_xticks(np.arange(0, Z.shape[0], step=50))
        ax.set_yticks(np.arange(0, Z.shape[1], step=50))
        if language == 'chinese':
            ax.set_xlabel('入度')
            ax.set_ylabel('出度')
        else:
            ax.set_xlabel('In-degree')
            ax.set_ylabel('Out-degree')
        plt.yticks(fontproperties = 'Times New Roman', size = stick_font_size)
        plt.xticks(fontproperties = 'Times New Roman', size = stick_font_size)
        plt.savefig(os.path.join(save_path, 'png', 'degree_heatmap.png'), dpi=600, 
                    format='png', bbox_inches='tight', pad_inches=0.05)
        plt.savefig(os.path.join(save_path, 'pdf', 'degree_heatmap.pdf'), dpi=600, 
                    format='pdf', bbox_inches='tight', pad_inches=0.05)
        plt.close()
    
    def draw_graph(self, data_path, save_path, figsize=(8,6)):
        
        with open(data_path, 'rb') as f:
            graph_data = pickle.load(f)
        H = graph_data.to_undirected()   
        # compute centrality
        centrality = nx.betweenness_centrality(H, k=10, endpoints=True)
        
        # compute community structure
        lpc = nx.community.label_propagation_communities(H)
        community_index = {n: i for i, com in enumerate(lpc) for n in com}
        
        fig, ax = plt.subplots(figsize=figsize)
        pos = nx.spring_layout(H, k=0.15, seed=4572321)
        node_color = [community_index[n] for n in H]
        node_num = len(node_color)
        node_size = [v * node_num for v in centrality.values()]
        print('Current node number: {}'.format(node_num))
        nx.draw_networkx(
            H,
            pos=pos,
            with_labels=False,
            node_color=node_color,
            node_size=node_size,
            edge_color="gainsboro",
            alpha=0.75,
        )
        ax.margins(0.1, 0.05)
        fig.tight_layout()
        plt.axis("off")
        plt.savefig(os.path.join(save_path, 'png', 'nework_description.png'), dpi=600, 
                    format='png', bbox_inches='tight', pad_inches=0.05)
        plt.savefig(os.path.join(save_path, 'pdf', 'nework_description.pdf'), dpi=600, 
                    format='pdf', bbox_inches='tight', pad_inches=0.05)
        plt.close()
            

if __name__ == '__main__':
    root_folder = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(root_folder, '..', 'data', 'twitter_network')
    figure_folder =  os.path.join(root_folder, '..', 'figs')
    
    twitter_net = TwitterNet(data_folder)
    join_degree_ditribution = twitter_net.extract_distribution()
    fig_size = (8, 6)
    
    for language in ['chinese']:
        twitter_net.draw_in_out_distribution(figure_folder, language, fig_size)
        twitter_net.draw_joint_distribution(figure_folder, language, fig_size)
    
    # graph_data_path = os.path.join(root_folder, '..', 'subgraph_10000.pkl')
    # twitter_net.draw_graph(graph_data_path, figure_folder, fig_size)
    