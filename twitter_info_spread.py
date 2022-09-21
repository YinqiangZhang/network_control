import os
from envs import InfoNetwork
from utils import TwitterNet

os.chdir(os.path.dirname(os.path.abspath(__file__)))

root_folder = os.path.dirname(os.path.abspath(__file__))
data_folder = os.path.join(root_folder, 'data', 'twitter_network')
config_path = os.path.join(root_folder, 'config', 'network_00.yaml')
figure_folder = os.path.join(root_folder, 'figs')

data_net = TwitterNet(data_folder)
joint_distribution = data_net.extract_distribution()
info_spread_net = InfoNetwork(id=0, param_path=config_path, prior_distribution=joint_distribution, enable_memory=True,
                              use_chinese=True)
info_spread_net.reset()
use_record = True
    

# forward dynamics of network 
if use_record:
    info_spread_net.load('dynamics.pkl')
else:
    done = False
    while not done:
        _, _, done, _ = info_spread_net.step(action = 0.0)

# visualization
info_spread_net.render(figure_folder)
info_spread_net.render(figure_folder)

# # creating animation
# animation = VideoClip(make_heatmap, duration = 2)

# # displaying animation with auto play and looping
# animation.write_gif("my_animation.gif", fps=50) # usually slower