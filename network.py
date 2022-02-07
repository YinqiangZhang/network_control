import os
import yaml
import numpy as np
import matplotlib.pyplot as plt


class Network(object):
    '''
        Implementation of single network and dynamics
        :param id: Network ID (must be set)
        :param param_path: hyperparameters path 
        :param enable_memory: flag for recording the historical data
    '''
    def __init__(self, id=None, param_path=None, enable_memory=False):
        """
            describes the strucutre and dynamics of a single network.
            param id: identifier of this network 
        """
        # network id 
        assert id is not None, 'Network ID should be set firsly.'
        self.ID = id

        # load parameters from yaml file
        assert param_path is not None, "parameters should be set before using the network."
        with open(param_path, 'r') as para_configs:
            self._params = yaml.safe_load(para_configs)['network_configs']
        for param, value in self._params.items():
            self.__dict__[str(param)] = value
        
        # shapes for variables
        self.total_steps = int(self.T / self.dt)
        self.degree_shape =(self.max_in_degree,self.max_out_degree)
        self.memory_shape =(self.max_in_degree,self.max_out_degree, self.total_steps)
        
        # generate distribution
        self._generate_ditribution()
        
        # visualization memory
        self.enable_memory = enable_memory
        
        # observable variables 
        self.observable_dict = ['S', 'H', 'I', 'L', 'R']
        
        # termination condition
        self.termination_S_condition = 1.0e-3
                  
    def _generate_ditribution(self):
        """
            generates the netowrk distribution with given parameters 
        """
        in_degree_distribution = (np.arange(self.max_in_degree)+1)**self.alpha
        out_degree_distribution = (np.arange(self.max_out_degree)+1)**self.alpha
        in_degree_distribution /= np.sum(in_degree_distribution)
        out_degree_distribution /= np.sum(out_degree_distribution)
    
        self.avg_in_degree = np.matmul(np.arange(self.max_in_degree)+1, in_degree_distribution)
        self.avg_out_degree = np.matmul(np.arange(self.max_out_degree)+1, out_degree_distribution)
        
        self.joint_degree_distribution = np.outer(in_degree_distribution,out_degree_distribution)

    def _generate_observations(self):
        """
            gets the current observations of the network. 
        """
        host_obs_dict = dict()
        for obs_variable in self.observable_dict:
            pass
        
    def reset(self):
        """
            resets the memory of variables in simulation
        """
        # reset time step
        self.current_time_step = 0
        
        # gamma update and settings
        self.gamma = np.ones(self.degree_shape)*0.5
        
        # define required state names
        self.main_state_name = ['S', 'H', 'I', 'L', 'R']
        self.auxiliary_state_name = ['R_H', 'R_I', 'R_L']
        
        # create node states
        for name in (self.main_state_name + self.auxiliary_state_name):
            self.__dict__[name] = np.zeros(self.degree_shape)
            if self.enable_memory:
                self.__dict__["memory_{}".format(name)] = np.zeros(self.memory_shape)
        
        # create edge states 
        self.theta = 1.0
        for name in self.main_state_name:
            self.__dict__['theta_{}'.format(name)] = 0.0

        # set initial values
        self.S = self.joint_degree_distribution
        self.theta_I = self.theta_I0
        self.theta_S = 1-self.theta_I0
        
        if self.enable_memory:
            self.memory_S[:, :, self.current_time_step] = self.S
        
        print('Network {} has been reset.'.format(self.ID))
    
    def step(self, action, guest_obs_dict=None):
        """
            forward simulation of the network in one time step 
            based on the new observations.
            param observations: the observations of outside environment 
            (the behavior of other network)
        """
        self.current_time_step += 1
        
        # set action (matrix like j-k degree distribution)
        self.u = action * np.ones(self.degree_shape)
        
        # set temporary variables
        delta_S = np.zeros_like(self.S)
        delta_H = np.zeros_like(self.H)
        delta_I = np.zeros_like(self.I)
        delta_L = np.zeros_like(self.L)
        delta_R = np.zeros_like(self.R)
        delta_R_I = np.zeros_like(self.R_I)
        delta_R_L = np.zeros_like(self.R_L)
        delta_R_H = np.zeros_like(self.R_H)
        temp_theta_I = 0
        temp_theta_L = 0
        temp_theta_H = 0
        temp_theta_S = 0
                
        # internal dynamics
        for j in range(self.max_in_degree):
            for k in range(self.max_out_degree):
                # state transitions
                delta_S[j, k] = -(j+1) * (self.r_I * self.theta_I + self.r_L * self.theta_L) / self.theta * self.S[j, k]
                delta_H[j, k] = -delta_S[j, k] - self.beta * self.H[j, k]
                delta_I[j, k] = self.beta * self.gamma[j, k] * self.H[j, k] - (self.mu_I + self.u[j, k]) * self.I[j, k] 
                delta_L[j, k] = self.u[j, k] * self.I[j, k] - self.mu_L * self.L[j, k]
                delta_R_I[j, k] = self.mu_I * self.I[j, k]
                delta_R_L[j, k] = self.mu_L * self.L[j, k]
                delta_R_H[j, k] = self.beta * (1 - self.gamma[j, k]) * self.H[j, k]
                delta_R[j, k] = delta_R_H[j, k] + delta_R_I[j, k] + delta_R_L[j, k]

                temp_theta_I += (k+1) * (self.beta * self.gamma[j, k] * self.H[j, k] - self.I[j, k] * self.u[j, k])
                temp_theta_L += (k+1) * self.I[j, k] * self.u[j, k]
                temp_theta_H += (k+1) * delta_H[j ,k]
                temp_theta_S += (k+1) * delta_S[j, k]
        
        # theta state transition
        delta_theta_S = temp_theta_S/self.avg_out_degree
        delta_theta_H = temp_theta_H/self.avg_out_degree
        delta_theta_I = temp_theta_I/self.avg_out_degree - (self.r_I + self.mu_I) * self.theta_I # TODO: observations
        delta_theta_L = temp_theta_L/self.avg_out_degree - (self.r_L + self.mu_L) * self.theta_L # TODO: observations
        delta_theta = -(self.r_I * self.theta_I + self.r_L * self.theta_L)
                
        # outside dynamics (from observations)
        if guest_obs_dict is not None:
            pass
        
        # update current states
        self.S += delta_S * self.dt
        self.H += delta_H * self.dt
        self.I += delta_I * self.dt
        self.L += delta_L * self.dt
        self.R_I += delta_R_I * self.dt
        self.R_L += delta_R_L * self.dt
        self.R_H += delta_R_H * self.dt
        self.R += delta_R * self.dt  
        
        self.theta += delta_theta * self.dt
        self.theta_I += delta_theta_I * self.dt
        self.theta_L += delta_theta_L * self.dt
        self.theta_S += delta_theta_S * self.dt
        self.theta_H += delta_theta_H * self.dt
        self.theta_R = self.theta - self.theta_S - self.theta_I - self.theta_L - self.theta_H
        
        # update memory states
        if self.enable_memory:
            self.memory_S[:, :, self.current_time_step] = self.S
            self.memory_H[:, :, self.current_time_step] = self.H
            self.memory_I[:, :, self.current_time_step] = self.I
            self.memory_L[:, :, self.current_time_step] = self.L
            self.memory_R[:, :, self.current_time_step] = self.R
            self.memory_R_I[:, :, self.current_time_step] = self.R_I
            self.memory_R_L[:, :, self.current_time_step] = self.R_L
            self.memory_R_H[:, :, self.current_time_step] = self.R_H
        
        # generate observation dictionary
        host_obs_dict = self._generate_observations()

        # generate reward 
        reward = None
    
        # generate info
        info = dict()
        
        # check done conditions
        done = False
        if np.sum(self.S) <= self.termination_S_condition or self.current_time_step >= self.total_steps-1:
            done = True
            
        return host_obs_dict, reward, done, info
        
    def render(self, save_path='.'):
        """
            draws the current state of network
            only node state is considered now
        """
        if self.enable_memory:
            plt.figure(figsize=(8,6))
            t = np.arange(0, (self.current_time_step+1)*self.dt, self.dt)
            plt.plot(t, np.sum(self.memory_S, axis=(0,1)), color='b', linewidth=2, label='S')
            plt.plot(t, np.sum(self.memory_H, axis=(0,1)), color='r', linewidth=2, label='H')
            plt.plot(t, np.sum(self.memory_I, axis=(0,1)), color='g', linewidth=2, label='I')
            plt.plot(t, np.sum(self.memory_L, axis=(0,1)), color='m', linewidth=2, label='L')
            plt.plot(t, np.sum(self.memory_R, axis=(0,1)), color='orange', linewidth=2, label='R')
            ax = plt.gca()
            ax.spines['bottom'].set_linewidth(1.5)
            ax.spines['left'].set_linewidth(1.5)
            ax.spines['right'].set_linewidth(1.5)
            ax.spines['top'].set_linewidth(1.5)
            plt.yticks(fontsize=12)
            plt.xticks(fontsize=12)
            plt.grid(linestyle='-.')
            plt.ylabel("State ratio", fontsize=12)
            plt.xlabel("Time", fontsize=12)
            plt.legend(fontsize=12, loc = 'upper right')
            plt.ylim([0.0, 1.0])
            plt.xlim([0.0, (self.current_time_step+1)*self.dt])
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, 'network_{}.png'.format(self.ID)), dpi=600, format='png', bbox_inches='tight', pad_inches=0.05)
            plt.close()
        else:
            print("Please enable the flag 'enable_memory' before the forward simulation of the network, thanks!")
        

if __name__ == '__main__':
    
    # initialize
    network_0 = Network(id=0, param_path='./config/network_00.yaml', enable_memory=True)
    network_0.reset()
    
    # forward dynamics of network 
    done = False
    while not done:
        _, _, done, _ = network_0.step(action = 0.1)
    
    # visualization
    network_0.render()
    
        