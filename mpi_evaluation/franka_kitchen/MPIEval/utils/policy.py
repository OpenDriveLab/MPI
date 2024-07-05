# ------------------------------------------------------------------------------------------------
# Modified from:
# R3M: https://github.com/facebookresearch/r3m/tree/eval/evaluation
# ------------------------------------------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
from MPIEval.utils.fc_network import FCNetwork, MAP_Adapter, MLP_Adapter
import torch
import torch.nn as nn
from torch.autograd import Variable
from mpi.models.util.extraction import MAPBlock


import torch.nn.functional as F

class MLP_Projector(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):  
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

# Proprio-conditioned adaptive
class PCAdaPolicy:
    def __init__(self, env_spec,
                 hidden_sizes=(64,64),
                 n_heads = 6,
                 proprio = 0,
                 init_log_std=0,
                 min_log_std=-3,
                 seed=None):
        """
        :param env_spec: specifications of the env (see utils/gym_env.py)
        :param hidden_sizes: network hidden layer sizes (currently 2 layers only)
        :param seed: random seed
        """
        super().__init__()
        self.n = env_spec.observation_dim # number of states
        self.m = env_spec.action_dim  # number of actions
        self.min_log_std = min_log_std
        # Set seed
        # ------------------------
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Proprio-conditioned Adapter
        self.adapter = MAP_Adapter(n_latents=1, obs_dim=self.n, n_heads=n_heads)
        self.proprio_projector = MLP_Projector(proprio, self.n, self.n, 2)
        
        # Policy head
        # ------------------------
        self.model = FCNetwork(self.n, self.m, hidden_sizes)
        # # make weights small
        # for param in list(self.model.parameters())[-2:]:  # only last layer
        #    param.data = 1e-2 * param.data

        # noise variable
        self.log_std = Variable(torch.ones(self.m) * init_log_std, requires_grad=True)
        self.log_std_val = np.float64(self.log_std.data.numpy().ravel())
    
        # Easy access variables
        self.trainable_params = list(self.proprio_projector.parameters()) + list(self.adapter.parameters()) +  list(self.model.parameters()) + [self.log_std]
        self.param_shapes = [p.data.numpy().shape for p in self.trainable_params]
        self.param_sizes = [p.data.numpy().size for p in self.trainable_params]
        self.d = np.sum(self.param_sizes)  # total number of params

        # Placeholders
        # ------------------------
        self.obs_var = Variable(torch.randn(self.n), requires_grad=False)

        

    # Utility functions
    # ============================================
    def get_param_values(self):
        params = np.concatenate([p.contiguous().view(-1).data.numpy()
                                 for p in self.trainable_params])
        return params.copy()

    def set_param_values(self, new_params):
        current_idx = 0
        for idx, param in enumerate(self.trainable_params):
            vals = new_params[current_idx:current_idx + self.param_sizes[idx]]
            vals = vals.reshape(self.param_shapes[idx])
            param.data = torch.from_numpy(vals).float()
            current_idx += self.param_sizes[idx]
        # clip std at minimum value
        self.trainable_params[-1].data = \
            torch.clamp(self.trainable_params[-1], self.min_log_std).data
        # update log_std_val for sampling
        self.log_std_val = np.float64(self.log_std.data.numpy().ravel())

    def switch_mode(self, mode):
        if mode == 'train':
            self.model.train()
            self.adapter.train()
            self.proprio_projector.train()
        elif mode == 'eval':
            self.model.eval()
            self.adapter.train()
            self.proprio_projector.train()

    # Main functions
    # ============================================
    def get_action(self, observation):
        img = observation['obs']
        proprio = observation['proprio']
        img = np.float32(np.expand_dims(img, 0))
        img = torch.from_numpy(img) 
        proprio = np.float32(np.expand_dims(proprio, 0))
        proprio = torch.from_numpy(proprio) 
        img = img.to('cpu')
        proprio = proprio.to('cpu')
        feat = self.adapter(img, self.proprio_projector(proprio))
        self.obs_var.data = feat
        mean = self.model(self.obs_var).data.numpy()
        mean = mean.ravel()
        noise = np.exp(self.log_std_val) * np.random.randn(self.m)
        action = mean + noise
        return [action, {'mean': mean, 'log_std': self.log_std_val, 'evaluation': mean}]
    
    def get_action_train(self, observation):
        img = observation['obs']
        proprio = observation['proprio']
        img = img.to('cpu')
        proprio = proprio.to('cpu')
        feat = self.adapter(img, self.proprio_projector(proprio))
        self.obs_var.data = feat
        mean = self.model(self.obs_var)
        return mean
    
    def forward(self,):
        pass

class OriginalMLP:
    def __init__(self, env_spec,
                 hidden_sizes=(64,64),
                 min_log_std=-3,
                 init_log_std=0,
                 proprio = 0,
                 seed=None):
        """
        :param env_spec: specifications of the env (see utils/gym_env.py)
        :param hidden_sizes: network hidden layer sizes (currently 2 layers only)
        :param min_log_std: log_std is clamped at this value and can't go below\vspace{20mm}
        :param init_log_std: initial log standard deviation
        :param seed: random seed
        """
        self.n = env_spec.observation_dim  # number of states
        self.m = env_spec.action_dim  # number of actions
        self.min_log_std = min_log_std

        # Set seed
        # ------------------------
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Policy network
        # ------------------------
        self.model = FCNetwork(self.n + proprio, self.m, hidden_sizes)
        # make weights small
        for param in list(self.model.parameters())[-2:]:  # only last layer
           param.data = 1e-2 * param.data
        self.log_std = Variable(torch.ones(self.m) * init_log_std, requires_grad=True)
        self.trainable_params = list(self.model.parameters()) + [self.log_std]

        # Easy access variables
        # -------------------------
        self.log_std_val = np.float64(self.log_std.data.numpy().ravel())
        self.param_shapes = [p.data.numpy().shape for p in self.trainable_params]
        self.param_sizes = [p.data.numpy().size for p in self.trainable_params]
        self.d = np.sum(self.param_sizes)  # total number of params

        # Placeholders
        # ------------------------
        self.obs_var = Variable(torch.randn(self.n), requires_grad=False)

    # Utility functions
    # ============================================
    def get_param_values(self):
        params = np.concatenate([p.contiguous().view(-1).data.numpy()
                                 for p in self.trainable_params])
        return params.copy()

    def set_param_values(self, new_params, set_new=True, set_old=True):
        if set_new:
            current_idx = 0
            for idx, param in enumerate(self.trainable_params):
                vals = new_params[current_idx:current_idx + self.param_sizes[idx]]
                vals = vals.reshape(self.param_shapes[idx])
                param.data = torch.from_numpy(vals).float()
                current_idx += self.param_sizes[idx]
            # clip std at minimum value
            self.trainable_params[-1].data = \
                torch.clamp(self.trainable_params[-1], self.min_log_std).data
            # update log_std_val for sampling
            self.log_std_val = np.float64(self.log_std.data.numpy().ravel())
        if set_old:
            current_idx = 0
            for idx, param in enumerate(self.old_params):
                vals = new_params[current_idx:current_idx + self.param_sizes[idx]]
                vals = vals.reshape(self.param_shapes[idx])
                param.data = torch.from_numpy(vals).float()
                current_idx += self.param_sizes[idx]
            # clip std at minimum value
            self.old_params[-1].data = \
                torch.clamp(self.old_params[-1], self.min_log_std).data
    
    # Main functions
    # ============================================
    def get_action(self, observation):
        # o = np.float32(observation.reshape(1, -1))
        # observation = {'img': obs, 'proprio': proprio}
        emb = observation['obs']
        proprio = observation['proprio']
        emb = np.float32(np.expand_dims(emb, 0))
        emb = torch.from_numpy(emb) 
        emb = emb.to('cpu')
        if proprio is not None:
            proprio = np.float32(np.expand_dims(proprio, 0))
            proprio = torch.from_numpy(proprio) 
            proprio = proprio.to('cpu')
            self.obs_var.data = torch.cat([emb, proprio], dim=-1) #o
        else:
            self.obs_var.data = emb
        mean = self.model(self.obs_var).data.numpy()
        mean = mean.ravel()
        noise = np.exp(self.log_std_val) * np.random.randn(self.m)
        action = mean + noise
        #action = mean
        return [action, {'mean': mean, 'log_std': self.log_std_val, 'evaluation': mean}]
    
    def get_action_train(self, observation):
        img = observation['obs']
        proprio = observation['proprio']
        img = img.to('cpu')
        o = img 
        if proprio is not None:
            proprio = proprio.to('cpu')
            self.obs_var.data = torch.cat([o, proprio], dim=-1)
        else:
            self.obs_var.data = o
        mean = self.model(self.obs_var)
        return mean

    def mean_LL(self, observations, actions, model=None, log_std=None):
        model = self.model if model is None else model
        log_std = self.log_std if log_std is None else log_std
        if type(observations) is not torch.Tensor:
            obs_var = Variable(torch.from_numpy(observations).float(), requires_grad=False)
        else:
            obs_var = observations
        if type(actions) is not torch.Tensor:
            act_var = Variable(torch.from_numpy(actions).float(), requires_grad=False)
        else:
            act_var = actions
        mean = model(obs_var)
        zs = (act_var - mean) / torch.exp(log_std)
        LL = - 0.5 * torch.sum(zs ** 2, dim=1) + \
             - torch.sum(log_std) + \
             - 0.5 * self.m * np.log(2 * np.pi)
        return mean, LL

    def log_likelihood(self, observations, actions, model=None, log_std=None):
        mean, LL = self.mean_LL(observations, actions, model, log_std)
        return LL.data.numpy()

    def old_dist_info(self, observations, actions):
        mean, LL = self.mean_LL(observations, actions, self.old_model, self.old_log_std)
        return [LL, mean, self.old_log_std]

    def new_dist_info(self, observations, actions):
        mean, LL = self.mean_LL(observations, actions, self.model, self.log_std)
        return [LL, mean, self.log_std]

    def likelihood_ratio(self, new_dist_info, old_dist_info):
        LL_old = old_dist_info[0]
        LL_new = new_dist_info[0]
        LR = torch.exp(LL_new - LL_old)
        return LR
