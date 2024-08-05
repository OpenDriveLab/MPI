# ------------------------------------------------------------------------------------------------
# Modified from:
# R3M: https://github.com/facebookresearch/r3m/tree/eval/evaluation
# ------------------------------------------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from collections import namedtuple
from MPIEval.utils.gym_env import GymEnv
from MPIEval.utils.obs_wrappers import MuJoCoPixelObs, StateEmbedding
from MPIEval.utils.sampling import sample_paths
from mpi_evaluation.franka_kitchen.MPIEval.utils.policy import PCAdaPolicy, OriginalMLP
from MPIEval.utils.behavior_cloning import BC
from tabulate import tabulate
from tqdm import tqdm
import mj_envs, gym 
import numpy as np, time as timer, multiprocessing, pickle, os
import os
import time
from collections import namedtuple
import torch.nn as nn


def env_constructor(env_name, device='cuda', image_width=256, image_height=256,
                    camera_name=None, embedding_name='resnet50', pixel_based=True,
                    render_gpu_id=0, load_path="", proprio=False, lang_cond=False, gc=False, 
                    path_demo = '.', path_ckpt = '.'):

    ## If pixel based will wrap in a pixel observation wrapper
    if pixel_based:
        print("env_name: ", env_name)
        e = gym.make(env_name)
        ## Wrap in pixel observation wrapper
        e = MuJoCoPixelObs(e, width=image_width, height=image_height, 
                           camera_name=camera_name, device_id=render_gpu_id)
        ## Wrapper which encodes state in pretrained model
        e = StateEmbedding(e, embedding_name=embedding_name, device=device, load_path=load_path, 
                        proprio=proprio, camera_name=camera_name, env_name=env_name, path_ckpt = path_ckpt)
        e = GymEnv(e)
    else:
        print("Only supports pixel based")
        assert(False)
    return e


def make_bc_agent(env_kwargs:dict, bc_kwargs:dict, demo_paths:list, epochs:int, seed:int, pixel_based=True):
    ## Creates environment
    e = env_constructor(**env_kwargs)
    print("successfully contrusct environment")
    ## Creates policy netwwork (Where the FC Network has a batchnorm in front of it)
    if 'mpi' in env_kwargs['load_path']:
        print(env_kwargs['embedding_name'])
        if env_kwargs['embedding_name'] == 'ViT-Base':
            policy = PCAdaPolicy(e.spec, hidden_sizes=(), n_heads=12, seed=seed, proprio = bc_kwargs['proprio']) # Base
        elif env_kwargs['embedding_name'] == 'ViT-Small':
            policy = PCAdaPolicy(e.spec, hidden_sizes=(), n_heads=6, seed=seed, proprio = bc_kwargs['proprio']) # Small
    elif 'r3m' in env_kwargs['load_path']:
        policy = OriginalMLP(e.spec, hidden_sizes=(256, 256,), seed=seed, proprio = bc_kwargs['proprio'])
    else:
        policy = OriginalMLP(e.spec, hidden_sizes=(256, 256,), seed=seed, proprio = bc_kwargs['proprio'])
    
    policy.model.proprio_only = False
        
    ## Pass the encoder params to the BC agent (for finetuning)
    if pixel_based:
        enc_p = e.env.embedding.parameters()
    else:
        print("Only supports pixel based")
        assert(False)
    bc_agent = BC(demo_paths, policy=policy, epochs=epochs, set_transforms=True, encoder_params=enc_p, **bc_kwargs)

    ## Pass the environmetns observation encoder to the BC agent to encode demo data
    if pixel_based:
        bc_agent.encodefn = e.env.encode_batch
    else:
        print("Only supports pixel based")
        assert(False)
    return e, bc_agent


def configure_cluster_GPUs(gpu_logical_id: int) -> int:
    # get the correct GPU ID
    if "SLURM_STEP_GPUS" in os.environ.keys():
        physical_gpu_ids = os.environ.get('SLURM_STEP_GPUS')
        gpu_id = int(physical_gpu_ids.split(',')[gpu_logical_id])
        print("Found slurm-GPUS: <Physical_id:{}>".format(physical_gpu_ids))
        print("Using GPU <Physical_id:{}, Logical_id:{}>".format(gpu_id, gpu_logical_id))
    else:
        gpu_id = 0 # base case when no GPUs detected in SLURM
        print("No GPUs detected. Defaulting to 0 as the device ID")
    return gpu_id


def bc_train_loop(job_data:dict) -> None:
    # configure GPUs
    print("render gpu id:", job_data['env_kwargs']['render_gpu_id'])
    data_dir = job_data['env_kwargs']['path_demo']
    demo_paths_loc = os.path.join(data_dir, 'final_paths_multiview_rb_200/', job_data['camera'], job_data['env_kwargs']['env_name'] + '.pickle')
    ## Loads the demos
    demo_paths = pickle.load(open(demo_paths_loc, 'rb'))
    demo_paths = demo_paths[:job_data['num_demos']]
    print(len(demo_paths))
    demo_score = np.mean([np.sum(p['rewards']) for p in demo_paths])
    print("Demonstration score : %.2f " % demo_score)

    # Make log dir
    if os.path.isdir(job_data['job_name']) == False: os.mkdir(job_data['job_name'])
    os.chdir(job_data['job_name']) # important! we are now in the directory to save data
    if os.path.isdir('iterations') == False: os.mkdir('iterations')
    if os.path.isdir('logs') == False: os.mkdir('logs')

    ## Creates agent and environment
    env_kwargs = job_data['env_kwargs']
    e, agent = make_bc_agent(env_kwargs=env_kwargs, bc_kwargs=job_data['bc_kwargs'], 
                             demo_paths=demo_paths, epochs=1, seed=job_data['seed'], pixel_based=job_data["pixel_based"])
    agent.logger.init_wb(job_data)

    max_success = 0
    epoch = 0
    start_time_total = time.time()
    while True:
        # update policy using one BC epoch
        last_step = agent.steps
        print("Step", last_step)
        # agent.policy.model.train()
        # agent.policy.adapter.train()
        # agent.policy.proprio_projector.train()
        agent.policy.switch_mode('train')
        # # If finetuning, wait until 25% of training is done then
        # ## set embedding to train mode and turn on finetuning
        # if (job_data['bc_kwargs']['finetune']) and (job_data['pixel_based']) and (job_data['env_kwargs']['load_path'] != "clip"):
        #     if last_step > (job_data['steps'] / 4.0):
        #         e.env.embedding.train()
        #         e.env.start_finetuning()
        if (job_data['bc_kwargs']['finetune']):
            e.env.embedding.train()
            e.env.start_finetuning()
            for _, param in e.env.embedding.named_parameters():
                param.requires_grad = True

        agent.train(job_data['pixel_based'], suppress_fit_tqdm=True, step = last_step)
        
        # perform evaluation rollouts every few epochs
        if ((agent.steps % job_data['eval_frequency']) < (last_step % job_data['eval_frequency'])):
            start_time_evaluation = time.time()
            # agent.policy.model.eval()
            # agent.policy.adapter.eval()
            # agent.policy.proprio_projector.eval()
            agent.policy.switch_mode('eval')
            if job_data['pixel_based']:
                e.env.embedding.eval()
            print("start sample path")
            print("base seed: ", job_data['seed'], ", epoch: ", epoch, ", final seed: ", job_data['seed']+epoch)
            paths = sample_paths(num_traj=job_data['eval_num_traj'], env=e, #env_constructor, 
                                 policy=agent.policy, eval_mode=True, horizon=e.horizon, 
                                 base_seed=job_data['seed']+epoch, num_cpu=job_data['num_cpu'], 
                                 env_kwargs=env_kwargs)
            print("end sample path")
            
            try:
                ## Success computation and logging for Adroit and Kitchen
                print("start evaluate success: {}".format(agent.steps))
                success_percentage = e.env.unwrapped.evaluate_success(paths)
                print("end evaluate success: {}".format(agent.steps))
                for i, path in enumerate(paths):
                    if (i < 10) and job_data['pixel_based']:
                        vid = path['images']
                        filename = f'./iterations/vid_{i}.gif'
                        from moviepy.editor import ImageSequenceClip
                        cl = ImageSequenceClip(vid, fps=20)
                        cl.write_gif(filename, fps=20)
            except:
                ## Success computation and logging for MetaWorld
                sc = []
                for i, path in enumerate(paths):
                    sc.append(path['env_infos']['success'][-1])
                    if (i < 10) and job_data['pixel_based']:
                        vid = path['images']
                        filename = f'./iterations/vid_{i}.gif'
                        from moviepy.editor import ImageSequenceClip
                        cl = ImageSequenceClip(vid, fps=20)
                        cl.write_gif(filename, fps=20)
                success_percentage = np.mean(sc) * 100
            agent.logger.log_kv('eval_epoch', epoch)
            agent.logger.log_kv('eval_success', success_percentage)
            
            # Tracking best success over training
            max_success = max(max_success, success_percentage)

            # save policy and logging
            pickle.dump(agent.policy, open('./iterations/policy_%i.pickle' % epoch, 'wb'))
            agent.logger.save_log('./logs/')
            agent.logger.save_wb(step=agent.steps)

            print_data = sorted(filter(lambda v: np.asarray(v[1]).size == 1,
                                        agent.logger.get_current_log().items()))
            print(tabulate(print_data))
            end_time_evaluation = time.time()
            print("evaluation time cost: {} min".format((end_time_evaluation-start_time_evaluation)/60))
        epoch += 1
        if agent.steps > job_data['steps']:
            break
    agent.logger.log_kv('max_success', max_success)
    agent.logger.save_wb(step=agent.steps)
    end_time_total = time.time()
    print("total time cost: {} min".format((end_time_total-start_time_total)/60))

