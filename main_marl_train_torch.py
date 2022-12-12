from __future__ import division, print_function
import random
import numpy as np
import Environment_marl
import os
from replay_memory import ReplayMemory
from pathlib import Path
import sys
import argparse
import tensorboardX

import torch
import torch.nn as nn
import torch.nn.functional as F

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Agent(object):
    def __init__(self, memory_entry_size):
        self.discount = 1
        self.double_q = True
        self.memory_entry_size = memory_entry_size
        self.memory = ReplayMemory(self.memory_entry_size)


# ################## SETTINGS ######################
up_lanes = [i/2.0 for i in [3.5/2,3.5/2 + 3.5,250+3.5/2, 250+3.5+3.5/2, 500+3.5/2, 500+3.5+3.5/2]]
down_lanes = [i/2.0 for i in [250-3.5-3.5/2,250-3.5/2,500-3.5-3.5/2,500-3.5/2,750-3.5-3.5/2,750-3.5/2]]
left_lanes = [i/2.0 for i in [3.5/2,3.5/2 + 3.5,433+3.5/2, 433+3.5+3.5/2, 866+3.5/2, 866+3.5+3.5/2]]
right_lanes = [i/2.0 for i in [433-3.5-3.5/2,433-3.5/2,866-3.5-3.5/2,866-3.5/2,1299-3.5-3.5/2,1299-3.5/2]]

width = 750/2
height = 1298/2

IS_TRAIN = 1
IS_TEST = 1-IS_TRAIN

parser = argparse.ArgumentParser()
parser.add_argument('--label', type=str, default='marl_model')
args = parser.parse_args()
outpath = Path('ckpt') / args.label
outpath.mkdir(parents=True, exist_ok=True) 

n_veh = 4
n_neighbor = 1
n_RB = n_veh

env = Environment_marl.Environ(down_lanes, up_lanes, left_lanes, right_lanes, width, height, n_veh, n_neighbor)
env.new_random_game()  # initialize parameters in env

n_episode = 3000
n_step_per_episode = int(env.time_slow/env.time_fast)
epsi_final = 0.02
epsi_anneal_length = int(0.8*n_episode)
mini_batch_step = n_step_per_episode
target_update_step = n_step_per_episode*4

n_episode_test = 100  # test episodes

######################################################


def get_state(env, idx=(0,0), ind_episode=1., epsi=0.02):
    """ Get state from the environment """

    # V2I_channel = (env.V2I_channels_with_fastfading[idx[0], :] - 80) / 60
    V2I_fast = (env.V2I_channels_with_fastfading[idx[0], :] - env.V2I_channels_abs[idx[0]] + 10)/35

    # V2V_channel = (env.V2V_channels_with_fastfading[:, env.vehicles[idx[0]].destinations[idx[1]], :] - 80) / 60
    V2V_fast = (env.V2V_channels_with_fastfading[:, env.vehicles[idx[0]].destinations[idx[1]], :] - env.V2V_channels_abs[:, env.vehicles[idx[0]].destinations[idx[1]]] + 10)/35

    V2V_interference = (-env.V2V_Interference_all[idx[0], idx[1], :] - 60) / 60

    V2I_abs = (env.V2I_channels_abs[idx[0]] - 80) / 60.0
    V2V_abs = (env.V2V_channels_abs[:, env.vehicles[idx[0]].destinations[idx[1]]] - 80)/60.0

    load_remaining = np.asarray([env.demand[idx[0], idx[1]] / env.demand_size])
    time_remaining = np.asarray([env.individual_time_limit[idx[0], idx[1]] / env.time_slow])

    # return np.concatenate((np.reshape(V2V_channel, -1), V2V_interference, V2I_abs, V2V_abs, time_remaining, load_remaining, np.asarray([ind_episode, epsi])))
    return np.concatenate((V2I_fast, np.reshape(V2V_fast, -1), V2V_interference, np.asarray([V2I_abs]), V2V_abs, time_remaining, load_remaining, np.asarray([ind_episode, epsi])))


# -----------------------------------------------------------
n_hidden_1 = 500
n_hidden_2 = 250
n_hidden_3 = 120
n_input = len(get_state(env=env))
n_output = n_RB * len(env.V2V_power_dB_List)

class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        self.backbone = nn.Sequential(
            nn.Linear(n_input, n_hidden_1),
            nn.BatchNorm1d(n_hidden_1),
            nn.Linear(n_hidden_1, n_hidden_2),
            nn.BatchNorm1d(n_hidden_2),
            nn.Linear(n_hidden_2, n_hidden_3),
            nn.BatchNorm1d(n_hidden_3),
            nn.Linear(n_hidden_3, n_output),
            nn.ReLU()
        )

    def forward(self, x):
        return self.backbone(x)


def init_agent_network():
    policy_net = DQN().to(device)
    target_net = DQN().to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optim = torch.optim.RMSprop(policy_net.parameters(), lr=0.001, momentum=0.95, eps=0.01)
    criterion = nn.MSELoss()

    return {
        'policy_net': policy_net,
        'target_net': target_net,
        'optim': optim,
        'criterion': criterion,
    }


def predict(policy_net, s_t, ep, test_ep = False):
    n_power_levels = len(env.V2V_power_dB_List)
    if np.random.rand() < ep and not test_ep:
        pred_action = np.random.randint(n_RB*n_power_levels)
    else:
        policy_net.eval()

        with torch.no_grad():
            s_t = torch.from_numpy(s_t).to(torch.float32).unsqueeze(0).to(device)
            output = policy_net(s_t)
            pred_action = torch.max(output, 1)[1].item()

        policy_net.train()
    return pred_action


def q_learning_mini_batch(current_agent, policy_net, target_net, optim, criterion):
    """ Training a sampled mini-batch """
    
    batch_s_t, batch_s_t_plus_1, batch_action, batch_reward = current_agent.memory.sample()
    batch_s_t = torch.from_numpy(batch_s_t).to(torch.float32).to(device)
    batch_s_t_plus_1 = torch.from_numpy(batch_s_t_plus_1).to(torch.float32).to(device)
    batch_action = torch.from_numpy(batch_action).to(torch.int64).to(device)
    batch_reward = torch.from_numpy(batch_reward).to(torch.float32).to(device)
    
    with torch.no_grad():
        if current_agent.double_q:  # double q-learning
            # select best action with Q
            policy_output = policy_net(batch_s_t_plus_1)
            pred_action = torch.argmax(policy_output.data, 1)
            
            # calculate expected reward with q' (target network)
            target_output = target_net(batch_s_t_plus_1)
            q_t_plus_1 = target_output.gather(1, pred_action.unsqueeze(1)).squeeze()
            
            batch_target_q_t = current_agent.discount * q_t_plus_1 + batch_reward
        else:
            q_t_plus_1 = target_net(batch_s_t_plus_1)
            max_q_t_plus_1 = torch.max(q_t_plus_1.data, 1)
            batch_target_q_t = current_agent.discount * max_q_t_plus_1 + batch_reward

    batch_pred_q_t = policy_net(batch_s_t).gather(1, batch_action.unsqueeze(1)).squeeze()
    loss = criterion(batch_pred_q_t, batch_target_q_t)

    # Optimize the model
    optim.zero_grad()
    loss.backward()
    optim.step()

    return loss.item()


def load_models(sess, model_path):
    """ Restore models from the current directory with the name filename """

    dir_ = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(dir_, "model/" + model_path)
    saver.restore(sess, model_path)


def print_weight(sess, target=False):
    """ debug """

    if not target:
        print(sess.run(w_1[0, 0:4]))
    else:
        print(sess.run(w_1_p[0, 0:4]))


# --------------------------------------------------------------
agents = []
models = []
for ind_agent in range(n_veh * n_neighbor):  # initialize agents
    print("Initializing agent", ind_agent)
    agent = Agent(memory_entry_size=len(get_state(env)))
    agents.append(agent)
    models.append(init_agent_network())


# ------------------------- Training -----------------------------
writer = tensorboardX.SummaryWriter(outpath / 'summaries')
if IS_TRAIN:
    for i_episode in range(n_episode):
        print("-------------------------")
        print('Episode:', i_episode)
        if i_episode < epsi_anneal_length:
            epsi = 1 - i_episode * (1 - epsi_final) / (epsi_anneal_length - 1)  # epsilon decreases over each episode
        else:
            epsi = epsi_final
        if i_episode%100 == 0:
            env.renew_positions() # update vehicle position
            env.renew_neighbor()
            env.renew_channel() # update channel slow fading
            env.renew_channels_fastfading() # update channel fast fading

        env.demand = env.demand_size * np.ones((env.n_Veh, env.n_neighbor))
        env.individual_time_limit = env.time_slow * np.ones((env.n_Veh, env.n_neighbor))
        env.active_links = np.ones((env.n_Veh, env.n_neighbor), dtype='bool')

        for i_step in range(n_step_per_episode):
            time_step = i_episode*n_step_per_episode + i_step
            state_old_all = []
            action_all = []
            action_all_training = np.zeros([n_veh, n_neighbor, 2], dtype='int32')
            for i in range(n_veh):
                for j in range(n_neighbor):
                    state = get_state(env, [i, j], i_episode/(n_episode-1), epsi)
                    state_old_all.append(state)
                    action = predict(models[i*n_neighbor+j]['policy_net'], state, epsi)
                    action_all.append(action)

                    action_all_training[i, j, 0] = action % n_RB  # chosen RB
                    action_all_training[i, j, 1] = int(np.floor(action / n_RB)) # power level

            # All agents take actions simultaneously, obtain shared reward, and update the environment.
            action_temp = action_all_training.copy()
            train_reward = env.act_for_training(action_temp) 
            writer.add_scalar('reward', train_reward, global_step=time_step)

            env.renew_channels_fastfading()
            env.Compute_Interference(action_temp)

            record_loss = []
            for i in range(n_veh):
                for j in range(n_neighbor):
                    state_old = state_old_all[n_neighbor * i + j]
                    action = action_all[n_neighbor * i + j]
                    state_new = get_state(env, [i, j], i_episode/(n_episode-1), epsi)
                    agents[i * n_neighbor + j].memory.add(state_old, state_new, train_reward, action)  # add entry to this agent's memory

                    # training this agent
                    if time_step % mini_batch_step == mini_batch_step-1:
                        loss_val_batch = q_learning_mini_batch(agents[i*n_neighbor+j], **(models[i*n_neighbor+j]))
                        record_loss.append(loss_val_batch)
                        if i == 0 and j == 0:
                            print('step:', time_step, 'agent',i*n_neighbor+j, 'loss', loss_val_batch)
                    if time_step % target_update_step == target_update_step-1:
                        policy_net = models[i*n_neighbor+j]['policy_net']
                        target_net = models[i*n_neighbor+j]['target_net']
                        target_net.load_state_dict(policy_net.state_dict())
                        if i == 0 and j == 0:
                            print('Update target Q network...')

            if time_step % mini_batch_step == mini_batch_step-1:
                writer.add_scalar('loss', np.mean(record_loss), global_step=time_step)
            
            

    print('Training Done. Saving models...')
    for i in range(n_veh):
        for j in range(n_neighbor):
            filename = f'agent_{i * n_neighbor + j}.pt'
            model_path = outpath / filename
            torch.save(models[i * n_neighbor + j]['policy_net'].state_dict, model_path)