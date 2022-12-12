from __future__ import division, print_function
import random
import numpy as np
import Environment_marl_test
import os
from replay_memory import ReplayMemory
from pathlib import Path
import sys
import argparse

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

# This main file is for testing only
IS_TRAIN = 0 # hard-coded to 0
IS_TEST = 1-IS_TRAIN

parser = argparse.ArgumentParser()
parser.add_argument('--label', type=str, default='marl_model')
args = parser.parse_args()
outpath = Path('ckpt') / args.label

n_veh = 4
n_neighbor = 1
n_RB = n_veh

env = Environment_marl_test.Environ(down_lanes, up_lanes, left_lanes, right_lanes, width, height, n_veh, n_neighbor)
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


def get_state_sarl(env, idx=(0,0), ind_episode=1., epsi=0.02):
    """ Get state from the environment """

    # V2I_channel = (env.V2I_channels_with_fastfading[idx[0], :] - 80) / 60
    V2I_fast = (env.V2I_channels_with_fastfading[idx[0], :] - env.V2I_channels_abs[idx[0]] + 10)/35

    # V2V_channel = (env.V2V_channels_with_fastfading[:, env.vehicles[idx[0]].destinations[idx[1]], :] - 80) / 60
    V2V_fast = (env.V2V_channels_with_fastfading[:, env.vehicles[idx[0]].destinations[idx[1]], :] - env.V2V_channels_abs[:, env.vehicles[idx[0]].destinations[idx[1]]] + 10)/35

    V2V_interference = (-env.V2V_Interference_all_sarl[idx[0], idx[1], :] - 60) / 60

    V2I_abs = (env.V2I_channels_abs[idx[0]] - 80) / 60.0
    V2V_abs = (env.V2V_channels_abs[:, env.vehicles[idx[0]].destinations[idx[1]]] - 80)/60.0

    load_remaining = np.asarray([env.demand_sarl[idx[0], idx[1]] / env.demand_size])
    time_remaining = np.asarray([env.individual_time_limit_sarl[idx[0], idx[1]] / env.time_slow])

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


# --------------------------------------------------------------
agents = []
for ind_agent in range(n_veh * n_neighbor):  # initialize agents
    print("Initializing agent", ind_agent)
    agent = Agent(memory_entry_size=len(get_state(env)))
    agents.append(agent)

# -------------- Testing --------------
if IS_TEST:
    print("\nRestoring the model...")

    models = []
    for i in range(n_veh):
        for j in range(n_neighbor):
            filename = f'agent_{i * n_neighbor + j}.pt'
            model_path = outpath / filename
            ckpt = DQN()
            ckpt.load_state_dict(torch.load(model_path))
            ckpt.to(device)
            models.append(ckpt)
            
    # restore the single-agent model
    # model_path_single = label_sarl + '/agent'
    # load_models(sess_sarl, model_path_single)

    V2I_rate_list = []
    V2V_success_list = []

    V2I_rate_list_rand = []
    V2V_success_list_rand = []

    V2I_rate_list_sarl = []
    V2V_success_list_sarl = []

    V2I_rate_list_dpra = []
    V2V_success_list_dpra = []

    rate_marl = np.zeros([n_episode_test, n_step_per_episode, n_veh, n_neighbor])
    rate_rand = np.zeros([n_episode_test, n_step_per_episode, n_veh, n_neighbor])
    demand_marl = env.demand_size * np.ones([n_episode_test, n_step_per_episode+1, n_veh, n_neighbor])
    demand_rand = env.demand_size * np.ones([n_episode_test, n_step_per_episode+1, n_veh, n_neighbor])

    action_all_testing_sarl = np.zeros([n_veh, n_neighbor, 2], dtype='int32')
    action_all_testing_dpra = np.zeros([n_veh, n_neighbor, 2], dtype='int32')
    for idx_episode in range(n_episode_test):
        print('----- Episode', idx_episode, '-----')

        env.renew_positions()
        env.renew_neighbor()
        env.renew_channel()
        env.renew_channels_fastfading()

        env.demand = env.demand_size * np.ones((env.n_Veh, env.n_neighbor))
        env.individual_time_limit = env.time_slow * np.ones((env.n_Veh, env.n_neighbor))
        env.active_links = np.ones((env.n_Veh, env.n_neighbor), dtype='bool')

        env.demand_rand = env.demand_size * np.ones((env.n_Veh, env.n_neighbor))
        env.individual_time_limit_rand = env.time_slow * np.ones((env.n_Veh, env.n_neighbor))
        env.active_links_rand = np.ones((env.n_Veh, env.n_neighbor), dtype='bool')

        env.demand_sarl = env.demand_size * np.ones((env.n_Veh, env.n_neighbor))
        env.individual_time_limit_sarl = env.time_slow * np.ones((env.n_Veh, env.n_neighbor))
        env.active_links_sarl = np.ones((env.n_Veh, env.n_neighbor), dtype='bool')

        env.demand_dpra = env.demand_size * np.ones((env.n_Veh, env.n_neighbor))
        env.individual_time_limit_dpra = env.time_slow * np.ones((env.n_Veh, env.n_neighbor))
        env.active_links_dpra = np.ones((env.n_Veh, env.n_neighbor), dtype='bool')

        V2I_rate_per_episode = []
        V2I_rate_per_episode_rand = []
        V2I_rate_per_episode_sarl = []
        V2I_rate_per_episode_dpra = []

        for test_step in range(n_step_per_episode):
            # trained models
            action_all_testing = np.zeros([n_veh, n_neighbor, 2], dtype='int32')
            for i in range(n_veh):
                for j in range(n_neighbor):
                    state_old = get_state(env, [i, j], 1, epsi_final)
                    action = predict(models[i*n_neighbor+j], state_old, epsi_final, True)
                    action_all_testing[i, j, 0] = action % n_RB  # chosen RB
                    action_all_testing[i, j, 1] = int(np.floor(action / n_RB))  # power level

            action_temp = action_all_testing.copy()
            V2I_rate, V2V_success, V2V_rate = env.act_for_testing(action_temp)
            V2I_rate_per_episode.append(np.sum(V2I_rate))  # sum V2I rate in bps

            rate_marl[idx_episode, test_step,:,:] = V2V_rate
            demand_marl[idx_episode, test_step+1,:,:] = env.demand

            # random baseline
            action_rand = np.zeros([n_veh, n_neighbor, 2], dtype='int32')
            action_rand[:, :, 0] = np.random.randint(0, n_RB, [n_veh, n_neighbor]) # band
            action_rand[:, :, 1] = np.random.randint(0, len(env.V2V_power_dB_List), [n_veh, n_neighbor]) # power
            V2I_rate_rand, V2V_success_rand, V2V_rate_rand = env.act_for_testing_rand(action_rand)
            V2I_rate_per_episode_rand.append(np.sum(V2I_rate_rand))  # sum V2I rate in bps

            rate_rand[idx_episode, test_step, :, :] = V2V_rate_rand
            demand_rand[idx_episode, test_step+1,:,:] = env.demand_rand

            action_dpra = np.zeros([n_veh, n_neighbor, 2], dtype='int32')
            # n_power_level = len(env.V2V_power_dB_List)
            n_power_level = 1
            store_action = np.zeros([(n_RB*n_power_level)**4, 4])
            rate_all_dpra = []
            t = 0
            # for i in range(n_RB*len(env.V2V_power_dB_List)):\
            for i in range(n_RB):
                for j in range(n_RB):
                    for m in range(n_RB):
                        for n in range(n_RB):
                            action_dpra[0, 0, 0] = i % n_RB
                            action_dpra[0, 0, 1] = int(np.floor(i / n_RB))  # power level

                            action_dpra[1, 0, 0] = j % n_RB
                            action_dpra[1, 0, 1] = int(np.floor(j / n_RB))  # power level

                            action_dpra[2, 0, 0] = m % n_RB
                            action_dpra[2, 0, 1] = int(np.floor(m / n_RB))  # power level

                            action_dpra[3, 0, 0] = n % n_RB
                            action_dpra[3, 0, 1] = int(np.floor(n / n_RB))  # power level

                            action_temp_findMax = action_dpra.copy()
                            V2I_rate_findMax, V2V_rate_findMax = env.Compute_Rate(action_temp_findMax)
                            rate_all_dpra.append(np.sum(V2V_rate_findMax))

                            store_action[t, :] = [i,j,m,n]
                            t += 1

            i = store_action[np.argmax(rate_all_dpra), 0]
            j = store_action[np.argmax(rate_all_dpra), 1]
            m = store_action[np.argmax(rate_all_dpra), 2]
            n = store_action[np.argmax(rate_all_dpra), 3]

            action_testing_dpra = np.zeros([n_veh, n_neighbor, 2], dtype='int32')

            action_testing_dpra[0, 0, 0] = i % n_RB
            action_testing_dpra[0, 0, 1] = int(np.floor(i / n_RB))  # power level

            action_testing_dpra[1, 0, 0] = j % n_RB
            action_testing_dpra[1, 0, 1] = int(np.floor(j / n_RB))  # power level

            action_testing_dpra[2, 0, 0] = m % n_RB
            action_testing_dpra[2, 0, 1] = int(np.floor(m / n_RB))  # power level

            action_testing_dpra[3, 0, 0] = n % n_RB
            action_testing_dpra[3, 0, 1] = int(np.floor(n / n_RB))  # power level

            V2I_rate_findMax, V2V_rate_findMax = env.Compute_Rate(action_testing_dpra)
            check_sum = np.sum(V2V_rate_findMax)

            action_temp_dpra = action_testing_dpra.copy()
            V2I_rate_dpra, V2V_success_dpra, V2V_rate_dpra = env.act_for_testing_dpra(action_temp_dpra)
            V2I_rate_per_episode_dpra.append(np.sum(V2I_rate_dpra))  # sum V2I rate in bps

            # update the environment and compute interference
            env.renew_channels_fastfading()
            env.Compute_Interference(action_temp)
            env.Compute_Interference_dpra(action_temp_dpra)

            if test_step == n_step_per_episode - 1:
                V2V_success_list.append(V2V_success)
                V2V_success_list_rand.append(V2V_success_rand)
                V2V_success_list_dpra.append(V2V_success_dpra)

        V2I_rate_list.append(np.mean(V2I_rate_per_episode))
        V2I_rate_list_rand.append(np.mean(V2I_rate_per_episode_rand))
        V2I_rate_list_dpra.append(np.mean(V2I_rate_per_episode_dpra))

        print('marl', round(np.average(V2I_rate_per_episode), 2), 'rand', round(np.average(V2I_rate_per_episode_rand), 2), 'dpra', round(np.average(V2I_rate_per_episode_dpra), 2))
        print('marl', V2V_success_list[idx_episode], 'rand', V2V_success_list_rand[idx_episode], 'dpra', V2V_success_list_dpra[idx_episode])

    print('-------- marl -------------')
    print('n_veh:', n_veh, ', n_neighbor:', n_neighbor)
    print('Sum V2I rate:', round(np.average(V2I_rate_list), 2), 'Mbps')
    print('Pr(V2V success):', round(np.average(V2V_success_list), 4))

    print('-------- random -------------')
    print('n_veh:', n_veh, ', n_neighbor:', n_neighbor)
    print('Sum V2I rate:', round(np.average(V2I_rate_list_rand), 2), 'Mbps')
    print('Pr(V2V success):', round(np.average(V2V_success_list_rand), 4))

    print('-------- DPRA -------------')
    print('n_veh:', n_veh, ', n_neighbor:', n_neighbor)
    print('Sum V2I rate:', round(np.average(V2I_rate_list_dpra), 2), 'Mbps')
    print('Pr(V2V success):', round(np.average(V2V_success_list_dpra), 4))

    # The name "DPRA" is used for historical reasons. Not really the case...

    with open(outpath / "Data.txt", "a") as f:
        f.write('-------- marl, ' + args.label + '------\n')
        f.write('n_veh: ' + str(n_veh) + ', n_neighbor: ' + str(n_neighbor) + '\n')
        f.write('Sum V2I rate: ' + str(round(np.average(V2I_rate_list), 5)) + ' Mbps\n')
        f.write('Pr(V2V): ' + str(round(np.average(V2V_success_list), 5)) + '\n')
        f.write('--------random ------------\n')
        f.write('Rand Sum V2I rate: ' + str(round(np.average(V2I_rate_list_rand), 5)) + ' Mbps\n')
        f.write('Rand Pr(V2V): ' + str(round(np.average(V2V_success_list_rand), 5)) + '\n')
        f.write('--------DPRA ------------\n')
        f.write('Dpra Sum V2I rate: ' + str(round(np.average(V2I_rate_list_dpra), 5)) + ' Mbps\n')
        f.write('Dpra Pr(V2V): ' + str(round(np.average(V2V_success_list_dpra), 5)) + '\n')

    np.save(outpath / 'rate_marl.npy', rate_marl)
    np.save(outpath / 'rate_rand.npy', rate_rand)
    np.save(outpath / 'demand_marl.npy', demand_marl)
    np.save(outpath / 'demand_rand.npy', demand_rand)