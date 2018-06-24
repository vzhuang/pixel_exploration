import gym
import gym.spaces
import itertools

import sys
import random
from collections import namedtuple

import torch
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from torchvision import utils
from utils.updated_replay import ReplayBuffer, MMCReplayBuffer
from utils.schedule import LinearSchedule
from utils.gym_atari_wrappers import get_wrapper_by_name
from torch.autograd import Variable
import pickle
import logging
import time
import tensorflow as tf


OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs"])

Statistic = {
    "mean_episode_rewards": [],
    "best_mean_episode_rewards": [],
    "episode_rewards": []
}

# use GPU if available
USE_CUDA = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor
Tensor = FloatTensor


def dqn_learn(env, q_func, optimizer_spec, density, cnn_kwargs, config,
              exploration=LinearSchedule(1000000, 0.1), stopping_criterion=None):
    """
    Run Deep Q-learning algorithm.
    """
    # this is just to make sure that you're operating in the correct environment
    assert type(env.observation_space) == gym.spaces.Box
    assert type(env.action_space) == gym.spaces.Discrete

    ###############
    # BUILD MODEL #
    ###############
    if len(env.observation_space.shape) == 1:
        # This means we are running on low-dimensional observations (e.g. RAM)
        input_shape = env.observation_space.shape
    else:
        img_h, img_w, img_c = env.observation_space.shape
        input_shape = (img_h, img_w, config.frame_history_len * img_c)
    num_actions = env.action_space.n

    # define Q network and target network (instantiate 2 DQN's)
    in_channel = input_shape[-1]
    Q = q_func(in_channel, num_actions)
    target_Q = deepcopy(Q)

    # define C network and target C
    C = q_func(in_channel, num_actions)
    target_C = deepcopy(C)


    # call tensorflow wrapper to get density model
    if config.bonus:
        tf_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        tf_config.gpu_options.allow_growth=True
        sess = tf.Session(config=tf_config)
        pixel_bonus = density(cnn_kwargs, sess, num_actions)
        tf.initialize_all_variables().run(session=sess)

    if USE_CUDA:
        Q.cuda()
        target_Q.cuda()
        C.cuda()
        target_C.cuda()

    # define eps-greedy exploration strategy
    def select_action(model, bonus_model, obs, t):
        """
        Selects random action w prob eps; otherwise returns best action
        :param exploration:
        :param t:
        :return:
        """
        def get_best_action(obs):
            obs = torch.from_numpy(obs).type(FloatTensor).unsqueeze(0) / 255.0
            Q_val = model(Variable(obs, volatile=True))
            C_val = bonus_model(Variable(obs, volatile=True))
            b = C_val
            if config.gaussian_ts:
                b = config.alpha * torch.distributions.normal.Normal(0, C_val)
            return (Q_val + b).data.max(1)[1].view(1, 1)
        if config.egreedy_exploration:
            sample = random.random()
            eps_threshold = exploration.value(t)
            if sample > eps_threshold:                   
                return get_best_action(obs)
            else:
                # return random action
                return LongTensor([[random.randrange(num_actions)]])
        # no exploration; just take best action
        else:
            return get_best_action(obs)
    # construct torch optimizer
    optimizer = optimizer_spec.constructor(Q.parameters(), **optimizer_spec.kwargs)

    # C optimizer
    C_optimizer = optimizer_spec.constructor(C.parameters(), **optimizer_spec.kwargs)
    
    # construct the replay buffer
    if config.mmc:
        replay_buffer = MMCReplayBuffer(config.replay_buffer_size, config.frame_history_len)
    else:
        replay_buffer = ReplayBuffer(config.replay_buffer_size, config.frame_history_len)

    ###############
    # RUN ENV     #
    ###############
    num_param_updates = 0
    mean_episode_reward = -float('nan')
    best_mean_episode_reward = -float('inf')
    last_obs = env.reset()
    prev = time.time()

    # index trackers for updating mc returns
    episode_indices_in_buffer = []
    reward_each_timestep = []
    timesteps_in_buffer = []
    cur_timestep = 0

    # t denotes frames
    for t in itertools.count():
        ### 1. Check stopping criterion
        if stopping_criterion is not None and stopping_criterion(env, t):
            break

        ### 2. Step the env and store the transition
        # process last_obs to include context from previous frame
        last_idx = replay_buffer.store_frame(last_obs)

        # record where this is in the buffer
        episode_indices_in_buffer.append(last_idx)
        timesteps_in_buffer.append(cur_timestep)
        # one more step in episode
        cur_timestep += 1

        # take latest observation pushed into buffer and compute corresponding input
        # that should be given to a Q network by appending some previous frames
        recent_obs = replay_buffer.encode_recent_observation()
        # recent_obs.shape is also (84, 84, 4)

        # choose random action if not yet started learning
        if t > config.learning_starts:
            action = select_action(Q, C, recent_obs, t)[0][0]
        else:
            action = random.randrange(num_actions)

        # advance one step
        obs, reward, done, _ = env.step(action)
        # clip reward to be in [-1, +1]
        reward = max(-1.0, min(reward, 1.0))

        ###############################################
        # do density model stuff here
        if config.bonus: # just assume this is true
            intrinsic_reward = pixel_bonus.bonus(obs, action, t, num_actions)
            if t % config.log_freq == 0:
                logging.info('t: {}\t intrinsic reward: {}'.format(t, intrinsic_reward))
                curr = time.time()
                diff = curr - prev
                prev = curr
                logging.info("Timestep %d" % (t,))
                logging.info("Time elapsed %f" % diff)
                # utils.save_image(pixel_bonus.sample_images(img_dim**2), 'images/iteration_{}.png'.format(t), nrow=img_dim, padding=0)
                # pixel_bonus.sample_images(3, t)                
                # utils.save_image(frame / 8.,'images/obs_{}.png'.format(t),padding=0)
            bonus = intrinsic_reward
            # TODO: add bonus/intrinsic_reward to replay buffer
            pixel_bonus.writer.add_scalar('data/bonus', bonus, t)
            # add intrinsic reward to clipped reward
            # NOTE: don't add bonus since we separate Q and C
            reward += intrinsic_reward
            # clip reward to be in [-1, +1] once again
            reward = max(-1.0, min(reward, 1.0))
            assert -1.0 <= reward <= 1.0
        ################################################

        # store reward in list to use for calculating MMC update
        reward_each_timestep.append(reward)
        replay_buffer.store_effect(last_idx, action, reward, done, bonus)

        # reset environment when reaching episode boundary
        if done:
            # only if computing MC return
            if config.mmc:
                # episode has terminated --> need to do MMC update here
                # loop through all transitions of this past episode and add in mc_returns
                print(len(timesteps_in_buffer), len(reward_each_timestep))
                assert len(timesteps_in_buffer) == len(reward_each_timestep)
                mc_returns = np.zeros(len(timesteps_in_buffer))

                # compute mc returns
                r = 0
                for i in reversed(range(len(mc_returns))):
                    r = reward_each_timestep[i] + config.gamma * r
                    mc_returns[i] = r

                # populate replay buffer
                for j in range(len(mc_returns)):
                    # get transition tuple in reward buffer and update
                    update_idx = episode_indices_in_buffer[j]
                    # put mmc return back into replay buffer
                    replay_buffer.mc_return_t[update_idx] = mc_returns[j]
            # reset because end of episode
            episode_indices_in_buffer = []
            timesteps_in_buffer = []
            cur_timestep = 0
            reward_each_timestep = []

            # reset
            obs = env.reset()
        last_obs = obs

        ### 3. Perform experience replay and train the network.
        # note that this is only done if the replay buffer contains enough samples
        # for us to learn something useful -- until then, the model will not be
        # initialized and random actions should be taken

        # perform training
        if (t > config.learning_starts and t % config.learning_freq == 0 and
                replay_buffer.can_sample(config.batch_size)):

            # sample batch of transitions
            if config.mmc:
                # also grab MMC batch if computing MMC return
                obs_batch, act_batch, rew_batch, next_obs_batch, bonus_batch, done_mask, mc_batch = \
                replay_buffer.sample(config.batch_size)
                mc_batch = Variable(torch.from_numpy(mc_batch).type(FloatTensor))
            else:
                obs_batch, act_batch, rew_batch, next_obs_batch, bonus_batch, done_mask = \
                replay_buffer.sample(config.batch_size)

            # convert variables to torch tensor variables
            obs_batch = Variable(torch.from_numpy(obs_batch).type(FloatTensor)/255.0)
            act_batch = Variable(torch.from_numpy(act_batch).type(LongTensor))
            rew_batch = Variable(torch.from_numpy(rew_batch).type(FloatTensor))
            next_obs_batch = Variable(torch.from_numpy(next_obs_batch).type(FloatTensor)/255.0)
            bonus_batch = Variable(torch.from_numpy(bonus_batch).type(FloatTensor))
            not_done_mask = Variable(torch.from_numpy(1 - done_mask).type(FloatTensor))

            # 3.c: train the model: perform gradient step and update the network
            current_Q_values = Q(obs_batch).gather(1, act_batch.unsqueeze(1)).squeeze()
            # this gives you a FloatTensor of size 32 // gives values of max
            next_max_q = target_Q(next_obs_batch).detach().max(1)[0]

            # torch.FloatTensor of size 32
            next_Q_values = not_done_mask * next_max_q

            # this is [r(x,a) + gamma * max_a' Q(x', a')]
            target_Q_values = rew_batch + (config.gamma * next_Q_values)

            if config.mmc:
                # replace target_Q_values with mixed target
                target_Q_values = ((1-config.beta) * target_Q_values) + (config.beta *
                                                                         mc_batch)
            # use huber loss
            loss = F.smooth_l1_loss(current_Q_values, target_Q_values)

            # zero out gradient
            optimizer.zero_grad()

            # backward pass
            loss.backward()

            # gradient clipping
            for params in Q.parameters():
                params.grad.data.clamp_(-1, 1)

            # perform param update
            optimizer.step()
            num_param_updates += 1

            # periodically update the target network
            if num_param_updates % config.target_update_freq == 0:
                target_Q = deepcopy(Q)

            ######### REPEAT ABOVE FOR C NETWORK ##################
            current_C_values = C(obs_batch).gather(1, act_batch.unsqueeze(1)).squeeze()
            # this gives you a FloatTensor of size 32 // gives values of max
            next_max_c = target_C(next_obs_batch).detach().max(1)[0]

            # torch.FloatTensor of size 32
            next_C_values = not_done_mask * next_max_c

            # this is [r(x,a) + gamma * max_a' Q(x', a')]
            target_C_values = bonus_batch + (config.gamma * next_C_values)

            #if config.mmc:
                # replace target_Q_values with mixed target
            #    target_C_values = ((1-config.beta) * target_C_values) + (config.beta *
            #                                                             mc_batch)
            # use huber loss
            C_loss = F.smooth_l1_loss(current_C_values, target_C_values)

            # zero out gradient
            C_optimizer.zero_grad()

            # backward pass
            C_loss.backward()

            # gradient clipping
            for params in C.parameters():
                params.grad.data.clamp_(-1, 1)

            # perform param update
            C_optimizer.step()
            num_param_updates += 1

            # periodically update the target network
            if num_param_updates % config.target_update_freq == 0:
                target_C = deepcopy(C)            

            ### 4. Log progress
            episode_rewards = get_wrapper_by_name(env, "Monitor").get_episode_rewards()
            if len(episode_rewards) > 0:
                mean_episode_reward = np.mean(episode_rewards[-100:])
            if len(episode_rewards) > 100:
                best_mean_episode_reward = max(best_mean_episode_reward, mean_episode_reward)
                
            # Tensorboard logging
            pixel_bonus.writer.add_scalar('data/bonus', intrinsic_reward, t)
            pixel_bonus.writer.add_scalar('data/Q_loss', loss, t)
            #pixel_bonus.writer.add_scalar('data/C_loss', C_loss, t)
            pixel_bonus.writer.add_scalar('data/episode_reward', episode_rewards[-1], t)

            # save statistics
            Statistic["mean_episode_rewards"].append(mean_episode_reward)
            Statistic["best_mean_episode_rewards"].append(best_mean_episode_reward)
            Statistic["episode_rewards"].append(episode_rewards)

            if t % config.log_freq == 0 and t > config.learning_starts:
                # curr = time.time()
                # diff = curr - prev
                # prev = curr
                # logging.info("Timestep %d" % (t,))
                # logging.info("Time elapsed %f" % diff)
                logging.info("mean reward (100 episodes) %f" % mean_episode_reward)
                logging.info("best mean reward %f" % best_mean_episode_reward)
                logging.info("episodes %d" % len(episode_rewards))
                logging.info("exploration %f" % exploration.value(t))                
                sys.stdout.flush()                

                # Dump statistics to pickle
            # if t % 1000000 == 0 and t > config.learning_starts:
            #     with open(config.output_path + 'statistics.pkl', 'wb') as f:
            #         pickle.dump(Statistic, f)
