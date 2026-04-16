import logging
import argparse
import configparser
import os
import torch
import numpy as np
import gym
from crowd_nav.utils.explorer import Explorer
from crowd_nav.policy.policy_factory import policy_factory
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.policy.orca import ORCA


def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--env_config', type=str, default='configs/env.config')
    parser.add_argument('--policy_config', type=str, default='configs/policy.config')
    parser.add_argument('--policy', type=str, default='orca')
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--il', default=False, action='store_true')
    parser.add_argument('--gpu', default=False, action='store_true')
    parser.add_argument('--visualize', default=False, action='store_true')
    parser.add_argument('--phase', type=str, default='test')
    parser.add_argument('--test_case', type=int, default=None)
    parser.add_argument('--square', default=False, action='store_true')
    parser.add_argument('--circle', default=False, action='store_true')
    parser.add_argument('--video_file', type=str, default=None)
    parser.add_argument('--traj', default=False, action='store_true')
    args = parser.parse_args()

    if args.model_dir is not None:
        env_config_file = os.path.join(args.model_dir, os.path.basename(args.env_config))
        policy_config_file = os.path.join(args.model_dir, os.path.basename(args.policy_config))
        if args.il:
            model_weights = os.path.join(args.model_dir, 'il_model.pth')
        else:
            if os.path.exists(os.path.join(args.model_dir, 'resumed_rl_model.pth')):
                model_weights = os.path.join(args.model_dir, 'resumed_rl_model.pth')
            else:
                model_weights = os.path.join(args.model_dir, 'rl_model.pth')
    else:
        env_config_file = args.env_config
        policy_config_file = args.env_config

    # configure logging and device
    logging.basicConfig(level=logging.INFO, format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S")
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
    logging.info('Using device: %s', device)

    # configure policy
    policy = policy_factory[args.policy]()
    policy_config = configparser.RawConfigParser()
    policy_config.read(policy_config_file)
    policy.configure(policy_config)
    if policy.trainable:
        if args.model_dir is None:
            parser.error('Trainable policy must be specified with a model weights directory')
        policy.get_model().load_state_dict(torch.load(model_weights))

    # configure environment
    env_config = configparser.RawConfigParser()
    env_config.read(env_config_file)
    
    # load environment
    env = gym.make('CrowdSim-v0')
    env.local_goal = [0,0]
    env.curr_post = [-11,-11]
    env.configure(env_config)
    if args.square:
        env.test_sim = 'square_crossing'
    if args.circle:
        env.test_sim = 'circle_crossing'
    robot = Robot(env_config, 'robot')
    robot.set_policy(policy)
    env.set_robot(robot)
    explorer = Explorer(env, robot, device, gamma=0.9)

    policy.set_phase(args.phase)
    policy.set_device(device)
    # set safety space for ORCA in non-cooperative simulation
    if isinstance(robot.policy, ORCA):
        if robot.visible:
            robot.policy.safety_space = 0
        else:
            # because invisible case breaks the reciprocal assumption
            # adding some safety space improves ORCA performance. Tune this value based on your need.
            robot.policy.safety_space = 0
        logging.info('ORCA agent buffer: %f', robot.policy.safety_space)

    policy.set_env(env)
    robot.print_info()
    
    ob = env.reset(args.phase, args.test_case)
    
    # define the multiple local goals
    start_point = [-11,-11]
    goal_list = [[-5,-9], [0, -10], [6,-9], [10,-5], [5,0], [-2,-1], [-8,2], [-7,10], [0,11], [6,11]]
    position_list = []

    start_x,start_y,goal_x,goal_y = None,None,None,None
    for goal_idx in range(len(goal_list)):
        goal_x, goal_y = goal_list[goal_idx][0], goal_list[goal_idx][1]

        if start_x is None and start_y is None:
            start_x, start_y = start_point[0], start_point[1]
        else:
            start_x, start_y = position_list[-1][0], position_list[-1][1]
        print('start:', start_x,start_y, 'goal:', goal_x, goal_y)

        # set the local goal using the given list
        env.robot_goalx = goal_x
        env.robot_goaly = goal_y
        env.robot_initx = start_x
        env.robot_inity = start_y
        
        # env.robot.set(env.robot_initx, env.robot_inity, env.robot_goalx, env.robot_goaly, 0, 0, np.pi / 2)

        if args.visualize:
            env.local_goal = [goal_x, goal_y]
            ob = env.light_reset(args.phase, args.test_case)
            last_pos = np.array(robot.get_position())
            print(last_pos)
            done = False
            while not done:
                action = robot.act(ob)
                ob, _, done, info = env.step(action)
                current_pos = np.array(robot.get_position())
                logging.debug('Speed: %.2f', np.linalg.norm(current_pos - last_pos) / robot.time_step)
                last_pos = current_pos
                position_list.append(last_pos.tolist())
            # if args.traj:
            #     env.render('traj', args.video_file)
            # else:
            #     env.render('video', args.video_file)
            env.curr_post = last_pos

            logging.info('It takes %.2f seconds to finish. Final status is %s', env.global_time, info)
            if robot.visible and info == 'reach goal':
                human_times = env.get_human_times()
                logging.info('Average time for humans to reach goal: %.2f', sum(human_times) / len(human_times))
        else:
            explorer.run_k_episodes(env.case_size[args.phase], args.phase, print_failure=True)

        # break
    env.render('video', goal_list, args.video_file)

        

if __name__ == '__main__':
    main()
