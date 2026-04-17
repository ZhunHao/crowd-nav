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
from crowd_sim.envs.utils.seeding import seed_everything
from crowd_sim.envs.utils.goal_allocator import GoalAllocator
from crowd_sim.envs.utils.phase_config import PhaseConfig
from crowd_sim.envs.utils.static_map import StaticMap
from crowd_nav.planner.theta_star import NoPathFound, ThetaStar
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
    parser.add_argument('--seed', type=int, default=None,
                        help='RNG seed. Overrides env.config [env] random_seed. Defaults to 42 if unset in both.')
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

    # Resolve seed: CLI flag > env.config > 42 (hardcoded default).
    if args.seed is not None:
        resolved_seed = args.seed
    elif env_config.has_option('env', 'random_seed'):
        resolved_seed = env_config.getint('env', 'random_seed')
    else:
        resolved_seed = 42
    seed_everything(resolved_seed)
    logging.info('Seeded RNGs with %d', resolved_seed)

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

    # ------------------------------------------------------------------
    # WP-2: allocate local goals non-overlapping between start and goal.
    # ------------------------------------------------------------------
    phase_cfg = PhaseConfig.from_configparser(env_config)
    allocator = GoalAllocator(max_tries=phase_cfg.params.max_tries)

    start_point: tuple[float, float] = (env.robot_initx, env.robot_inity)
    global_goal: tuple[float, float] = (env.robot_goalx, env.robot_goaly)

    # WP-3: wire StaticMap.is_free so waypoints avoid obstacles. WP-4 will
    # also inject Theta* as waypoint_source.
    static_map: StaticMap | None = (
        StaticMap.from_static_obstacles(
            env.static_obstacles, margin=phase_cfg.static_map.margin
        )
        if phase_cfg.static_map.enabled and getattr(env, "static_obstacles", None)
        else None
    )
    is_free = static_map.is_free if static_map is not None else None

    # WP-3 hotfix: if the user-supplied global goal is inside an obstacle,
    # project it to the nearest free point before the allocator pins the last
    # waypoint. Keeps env.robot_goal* in sync so the final light_reset uses
    # the projected coord.
    if static_map is not None and not static_map.is_free(
        *global_goal, margin=phase_cfg.static_map.margin
    ):
        projected = static_map.project_to_free(
            *global_goal, margin=phase_cfg.static_map.margin
        )
        logging.warning(
            "Global goal %s lies inside an obstacle; projected to %s",
            global_goal, projected,
        )
        global_goal = projected
        env.robot_goalx, env.robot_goaly = projected

    # WP-4: Theta* global planner. Falls back to straight-line when disabled,
    # when no StaticMap is available, or when NoPathFound is raised at plan
    # time (logged as a WARNING so the run still completes).
    waypoint_source = None
    if static_map is not None and phase_cfg.planner.enabled:
        planner = ThetaStar(
            static_map=static_map,
            inflation=phase_cfg.planner.inflation_radius,
            grid_resolution=phase_cfg.planner.grid_resolution,
            bounds=phase_cfg.planner.bounds,
            simplify=phase_cfg.planner.waypoint_simplify,
        )
        waypoint_source = planner.as_waypoint_source(phase_cfg.params.num_waypoints)

    try:
        waypoints: list[tuple[float, float]] = allocator.allocate_waypoints(
            start=start_point,
            goal=global_goal,
            num_waypoints=phase_cfg.params.num_waypoints,
            min_inter_dist=phase_cfg.params.min_inter_waypoint_dist,
            is_free=is_free,
            waypoint_source=waypoint_source,
        )
    except NoPathFound as exc:
        logging.warning(
            "Theta* found no path (%s); falling back to straight-line waypoints.",
            exc,
        )
        waypoints = allocator.allocate_waypoints(
            start=start_point,
            goal=global_goal,
            num_waypoints=phase_cfg.params.num_waypoints,
            min_inter_dist=phase_cfg.params.min_inter_waypoint_dist,
            is_free=is_free,
            waypoint_source=None,
        )
    logging.info(
        "Allocated %d waypoints (start=%s goal=%s)",
        len(waypoints), start_point, global_goal,
    )

    position_list: list[list[float]] = []
    prev_x, prev_y = start_point
    for goal_idx, (goal_x, goal_y) in enumerate(waypoints):
        start_x, start_y = (prev_x, prev_y) if goal_idx == 0 else (
            position_list[-1][0], position_list[-1][1]
        )
        logging.info("start: %.2f %.2f goal: %.2f %.2f", start_x, start_y, goal_x, goal_y)

        # The env.light_reset seam (SYSTEM_DESIGN §5.4) expects these 4 writes.
        env.robot_goalx = goal_x
        env.robot_goaly = goal_y
        env.robot_initx = start_x
        env.robot_inity = start_y

        if args.visualize:
            env.local_goal = [goal_x, goal_y]
            ob = env.light_reset(args.phase, args.test_case)
            last_pos = np.array(robot.get_position())
            done = False
            while not done:
                action = robot.act(ob)
                ob, _, done, info = env.step(action)
                current_pos = np.array(robot.get_position())
                logging.debug(
                    "Speed: %.2f",
                    np.linalg.norm(current_pos - last_pos) / robot.time_step,
                )
                last_pos = current_pos
                position_list.append(last_pos.tolist())
            env.curr_post = last_pos

            logging.info(
                "It takes %.2f seconds to finish. Final status is %s",
                env.global_time, info,
            )
            if robot.visible and info == "reach goal":
                human_times = env.get_human_times()
                logging.info(
                    "Average time for humans to reach goal: %.2f",
                    sum(human_times) / len(human_times),
                )
        else:
            explorer.run_k_episodes(env.case_size[args.phase], args.phase, print_failure=True)

    env.render("video", [list(w) for w in waypoints], args.video_file)

        

if __name__ == '__main__':
    main()
