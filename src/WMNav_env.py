import gzip
import json
import logging
import math
import os
import random
import requests
import traceback
import habitat_sim

import pandas as pd
import numpy as np

from PIL import Image
from simWrapper import PolarAction, SimWrapper
from WMNav_agent import *
from custom_agent import *
from utils import *

class Env:
    """
    Base class for creating an environment for embodied navigation tasks.
    This class defines the setup, logging, running, and evaluation of episodes.
    """

    task = 'Not defined'

    def __init__(self, cfg: dict):
        """
        Initializes the environment with the provided configuration.

        Args:
            cfg (dict): Configuration dictionary containing environment, simulation, and agent settings.
        """
        self.cfg = cfg['env_cfg']
        self.sim_cfg = cfg['sim_cfg']
        if self.cfg['name'] == 'default':
            self.cfg['name'] = f'default_{random.randint(0, 1000)}'
        self._initialize_logging(cfg)
        self._initialize_agent(cfg)
        self.outer_run_name = self.task + '_' + self.cfg['name']
        self.inner_run_name = f'{self.cfg["instance"]}_of_{self.cfg["instances"]}'
        self.curr_run_name = "Not initialized"
        self.path_calculator = habitat_sim.MultiGoalShortestPath()
        self.simWrapper = None  # 修改self.simWrapper: SimWrapper = None
        self.num_episodes = 0
        self._initialize_experiment()

    def _initialize_agent(self, cfg: dict):
        """Initializes the agent for the environment."""
        PolarAction.default = PolarAction(cfg['agent_cfg']['default_action'], 0, 'default')
        cfg['agent_cfg']['sensor_cfg'] = cfg['sim_cfg']['sensor_cfg']
        agent_cls = globals()[cfg['agent_cls']]
        self.agent: Agent = agent_cls(cfg['agent_cfg'])
        self.agent_cls = cfg['agent_cls']

    def _initialize_logging(self, cfg: dict):
        """
        Initializes logging for the environment.

        Args:
            cfg (dict): Configuration dictionary containing logging settings.
        """
        self.log_file = os.path.join(os.environ.get("LOG_DIR"), f'{cfg["task"]}_{self.cfg["name"]}/{self.cfg["instance"]}_of_{self.cfg["instances"]}.txt')
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        if self.cfg['parallel']:
            logging.basicConfig(
                filename=self.log_file,
                level=logging.INFO,
                format='%(asctime)s %(levelname)s: %(message)s'
            )
        else:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s %(levelname)s: %(message)s'
            )

    def _initialize_experiment(self):
        """
        Abstract method for setting up the environment and initializing all required variables.
        Should be implemented in derived classes.
        """
        raise NotImplementedError

    def run_experiment(self):
        """
        Runs the experiment by iterating over episodes.
        """
        instance_size = math.ceil(self.num_episodes / self.cfg['instances'])  # 1000
        start_ndx = self.cfg['instance'] * instance_size
        end_ndx = self.num_episodes

        for episode_ndx in range(start_ndx, min(start_ndx + self.cfg['num_episodes'], end_ndx)):

            self.wandb_log_data = {
                'episode_ndx': episode_ndx,  # 0
                'instance': self.inner_run_name,  # 0_of_1
                'total_episodes': self.cfg['instances'] * self.cfg['num_episodes'],  # 1
                'task': self.task,  # ObjectNav
                'task_data': {},
                'spl': 0,
                'goal_reached': False
            }

            try:
                self._run_episode(episode_ndx)
            except Exception as e:
                log_exception(e)
                self.simWrapper.reset()


    def _run_episode(self, episode_ndx: int):
        """
        Runs a single episode.p

        Args:
            episode_ndx (int): The index of the episode to run.
        """
        obs = self._initialize_episode(episode_ndx)  # color_sensor(1080, 1920, 4) depth_sensor(1080, 1920) agent_state[position rotation sensor_states]

        logging.info(f'\n===================STARTING RUN: {self.curr_run_name} ===================\n')
        for _ in range(self.cfg['max_steps']):
            try:
                agent_action = self._step_env(obs)  # 根据单张RGB图片、深度图和agent以及相机位姿确定agent的下一步动作，保存运行结果
                if agent_action is None:
                    break
                obs = self.simWrapper.step(agent_action)  # 执行操作，更新agent的状态和观察

            except Exception as e:
                log_exception(e)

            finally:
                self.step += 1
        self._post_episode()

    def _initialize_episode(self, episode_ndx: int):
        """
        Initializes the episode. This method should be implemented in derived classes.

        Args:
            episode_ndx (int): The index of the episode to initialize.
        """
        self.step = 0
        self.init_pos = None
        self.df = pd.DataFrame({})
        self.agent_distance_traveled = 0
        self.prev_agent_position = None

    def _step_env(self, obs: dict):
        """
        Takes a step in the environment. This method should be implemented in derived classes.

        Args:
            obs (dict): The current observation. Contains agent state and sensor observations.

        Returns:
            PolarAction: The next action to be taken by the agent.
        """
        logging.info(f'Step {self.step}')
        agent_state = obs['agent_state']
        if self.prev_agent_position is not None:
            self.agent_distance_traveled += np.linalg.norm(agent_state.position - self.prev_agent_position)
        self.prev_agent_position = agent_state.position

        return None

    def _post_episode(self):
        """
        Called after the episode is complete, saves the dataframe log, and resets the environment.
        Sends a request to the aggregator server if parallel is set to True.
        """
        self.df.to_pickle(os.path.join(os.environ.get("LOG_DIR"), f'{self.outer_run_name}/{self.inner_run_name}/{self.curr_run_name}/df_results.pkl'))
        self.simWrapper.reset()
        self.agent.reset()
        if self.cfg['parallel']:
            try:
                self.wandb_log_data['spend'] = self.agent.get_spend()
                self.wandb_log_data['default_rate'] = len(self.df[self.df['success'] == 0]) / len(self.df)
                response = requests.post(f'http://localhost:{self.cfg["port"]}/log', json=self.wandb_log_data)
                if response.status_code != 200:
                    logging.error(f"Failed to send metrics: {response.text}")
            except Exception as e:
                tb = traceback.extract_tb(e.__traceback__)
                for frame in tb:
                    logging.error(f"Frame {frame.filename} line {frame.lineno}")
                logging.error(e)

        logging.info(f"Success: {self.wandb_log_data['goal_reached']}")
        logging.info('\n===================RUN COMPLETE===================\n')
        if self.cfg['log_freq'] == 1:
            create_gif(
                os.path.join(os.environ.get("LOG_DIR"), f'{self.outer_run_name}/{self.inner_run_name}/{self.curr_run_name}'), self.agent.cfg['sensor_cfg']['img_height'], self.agent.cfg['sensor_cfg']['img_width'], agent_cls=self.agent_cls
            )
            create_gif_voxel(
                os.path.join(os.environ.get("LOG_DIR"), f'{self.outer_run_name}/{self.inner_run_name}/{self.curr_run_name}'),
                1800, 1800
            )

    def _log(self, images: dict, step_metadata: dict, logging_data: dict):
        """
        Appends the step metadata to the dataframe, and saves the images and general metadata to disk.

        Args:
            images (dict): Images generated during the step.
            step_metadata (dict): Metadata for the current step.
            logging_data (dict): General logging data.
        """
        self.df = pd.concat([self.df, pd.DataFrame([step_metadata])], ignore_index=True)

        if self.step % self.cfg['log_freq'] == 0 or step_metadata['success'] == 0:
            path = os.path.join(os.environ.get("LOG_DIR"), f'{self.outer_run_name}/{self.inner_run_name}/{self.curr_run_name}/step{self.step}')
            if not step_metadata['success']:
                path += '_ERROR'
            os.makedirs(path, exist_ok=True)
            for name, im in images.items():
                if im is not None:
                    im = Image.fromarray(im[:, :, 0:3], mode='RGB')
                    im.save(f'{path}/{name}.png')
            with open(f'{path}/details.txt', 'w') as file:
                if step_metadata['success']:
                    for k, v in logging_data.items():
                        file.write(f'{k}\n{v}\n\n')

    def _calculate_metrics(self, agent_state: habitat_sim.AgentState, agent_action: PolarAction, geodesic_path: int, max_steps: int):
        """
        Calculates the navigation metrics at a given step.

        Args:
            agent_state: The state of the agent.
            agent_action: The action taken by the agent.
            geodesic_path: The shortest path to the goal.
            max_steps (int): Maximum steps allowed for the episode.

        Returns:
            dict: A dictionary containing calculated metrics.
        """
        metrics = {}
        self.path_calculator.requested_start = agent_state.position
        metrics['distance_to_goal'] = self.simWrapper.get_path(self.path_calculator)
        metrics['spl'] = 0
        metrics['goal_reached'] = False
        metrics['done'] = False
        metrics['finish_status'] = 'running'

        if agent_action is PolarAction.stop or self.step + 1 == max_steps:
            metrics['done'] = True

            if metrics['distance_to_goal'] < self.cfg['success_threshold']:
                metrics['finish_status'] = 'success'
                metrics['goal_reached'] = True
                metrics['spl'] = geodesic_path / max(geodesic_path, self.agent_distance_traveled)
                self.wandb_log_data.update({
                    'spl': metrics['spl'],
                    'goal_reached': metrics['goal_reached']
                })
            else:
                if agent_action is PolarAction.stop:
                    metrics['finish_status'] = 'fp'
                else:
                    metrics['finish_status'] = 'max_steps'

        return metrics

class WMNavEnv(Env):

    task = 'ObjectNav'

    def _initialize_experiment(self):
        """
        Initializes the experiment by setting up the dataset split, scene configuration, and goals.
        """
        self.all_episodes = []
        if self.cfg['dataset']  == 'hm3d_v0.1':
            scene_config_path = 'hm3d_v0.1/hm3d_annotated_basis.scene_dataset_config.json'
            objnav_path = 'objectnav_hm3d_v1'
        elif self.cfg['dataset']  == 'hm3d_v0.2':
            scene_config_path = 'hm3d_v0.2/hm3d_annotated_basis.scene_dataset_config.json'
            objnav_path = 'objectnav_hm3d_v2'
        elif self.cfg['dataset']  == 'mp3d':
            scene_config_path = 'mp3d/mp3d_annotated_basis.scene_dataset_config.json'
            objnav_path = 'objectnav_mp3d'
        else:
            raise ValueError('Dataset type must be hm3d_v0.1, hm3d_v0.2, or mp3d')

        self.sim_cfg['scene_config'] = os.path.join(os.environ.get("DATASET_ROOT"), scene_config_path)
        self.goals = {}

        for f in sorted(os.listdir(os.path.join(os.environ.get("DATASET_ROOT"), objnav_path, f'{self.cfg["split"]}/content'))):
            with gzip.open(os.path.join(os.environ.get("DATASET_ROOT"), objnav_path, f'{self.cfg["split"]}/content/{f}'), 'rt') as gz:
                js = json.load(gz)
                hsh = f.split('.')[0]
                self.goals[hsh] = js['goals_by_category']
                self.all_episodes += js['episodes']

        self.num_episodes = len(self.all_episodes)

    def _initialize_episode(self, episode_ndx: int):
        """
        Initializes the episode for the BASE task.

        Args:
            episode_ndx (int): The index of the episode to initialize.
        """
        super()._initialize_episode(episode_ndx)
        episode = self.all_episodes[episode_ndx]
        if 'hm3d' in self.cfg['dataset']:
            f = episode['scene_id'].split('/')[1:]
            self.sim_cfg['scene_id'] = f[1][2:5]
            self.sim_cfg['scene_path'] = os.path.join(os.environ.get("DATASET_ROOT"), 'hm3d_v0.1' if self.cfg['dataset'] == 'hm3d_v0.1' else 'hm3d_v0.2', f'{self.cfg["split"]}/{f[1]}/{f[2]}')
            self.simWrapper = SimWrapper(self.sim_cfg)

            goals = self.goals[f[1][6:]]
            all_objects = goals[f'{f[-1]}_{episode["object_category"]}']
        elif 'mp3d' in self.cfg['dataset']:
            self.sim_cfg['scene_id'] = episode['scene_id'].split('/')[1]
            self.sim_cfg['scene_path'] = os.path.join(os.environ.get("DATASET_ROOT"), f'{episode["scene_id"]}')
            self.simWrapper = SimWrapper(self.sim_cfg)

            goals = self.goals[self.sim_cfg['scene_id']]
            all_objects = goals[f'{episode["scene_id"].split("/")[2]}_{episode["object_category"]}']
        else:
            raise ValueError('Dataset type must be hm3d_v0.1, hm3d_v0.2, or mp3d')
        view_positions = []
        for obj in all_objects:
            for vp in obj['view_points']:
                view_positions.append(vp['agent_state']['position'])
        self.path_calculator.requested_ends = np.array(view_positions, dtype=np.float32)
        logging.info(f'RUNNING EPISODE {episode_ndx} with {episode["object_category"]} and {len(all_objects)} instances. GEODESIC DISTANCE: {episode["info"]["geodesic_distance"]}')
        if episode['object_category'] == 'tv_monitor':
            episode['object_category'] = 'tv screen'
        self.current_episode = {
            'object': episode['object_category'],
            'shortest_path': episode['info']['geodesic_distance'],
            'object_positions': [a['position'] for a in all_objects],
            'view_positions': view_positions
        }
        self.init_pos = np.array(episode['start_position'])
        self.simWrapper.set_state(pos=self.init_pos, quat=episode['start_rotation'])
        self.curr_run_name = f"{episode_ndx}_{self.simWrapper.scene_id}"

        obs = self.simWrapper.step(PolarAction.null)

        self.previous_subtask = '{}'  # Initialize the last subtask with an empty dictionary
        return obs

    def _step_env(self, obs: dict):
        """
        Takes a step in the environment for the BASEV1 task.

        Args:
            obs (dict): The current observation.

        Returns:
            list: The next action to be taken by the agent.
        """
        episode_images = [(obs['color_sensor'].copy())[:, :, :3]]
        color_origin = episode_images[0]

        loop_action_clockwise = PolarAction(0, -0.167 * np.pi)
        loop_action_counterclock = PolarAction(0, 0.167 * np.pi)

        #  确定目标方向
        for _ in range(11):
            obs = self.simWrapper.step(loop_action_clockwise)
            if _ % 2 == 0:
                self.agent.navigability(obs, _+1)
            episode_images.append((obs['color_sensor'].copy())[:, :, :3])
        nav_map = self.agent.generate_voxel(obs['agent_state'])
        panoramic_image, explorable_value, reason = self.agent.make_curiosity_value(episode_images[-12:], self.current_episode['object'])
        goal_rotate, goal_reason = self.agent.update_curiosity_value(explorable_value, reason)

        direction_image = episode_images[-12:][goal_rotate]
        goal_flag, subtask = self.agent.make_plan(direction_image, self.previous_subtask, goal_reason, self.current_episode['object'])
        self.previous_subtask = subtask # update last subtask
        #  转向目标方向
        for j in range(min(11 - goal_rotate, 1 + goal_rotate)):
            if goal_rotate <= 6:
                obs = self.simWrapper.step(loop_action_clockwise)
            else:
                obs = self.simWrapper.step(loop_action_counterclock)

        cvalue_map = self.agent.draw_cvalue_map(obs['agent_state'])

        super()._step_env(obs)
        obs['goal'] = self.current_episode['object']  # 目标的类别，最短距离，目标位置，所有可到点
        obs['subtask'] = subtask  # 子目标
        obs['goal_flag'] = goal_flag  # 是否发现目标
        agent_state = obs['agent_state']
        self.agent_distance_traveled += np.linalg.norm(agent_state.position - self.prev_agent_position)
        self.prev_agent_position = agent_state.position
        agent_action, metadata = self.agent.step(obs)  # 整个模型前向运行一次，返回动作和结果
        step_metadata = metadata['step_metadata']
        metadata['logging_data']['EVALUATOR_RESPONSE'] = str({'goal_rotate':goal_rotate*30, 'explorable_value': explorable_value, 'reason': reason})
        metadata['logging_data']['PLANNING_RESPONSE'] = str({'goal_flag': goal_flag, 'subtask': subtask})
        logging_data = metadata['logging_data']

        images = metadata['images']

        if metadata['step'] is not None:
            step_text = f"step {metadata['step']}"
            color_origin = np.ascontiguousarray(color_origin)
            color_origin = cv2.putText(color_origin, step_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        if obs['goal'] is not None:
            scale_factor = color_origin.shape[0] / 1080
            padding = 20
            text_size = 2.5 * scale_factor
            text_thickness = 2
            (text_width, text_height), _ = cv2.getTextSize(f"goal:{obs['goal']}", cv2.FONT_HERSHEY_SIMPLEX, text_size, text_thickness)
            text_position = (color_origin.shape[1] - text_width - padding, padding + text_height)
            cv2.putText(color_origin, f"goal:{obs['goal']}", text_position, cv2.FONT_HERSHEY_SIMPLEX, text_size, (255, 0, 0), text_thickness,
                        cv2.LINE_AA)

        planner_images = {'panoramic': panoramic_image,
                          'color_origin': color_origin,
                          'nav_map': nav_map,
                          'cvalue_map': cvalue_map}
        images.update(planner_images)  # 保存规划过程的图片

        metrics = self._calculate_metrics(agent_state, agent_action, self.current_episode['shortest_path'], self.cfg['max_steps'])
        step_metadata.update(metrics)

        self._log(images, step_metadata, logging_data)

        if metrics['done']:
            agent_action = None

        return agent_action

    def _post_episode(self):
        """
        Called after the episode is complete, saves the dataframe log, and resets the environment.
        Sends a request to the aggregator server if parallel is set to True.
        """
        self.df.to_pickle(os.path.join(os.environ.get("LOG_DIR"), f'{self.outer_run_name}/{self.inner_run_name}/{self.curr_run_name}/df_results.pkl'))
        self.simWrapper.reset()
        self.agent.reset()
        if self.cfg['parallel']:
            try:
                self.wandb_log_data['spend'] = self.agent.get_spend()
                self.wandb_log_data['default_rate'] = len(self.df[self.df['success'] == 0]) / len(self.df)
                response = requests.post(f'http://localhost:{self.cfg["port"]}/log', json=self.wandb_log_data)
                if response.status_code != 200:
                    logging.error(f"Failed to send metrics: {response.text}")
            except Exception as e:
                tb = traceback.extract_tb(e.__traceback__)
                for frame in tb:
                    logging.error(f"Frame {frame.filename} line {frame.lineno}")
                logging.error(e)

        logging.info(f"Success: {self.wandb_log_data['goal_reached']}")
        logging.info('\n===================RUN COMPLETE===================\n')
        if self.cfg['log_freq'] == 1:
            create_gif(
                os.path.join(os.environ.get("LOG_DIR"), f'{self.outer_run_name}/{self.inner_run_name}/{self.curr_run_name}'), self.agent.cfg['sensor_cfg']['img_height'], self.agent.cfg['sensor_cfg']['img_width'], agent_cls=self.agent_cls
            )
            create_gif_nav(
                    os.path.join(os.environ.get("LOG_DIR"), f'{self.outer_run_name}/{self.inner_run_name}/{self.curr_run_name}'),
                    1800, 1800
            )
            create_gif_cvalue(
                os.path.join(os.environ.get("LOG_DIR"), f'{self.outer_run_name}/{self.inner_run_name}/{self.curr_run_name}'),
                1800, 1800
            )
