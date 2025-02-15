def _initialize_experiment(self):
    """
    Initializes the experiment by setting up the dataset split, scene configuration, and goals.
    """
    self.split = 'val' if 'val' in self.cfg['split'] else 'train'
    self.sim_cfg[
        'scene_config'] = "/file_system/vepfs/algorithm/dujun.nie/data/hm3d_v0.2/hm3d_annotated_basis.scene_dataset_config.json"

    self.all_episodes = []
    self.goals = {}
    for f in sorted(os.listdir(
            f'/file_system/vepfs/algorithm/dujun.nie/data/goat_bench/datasets/goat_bench/hm3d/v1/{self.cfg["split"]}/content')):
        with gzip.open(
                f'/file_system/vepfs/algorithm/dujun.nie/data/goat_bench/datasets/goat_bench/hm3d/v1/{self.cfg["split"]}/content/{f}',
                'rt') as gz:
            js = json.load(gz)
            hsh = f.split('.')[0]
            self.goals[hsh] = js['goals']
            self.all_episodes += js['episodes']
    self.num_episodes = len(self.all_episodes)
def _initialize_episode(self, episode_ndx: int):
    """
    Initializes the episode for the GOAT task.

    Args:
        episode_ndx (int): The index of the episode to initialize.
    """
    super()._initialize_episode(episode_ndx)

    episode = self.all_episodes[episode_ndx]
    f, glb = episode['scene_id'].split('/')[-2:]
    hsh = f[6:]
    goals = self.goals[hsh]
    self.sim_cfg['scene_id'] = f[2:5]
    self.sim_cfg['scene_path'] = f'/file_system/vepfs/algorithm/dujun.nie/data/hm3d_v0.2/{self.split}/{f}/{glb}'
    self.simWrapper = SimWrapper(self.sim_cfg)
    self.current_episode = []

    for goal in episode['tasks']:
        name = goal[0]
        mode = goal[1]
        subgoal = {'name': name, 'mode': mode, 'id': goal[2], 'view_points': []}
        for obj in goals[f'{f[6:]}.basis.glb_{name}']:
            if mode == 'object':
                subgoal['view_points'] += [a['agent_state']['position'] for a in obj['view_points']]
            else:
                if obj['object_id'] == goal[2]:
                    subgoal['view_points'] = [a['agent_state']['position'] for a in obj['view_points']]
                    if mode == 'description':
                        subgoal['lang_desc'] = obj['lang_desc']
                    if mode == 'image':
                        image_ndx = goal[3]
                        subgoal['image_position'] = obj['image_goals'][image_ndx]['position']
                        subgoal['image_rotation'] = obj['image_goals'][image_ndx]['rotation']

        self.current_episode.append(subgoal)

    logging.info(f'\nRUNNING EPISODE {episode_ndx}, SCENE: {self.simWrapper.scene_id}')
    for i, subgoal in enumerate(self.current_episode):
        logging.info(f'Goal {i}: {subgoal["name"]}, {subgoal["mode"]}')

    self.init_pos = np.array(episode['start_position'])
    self.simWrapper.set_state(pos=self.init_pos, quat=episode['start_rotation'])
    self.curr_goal_ndx = 0
    self.curr_run_name = f"{episode_ndx}_{self.simWrapper.scene_id}"
    self.last_goal_reset = -1
    self.path_calculator.requested_ends = np.array(self.current_episode[self.curr_goal_ndx]['view_points'],
                                                   dtype=np.float32)
    self.path_calculator.requested_start = self.init_pos
    self.curr_shortest_path = self.simWrapper.get_path(self.path_calculator)

    obs = self.simWrapper.step(PolarAction.null)
    return obs


def _step_env(self, obs: dict):
    """
    Takes a step in the environment for the GOAT task.

    Args:
        obs (dict): The current observation.

    Returns:
        list: The next action to be taken by the agent.
    """
    super()._step_env(obs)

    goal = self.current_episode[self.curr_goal_ndx]
    obs['goal'] = goal
    if goal['mode'] == 'image':
        position = goal['image_position']
        rotation = goal['image_rotation']
        goal_im = self.simWrapper.get_goal_image(position, rotation)
        put_text_on_image(goal_im, f"GOAL IMAGE: {goal['name']}", bg_color=(255, 255, 255), location='top_center')
        obs['goal']['goal_image'] = goal_im

    agent_state = obs['agent_state']
    agent_action, metadata = self.agent.step(obs)
    step_metadata = metadata['step_metadata']
    logging_data = metadata['logging_data']
    images = metadata['images']

    metrics = self._calculate_metrics(agent_state, agent_action, self.curr_shortest_path,
                                      self.last_goal_reset + 1 + self.cfg['max_steps_per_subgoal'])
    step_metadata.update(metrics)

    if metrics['done']:
        self.wandb_log_data['task_data'].setdefault('goal_data', []).append({
            'goal_mode': goal['mode'],
            'goal_reached': metrics['goal_reached'],
            'spl': metrics['spl'],
        })

        if 'spl' in self.wandb_log_data:
            del self.wandb_log_data['spl']
        if 'goal_reached' in self.wandb_log_data:
            del self.wandb_log_data['goal_reached']

        if self.curr_goal_ndx + 1 == len(self.current_episode):
            agent_action = None
        else:
            self.agent.reset_goal()
            self.agent_distance_traveled = 0
            agent_action = PolarAction.null
            self.curr_goal_ndx += 1
            self.last_goal_reset = self.step
            goal = self.current_episode[self.curr_goal_ndx]
            self.path_calculator.requested_ends = np.array(goal['view_points'], dtype=np.float32)
            self.path_calculator.requested_start = obs['agent_state'].position
            self.curr_shortest_path = self.simWrapper.get_path(self.path_calculator)

            logging.info(f'New goal {goal["mode"]}: {goal["name"]}, GEODESIC: {self.curr_shortest_path}')

    self._log(images, step_metadata, logging_data)

    return agent_action