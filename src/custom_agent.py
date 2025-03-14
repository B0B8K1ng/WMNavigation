import logging
import math
import random
import habitat_sim
import numpy as np
import cv2
import ast
import concurrent.futures

from simWrapper import PolarAction
from utils import *
from api import *
from WMNav_agent import *

class a(VLMNavAgent):
    def _initialize_vlms(self, cfg: dict):
        vlm_cls = globals()[cfg['model_cls']]
        system_instruction = (
            "You are an embodied robotic assistant, with an RGB image sensor. You observe the image and instructions "
            "given to you and output a textual response, which is converted into actions that physically move you "
            "within the environment. You cannot move through closed doors. "
        )
        self.ActionVLM: VLM = vlm_cls(**cfg['model_kwargs'], system_instruction=system_instruction)
        self.PlanVLM: VLM = vlm_cls(**cfg['model_kwargs'])
        self.stoppingVLM: VLM = vlm_cls(**cfg['model_kwargs'])

    def _update_panoramic_voxel(self, r: float, theta: float, agent_state: habitat_sim.AgentState, clip_dist: float,
                                clip_frac: float):
        """Update the voxel map to mark actions as explored"""
        agent_coords = self._global_to_grid(agent_state.position)

        # Mark circle explored regions
        clipped = min(clip_frac * r, clip_dist)
        local_coords = np.array([clipped * np.sin(theta), 0, -clipped * np.cos(theta)])
        global_coords = local_to_global(agent_state.position, agent_state.rotation, local_coords)
        point = self._global_to_grid(global_coords)

        radius = int(np.linalg.norm(np.array(agent_coords) - np.array(point)))
        cv2.circle(self.explored_map, agent_coords, radius, self.explored_color, -1)  # -1表示实心圆

    def _navigability(self, obs: dict):
        """Generates the set of navigability actions and updates the voxel map accordingly."""
        agent_state: habitat_sim.AgentState = obs['agent_state']
        sensor_state = agent_state.sensor_states['color_sensor']
        rgb_image = obs['color_sensor']
        depth_image = obs[f'depth_sensor']
        if self.cfg['navigability_mode'] == 'depth_estimate':
            depth_image = self.depth_estimator.call(rgb_image)
        if self.cfg['navigability_mode'] == 'segmentation':
            depth_image = None

        navigability_mask = self._get_navigability_mask(
            rgb_image, depth_image, agent_state, sensor_state
        )

        sensor_range = np.deg2rad(self.fov / 2) * 1.5

        all_thetas = np.linspace(-sensor_range, sensor_range, self.cfg['num_theta'])
        start = agent_frame_to_image_coords(
            [0, 0, 0], agent_state, sensor_state,
            resolution=self.resolution, focal_length=self.focal_length
        )  # agent的原点在图像坐标系的位置（像素）

        a_initial = []
        for theta_i in all_thetas:
            r_i, theta_i = self._get_radial_distance(start, theta_i, navigability_mask, agent_state, sensor_state,
                                                     depth_image)
            if r_i is not None:
                self._update_voxel(
                    r_i, theta_i, agent_state,
                    clip_dist=self.cfg['max_action_dist'], clip_frac=self.e_i_scaling
                )
                a_initial.append((r_i, theta_i))

        # draw explored circle
        if self.cfg['panoramic_padding'] == True:
            r_max, theta_max = max(a_initial, key=lambda x: x[0])
            self._update_panoramic_voxel(r_max, theta_max, agent_state,
                                         clip_dist=self.cfg['max_action_dist'], clip_frac=self.e_i_scaling)
        return a_initial

    def get_spend(self):
        # 可以修改原始逻辑或添加新的逻辑
        original_spend = super().get_spend()  # 调用父类的 get_spend 方法
        # 假设我们要将父类的返回值加上一个额外的花费
        return original_spend + self.PlanVLM.get_spend()

    def _choose_action(self, obs: dict):
        agent_state = obs['agent_state']
        goal = obs['goal']

        a_final, images, step_metadata, stopping_response = self._run_threads(obs, [obs['color_sensor']], goal)
        step_metadata['object'] = goal

        # If the model calls stop two times in a row, terminate the episode
        if len(self.stopping_calls) >= 2 and self.stopping_calls[-2] == self.step_ndx - 1:
            step_metadata['action_number'] = -1
            agent_action = PolarAction.stop
            logging_data = {}
        else:
            step_metadata, logging_data, _ = self._prompting(goal, a_final, images, step_metadata)
            agent_action = self._action_number_to_polar(step_metadata['action_number'], list(a_final))

        logging_data['STOPPING RESPONSE'] = stopping_response
        metadata = {
            'step_metadata': step_metadata,
            'logging_data': logging_data,
            'a_final': a_final,
            'images': images,
            'step': self.step_ndx
        }
        return agent_action, metadata

    def _planning_module(self, planning_image: list[np.array], goal):
        """Determines if the agent should stop."""
        planning_prompt = self._construct_prompt(goal, 'planning')
        planning_response = self.PlanVLM.call([planning_image], planning_prompt)
        planning_response = planning_response.replace('false', 'False').replace('true', 'True')
        dct = self._eval_response(planning_response)

        return dct

    @staticmethod
    def _concat_panoramic(images, angles):
        try:
            height, width = images[0].shape[0], images[0].shape[1]
        except:
            height, width = 480, 640
        background_image = np.zeros((2 * height + 3 * 10, 3 * width + 4 * 10, 3), np.uint8)
        copy_images = np.array(images, dtype=np.uint8)
        for i in range(len(copy_images)):
            if i % 2 == 0:
                continue
            copy_images[i] = cv2.putText(copy_images[i], "Angle %d" % angles[i], (100, 100), cv2.FONT_HERSHEY_SIMPLEX,
                                         2, (255, 0, 0), 6, cv2.LINE_AA)
            row = i // 6
            col = (i // 2) % 3
            background_image[10 * (row + 1) + row * height:10 * (row + 1) + row * height + height:,
            10 * (col + 1) + col * width:10 * (col + 1) + col * width + width, :] = copy_images[i]
        return background_image

    def make_plan(self, pano_images, goal):
        angles = (np.arange(len(pano_images))) * 30
        inference_image = self._concat_panoramic(pano_images, angles)

        response = self._planning_module(inference_image, goal)

        try:
            direction, goal_flag, reason = (int(response['Angle'] // 30)) % 12, response['Flag'], response['Reason']
        except:
            print("planning failed!")
            print('response:', response)
            direction, goal_flag, reason = np.random.randint(0, 12), False, None
        direction_image = cv2.cvtColor(pano_images[direction], cv2.COLOR_BGR2RGB)

        return inference_image, direction_image, direction, reason

    def _construct_prompt(self, goal: str, prompt_type: str, num_actions: int = 0):
        if prompt_type == 'stopping':
            stopping_prompt = (
                f"The agent has has been tasked with navigating to a {goal.upper()}. The agent has sent you an image taken from its current location. "
                f'Your job is to determine whether the agent is VERY CLOSE to a {goal}. Note a chair is NOT sofa which is NOT a bed. '
                f"First, tell me what you see in the image, and tell me if there is a {goal}. Second, return 1 if the agent is VERY CLOSE to the {goal} - make sure the object you see is ACTUALLY a {goal}, Return 0 if if there is no {goal}, or if it is far away, or if you are not sure. Format your answer in the json {{'done': <1 or 0>}}")
            return stopping_prompt
        if prompt_type == 'planning':
            planning_prompt = (
                f"The agent has has been tasked with navigating to a {goal.upper()}. The agent has sent you the panoramic image describing your surrounding environment, each image contains a label indicating the relative rotation angle with red fonts. "
                f'Your job is to select the best direction. '
                f'To help you select the best direction, I can give you some human suggestions:. '
                f'(1) For each direction, first confirm whether there are floor in the view, do not choose the directions without seen navigable areas. Note EACH DIRECTION MUST BE DESCRIBED. '
                f'(2) Try to avoid going backwards (selecting 150,210), unless all the forward directions do not meet the requirements of (1). Note you CANNOT GO THROUGH CLOSED DOORS, and you DO NOT NEED TO GO UP OR DOWN STAIRS'
                f'(3) For each direction, analyze the room type and think about whether the {goal} is likely to occur in that room. Note a chair is NOT sofa which is NOT a bed. '
                f"Format your answer in the json {{'Reason':<Analyze each view image, and tell me your reason. Don't output the same sentence over and over again>, 'Angle':<Your Select Angle>, 'Flag': <Whether the target is in your selected view, True or False>}}. ")
            return planning_prompt
        if prompt_type == 'action':
            action_prompt = (
                f"TASK: NAVIGATE TO THE NEAREST {goal.upper()}, and get as close to it as possible. Use your prior knowledge about where items are typically located within a home. "
                f"There are {num_actions - 1} red arrows superimposed onto your observation, which represent potential actions. "
                f"These are labeled with a number in a white circle, which represent the location you would move to if you took that action. {'NOTE: choose action 0 if you want to TURN AROUND or DONT SEE ANY GOOD ACTIONS. ' if self.step_ndx - self.turned >= self.cfg['turn_around_cooldown'] else ''}"
                f"First, tell me what you see in your sensor observation, and if you have any leads on finding the {goal.upper()}. Second, tell me which general direction you should go in. Don't output the same sentence over and over again"
                f"Lastly, explain which action acheives that best, and return it as {{'action': <action_key>}}. Note you CANNOT GO THROUGH CLOSED DOORS, and you DO NOT NEED TO GO UP OR DOWN STAIRS"
            )
            return action_prompt

        raise ValueError('Prompt type must be stopping, planning, or action')

class b(VLMNavAgent):

    def _initialize_vlms(self, cfg: dict):
        # 先调用父类的 _initialize_vlms 方法，初始化 ActionVLM 和 stoppingVLM
        super()._initialize_vlms(cfg)

        # 现在初始化新的 PlanVLM
        vlm_cls = globals()[cfg['model_cls']]

        self.PlanVLM: VLM = vlm_cls(**cfg['model_kwargs'])
        self.descripVLM: VLM = vlm_cls(**cfg['model_kwargs'])

    def _update_panoramic_voxel(self, r: float, theta: float, agent_state: habitat_sim.AgentState, clip_dist: float,
                                clip_frac: float):
        """Update the voxel map to mark actions as explored"""
        agent_coords = self._global_to_grid(agent_state.position)

        # Mark circle explored regions
        clipped = min(clip_frac * r, clip_dist)
        local_coords = np.array([clipped * np.sin(theta), 0, -clipped * np.cos(theta)])
        global_coords = local_to_global(agent_state.position, agent_state.rotation, local_coords)
        point = self._global_to_grid(global_coords)

        radius = int(np.linalg.norm(np.array(agent_coords) - np.array(point)))
        cv2.circle(self.explored_map, agent_coords, radius, self.explored_color, -1)  # -1表示实心圆

    def _navigability(self, obs: dict):
        """Generates the set of navigability actions and updates the voxel map accordingly."""
        agent_state: habitat_sim.AgentState = obs['agent_state']
        sensor_state = agent_state.sensor_states['color_sensor']
        rgb_image = obs['color_sensor']
        depth_image = obs[f'depth_sensor']
        if self.cfg['navigability_mode'] == 'depth_estimate':
            depth_image = self.depth_estimator.call(rgb_image)
        if self.cfg['navigability_mode'] == 'segmentation':
            depth_image = None

        navigability_mask = self._get_navigability_mask(
            rgb_image, depth_image, agent_state, sensor_state
        )

        sensor_range = np.deg2rad(self.fov / 2) * 1.5

        all_thetas = np.linspace(-sensor_range, sensor_range, self.cfg['num_theta'])
        start = agent_frame_to_image_coords(
            [0, 0, 0], agent_state, sensor_state,
            resolution=self.resolution, focal_length=self.focal_length
        )  # agent的原点在图像坐标系的位置（像素）

        a_initial = []
        for theta_i in all_thetas:
            r_i, theta_i = self._get_radial_distance(start, theta_i, navigability_mask, agent_state, sensor_state,
                                                     depth_image)
            if r_i is not None:
                self._update_voxel(
                    r_i, theta_i, agent_state,
                    clip_dist=self.cfg['max_action_dist'], clip_frac=self.e_i_scaling
                )
                a_initial.append((r_i, theta_i))

        # draw explored circle
        if self.cfg['panoramic_padding'] == True:
            r_max, theta_max = max(a_initial, key=lambda x: x[0])
            self._update_panoramic_voxel(r_max, theta_max, agent_state,
                                         clip_dist=self.cfg['max_action_dist'], clip_frac=self.e_i_scaling)
        return a_initial

    def get_spend(self):
        # 可以修改原始逻辑或添加新的逻辑
        original_spend = super().get_spend()  # 调用父类的 get_spend 方法
        # 假设我们要将父类的返回值加上一个额外的花费
        return original_spend + self.PlanVLM.get_spend() + self.descripVLM.get_spend()

    def _prompting(self, goal, a_final: list, images: dict, step_metadata: dict, subtask: str):
        """
        Prompting component of BASEV2. Constructs the textual prompt and calls the action model.
        Parses the response for the chosen action number.
        """
        prompt_type = 'action'
        action_prompt = self._construct_prompt(goal, prompt_type, subtask, num_actions=len(a_final))

        prompt_images = [images['color_sensor']]
        if 'goal_image' in images:
            prompt_images.append(images['goal_image'])

        response = self.ActionVLM.call_chat(self.cfg['context_history'], prompt_images, action_prompt)

        logging_data = {}
        try:
            response_dict = self._eval_response(response)
            step_metadata['action_number'] = int(response_dict['action'])
        except (IndexError, KeyError, TypeError, ValueError) as e:
            logging.error(f'Error parsing response {e}')
            step_metadata['success'] = 0
        finally:
            logging_data['ACTION_NUMBER'] = step_metadata.get('action_number')
            logging_data['ACTION_PROMPT'] = action_prompt
            logging_data['ACTION_RESPONSE'] = response

        return step_metadata, logging_data, response

    def _choose_action(self, obs: dict):
        agent_state = obs['agent_state']
        goal = obs['goal']

        a_final, images, step_metadata, stopping_response = self._run_threads(obs, [obs['color_sensor']], goal)
        step_metadata['object'] = goal

        # If the model calls stop two times in a row, terminate the episode
        if len(self.stopping_calls) >= 2 and self.stopping_calls[-2] == self.step_ndx - 1:
            step_metadata['action_number'] = -1
            agent_action = PolarAction.stop
            logging_data = {}
        else:
            step_metadata, logging_data, _ = self._prompting(goal, a_final, images, step_metadata, obs['subtask'])
            agent_action = self._action_number_to_polar(step_metadata['action_number'], list(a_final))

        logging_data['STOPPING RESPONSE'] = stopping_response
        metadata = {
            'step_metadata': step_metadata,
            'logging_data': logging_data,
            'a_final': a_final,
            'images': images,
            'step': self.step_ndx
        }
        return agent_action, metadata

    def _eval_response(self, response: str):
        """Converts the VLM response string into a dictionary, if possible"""
        try:
            eval_resp = ast.literal_eval(response[response.index('{') + 1:response.rindex('}')])
            if isinstance(eval_resp, dict):
                return eval_resp
            else:
                raise ValueError
        except (ValueError, SyntaxError):
            logging.error(f'Error parsing response {response}')
            return {}

    def _planning_module(self, planning_image: list[np.array], previous_subtask, desciption: str, goal):
        """Determines if the agent should stop."""
        planning_prompt = self._construct_prompt(goal, 'planning', previous_subtask, desciption)
        planning_response = self.PlanVLM.call([planning_image], planning_prompt)
        planning_response = planning_response.replace('false', 'False').replace('true', 'True')
        dct = self._eval_response(planning_response)

        return dct

    def _descrip_module(self, descrip_image, goal):
        """Determines if the agent should stop."""
        descrip_prompt = self._construct_prompt(goal, 'descrip')
        descrip_response = self.descripVLM.call([descrip_image], descrip_prompt)
        dct = self._eval_response(descrip_response)

        return dct

    @staticmethod
    def _concat_panoramic(images, angles):
        try:
            height, width = images[0].shape[0], images[0].shape[1]
        except:
            height, width = 480, 640
        background_image = np.zeros((2 * height + 3 * 10, 3 * width + 4 * 10, 3), np.uint8)
        copy_images = np.array(images, dtype=np.uint8)
        for i in range(len(copy_images)):
            if i % 2 == 0:
                continue
            copy_images[i] = cv2.putText(copy_images[i], "Angle %d" % angles[i], (100, 100), cv2.FONT_HERSHEY_SIMPLEX,
                                         2, (255, 0, 0), 6, cv2.LINE_AA)
            row = i // 6
            col = (i // 2) % 3
            background_image[10 * (row + 1) + row * height:10 * (row + 1) + row * height + height:,
            10 * (col + 1) + col * width:10 * (col + 1) + col * width + width, :] = copy_images[i]
        return background_image

    def describe(self, response):
        import json
        description_str = json.dumps(response, ensure_ascii=False)

        return description_str

    def make_description(self, pano_images, goal):
        angles = (np.arange(len(pano_images))) * 30
        inference_image = cv2.cvtColor(self._concat_panoramic(pano_images, angles), cv2.COLOR_BGR2RGB)

        response = self._descrip_module(inference_image, goal)

        description = self.describe(response)

        return inference_image, description

    def make_plan(self, pano_images, previous_subtask, description, goal):
        response = self._planning_module(pano_images, previous_subtask, description, goal)

        try:
            direction, goal_flag, subtask = (int(response['Angle'] // 30)) % 12, response['Flag'], response['Subtask']
        except:
            print("planning failed!")
            print('response:', response)
            direction, goal_flag, subtask = np.random.randint(0, 12), False, '{}'

        return direction, goal_flag, subtask

    def _construct_prompt(self, goal: str, prompt_type: str, subtask: str = '{}', descrip: str = '{}',
                          num_actions: int = 0):
        if prompt_type == 'stopping':
            stopping_prompt = (
                f"The agent has been tasked with navigating to a {goal.upper()}. The agent has sent you an image taken from its current location. "
                f'Your job is to determine whether the agent is VERY CLOSE to a {goal}. Note a chair is NOT sofa(couch) which is NOT a bed. '
                f"First, tell me what you see in the image, and tell me if there is a {goal}. Second, return 1 if the agent is VERY CLOSE to the {goal} - make sure the object you see is ACTUALLY a {goal}, Return 0 if if there is no {goal}, or if it is far away, or if you are not sure. "
                "Format your answer in the json {{'done': <1 or 0>}}")
            return stopping_prompt
        if prompt_type == 'descrip':
            descrip_prompt = (
                f"The agent has been tasked with navigating to a {goal.upper()}. The agent has sent you the panoramic image describing your surrounding environment, each image contains a label indicating the relative rotation angle with red fonts. "
                f'Your job is to understand the layout of your surrounding. '
                f'To help you describe the layout of your surrounding,  please follow my step-by-step instructions: '
                f'(1) For each direction, describe what kinds of objects it has and what type of room it belongs to. Note doors are also under consideration for objects(Determine whether a door is open or closed, use open door or closed door). Note each class of objects is listed only once. Note a chair is NOT sofa(couch) which is NOT a bed. '
                f'(2) Locate yourself in which room you are currently in.  '
                "Format your answer in the json {{'descriptions':[{'Type': <Room type>, 'Angle': <direction angle for the room type>, 'Objects': <Objects that the corresponding angle of the image has(Don't output the same class of objects over and over again)>}], 'self_localization':<The type of room you are in>}}. "
                "Answer Example: {{'descriptions':[{'Type': Dining Room, 'Angle': 30, 'Objects': [table, ...]}, {'Type': Living Room, 'Angle': 90, 'Objects': [sofa, ...]}, {'Type': Kitchen, 'Angle': 150, 'Objects': [sink, refrigerator, ...]}, ...], 'self_localization': Living Room}}")
            return descrip_prompt
        if prompt_type == 'planning':
            if descrip != '{}' and subtask != '{}':
                planning_prompt = (
                            f"The agent has been tasked with navigating to a {goal.upper()}. The agent has sent you the following elements:"
                            f"(1)<The panoramic image>: The image describes your surrounding environment, each image contains a label indicating the relative rotation angle with red fonts. "
                            f"(2){descrip}. This dictionary describes the room type and the objects within it corresponding to each orientation angle, as well as the room type where the agent is currently located. "
                            f'Your job is to select the best direction and describe next place to go. '
                            f'To help you select the best direction and plan your next step, I can give you some human suggestions:. '
                            f'(1) For each direction, first confirm whether there are floor in the view, do not choose the directions without seen navigable areas. '
                            f'(2) Try to avoid going backwards (selecting 150,210), unless all the forward directions do not meet the requirements of (1). Note you CANNOT GO THROUGH CLOSED DOORS, and you DO NOT NEED TO GO UP OR DOWN STAIRS. '
                            f'(3) If the {goal} appears in the panoramic image, select the direction in which the {goal} appears. '
                            f'(4) If the {goal} is not found and the previous subtask {subtask} has not completed, choose one direction from the direction of angle 30 and angle 330 to continue to complete the last subtask {subtask} that has not been completed.'
                            f'(5) If the {goal} is not found and the previous subtask {subtask} has already been completed. Identify a new subtask by describing where you are going next to be more likely to find clues to the the {goal} and analyze the room type and think about whether the {goal} is likely to occur in that direction. Note you need to pay special attention to open doors and hallways, as they can lead to other unseen rooms. '
                            "Format your answer in the json {{'Subtask': <Where you are going next>, 'Angle': <Your Select Angle>, 'Flag': <Whether the target is in your selected view, True or False>}}. "
                            "Answer Example: {{'Subtask': 'Go to the hallway', 'Angle': 30, 'Flag': False}} or {{'Subtask': " + f"'Go to the {goal}'" + ", 'Angle': 150, 'Flag': True}} or {{'Subtask': 'Go to the open door', 'Angle': 150, 'Flag': True}}")
            else:
                planning_prompt = (
                            f"The agent has been tasked with navigating to a {goal.upper()}. The agent has sent you the panoramic image describing your surrounding environment, each image contains a label indicating the relative rotation angle with red fonts. "
                            f'Your job is to select the best direction and describe next place to go. '
                            f'To help you select the best direction and plan your next step, I can give you some human suggestions:. '
                            f'(1) For each direction, first confirm whether there are floor in the view, do not choose the directions without seen navigable areas. '
                            f'(2) Try to avoid going backwards (selecting 150,210), unless all the forward directions do not meet the requirements of (1). Note you CANNOT GO THROUGH CLOSED DOORS, and you DO NOT NEED TO GO UP OR DOWN STAIRS. '
                            f'(3) If the {goal} appears in the panoramic image, select the direction in which the {goal} appears. '
                            f'(4) If the {goal} is not found, describe where you are going next to be more likely to find clues to the the {goal} and analyze the room type and think about whether the {goal} is likely to occur in that direction. Note you need to pay special attention to open doors and hallways, as they can lead to other unseen rooms. '
                            "Format your answer in the json {{'Subtask': <Where you are going next>, 'Angle': <Your Select Angle>, 'Flag': <Whether the target is in your selected view, True or False>}}. "
                            "Answer Example: {{'Subtask': 'Go to the hallway', 'Angle': 30, 'Flag': False}} or {{'Subtask': " + f"'Go to the {goal}'" + ", 'Angle': 150, 'Flag': True}} or {{'Subtask': 'Go to the open door', 'Angle': 150, 'Flag': True}}")
            return planning_prompt
        if prompt_type == 'action':
            if subtask != '{}':
                action_prompt = (
                    f"TASK: {subtask}. Your final task is to NAVIGATE TO THE NEAREST {goal.upper()}, and get as close to it as possible. "
                    f"There are {num_actions - 1} red arrows superimposed onto your observation, which represent potential actions. "
                    f"These are labeled with a number in a white circle, which represent the location you would move to if you took that action. {'NOTE: choose action 0 if you want to TURN AROUND or DONT SEE ANY GOOD ACTIONS. ' if self.step_ndx - self.turned >= self.cfg['turn_around_cooldown'] else ''}"
                    f"In order to complete the subtask {subtask} and eventually the final task NAVIGATING TO THE NEAREST {goal.upper()}. Explain which action acheives that best. "
                    "Return your answer as {{'action': <action_key>}}. Note you CANNOT GO THROUGH CLOSED DOORS, and you DO NOT NEED TO GO UP OR DOWN STAIRS"
                )
            else:
                action_prompt = (
                    f"TASK: NAVIGATE TO THE NEAREST {goal.upper()}, and get as close to it as possible. Use your prior knowledge about where items are typically located within a home. "
                    f"There are {num_actions - 1} red arrows superimposed onto your observation, which represent potential actions. "
                    f"These are labeled with a number in a white circle, which represent the location you would move to if you took that action. {'NOTE: choose action 0 if you want to TURN AROUND or DONT SEE ANY GOOD ACTIONS. ' if self.step_ndx - self.turned >= self.cfg['turn_around_cooldown'] else ''}"
                    f"First, tell me what you see in your sensor observation, and if you have any leads on finding the {goal.upper()}. Second, tell me which general direction you should go in. "
                    "Lastly, explain which action acheives that best, and return it as {{'action': <action_key>}}. Note you CANNOT GO THROUGH CLOSED DOORS, and you DO NOT NEED TO GO UP OR DOWN STAIRS"
                )
            return action_prompt

        raise ValueError('Prompt type must be stopping, descrip, planning, or action')

class c(VLMNavAgent):

    def _initialize_vlms(self, cfg: dict):
        # 先调用父类的 _initialize_vlms 方法，初始化 ActionVLM 和 stoppingVLM
        super()._initialize_vlms(cfg)

        # 现在初始化新的 PlanVLM
        vlm_cls = globals()[cfg['model_cls']]

        self.PlanVLM: VLM = vlm_cls(**cfg['model_kwargs'])
        self.descripVLM: VLM = vlm_cls(**cfg['model_kwargs'])


    def _update_panoramic_voxel(self, r: float, theta: float, agent_state: habitat_sim.AgentState, clip_dist: float, clip_frac: float):
        """Update the voxel map to mark actions as explored"""
        agent_coords = self._global_to_grid(agent_state.position)

        # Mark circle explored regions
        clipped = min(clip_frac * r, clip_dist)
        local_coords = np.array([clipped * np.sin(theta), 0, -clipped * np.cos(theta)])
        global_coords = local_to_global(agent_state.position, agent_state.rotation, local_coords)
        point = self._global_to_grid(global_coords)

        radius = int(np.linalg.norm(np.array(agent_coords) - np.array(point)))
        cv2.circle(self.explored_map, agent_coords, radius, self.explored_color, -1)  # -1表示实心圆


    def _navigability(self, obs: dict):
        """Generates the set of navigability actions and updates the voxel map accordingly."""
        agent_state: habitat_sim.AgentState = obs['agent_state']
        sensor_state = agent_state.sensor_states['color_sensor']
        rgb_image = obs['color_sensor']
        depth_image = obs[f'depth_sensor']
        if self.cfg['navigability_mode'] == 'depth_estimate':
            depth_image = self.depth_estimator.call(rgb_image)
        if self.cfg['navigability_mode'] == 'segmentation':
            depth_image = None

        navigability_mask = self._get_navigability_mask(
            rgb_image, depth_image, agent_state, sensor_state
        )

        sensor_range =  np.deg2rad(self.fov / 2) * 1.5

        all_thetas = np.linspace(-sensor_range, sensor_range, self.cfg['num_theta'])
        start = agent_frame_to_image_coords(
            [0, 0, 0], agent_state, sensor_state,
            resolution=self.resolution, focal_length=self.focal_length
        )  # agent的原点在图像坐标系的位置（像素）

        a_initial = []
        for theta_i in all_thetas:
            r_i, theta_i = self._get_radial_distance(start, theta_i, navigability_mask, agent_state, sensor_state, depth_image)
            if r_i is not None:
                self._update_voxel(
                    r_i, theta_i, agent_state,
                    clip_dist=self.cfg['max_action_dist'], clip_frac=self.e_i_scaling
                )
                a_initial.append((r_i, theta_i))

        # draw explored circle
        if self.cfg['panoramic_padding'] == True:
            r_max, theta_max = max(a_initial, key=lambda x: x[0])
            self._update_panoramic_voxel(r_max, theta_max, agent_state,
                    clip_dist=self.cfg['max_action_dist'], clip_frac=self.e_i_scaling)
        return a_initial

    def get_spend(self):
        # 可以修改原始逻辑或添加新的逻辑
        original_spend = super().get_spend()  # 调用父类的 get_spend 方法
        # 假设我们要将父类的返回值加上一个额外的花费
        return original_spend + self.PlanVLM.get_spend() + self.descripVLM.get_spend()

    def _prompting(self, goal, a_final: list, images: dict, step_metadata: dict, subtask: str):
        """
        Prompting component of BASEV2. Constructs the textual prompt and calls the action model.
        Parses the response for the chosen action number.
        """
        prompt_type = 'action'
        action_prompt = self._construct_prompt(goal, prompt_type, subtask, num_actions=len(a_final))

        prompt_images = [images['color_sensor']]
        if 'goal_image' in images:
            prompt_images.append(images['goal_image'])

        response = self.ActionVLM.call_chat(self.cfg['context_history'], prompt_images, action_prompt)

        logging_data = {}
        try:
            response_dict = self._eval_response(response)
            step_metadata['action_number'] = int(response_dict['action'])
        except (IndexError, KeyError, TypeError, ValueError) as e:
            logging.error(f'Error parsing response {e}')
            step_metadata['success'] = 0
        finally:
            logging_data['ACTION_NUMBER'] = step_metadata.get('action_number')
            logging_data['ACTION_PROMPT'] = action_prompt
            logging_data['ACTION_RESPONSE'] = response

        return step_metadata, logging_data, response

    def _choose_action(self, obs: dict):
        agent_state = obs['agent_state']
        goal = obs['goal']

        a_final, images, step_metadata, stopping_response = self._run_threads(obs, [obs['color_sensor']], goal)
        step_metadata['object'] = goal

        # If the model calls stop two times in a row, terminate the episode
        if len(self.stopping_calls) >= 2 and self.stopping_calls[-2] == self.step_ndx - 1:
            step_metadata['action_number'] = -1
            agent_action = PolarAction.stop
            logging_data = {}
        else:
            step_metadata, logging_data, _ = self._prompting(goal, a_final, images, step_metadata, obs['subtask'])
            agent_action = self._action_number_to_polar(step_metadata['action_number'], list(a_final))

        logging_data['STOPPING RESPONSE'] = stopping_response
        metadata = {
            'step_metadata': step_metadata,
            'logging_data': logging_data,
            'a_final': a_final,
            'images': images,
            'step': self.step_ndx
        }
        return agent_action, metadata

    def _eval_response(self, response: str):
        """Converts the VLM response string into a dictionary, if possible"""
        try:
            eval_resp = ast.literal_eval(response[response.index('{')+1:response.rindex('}')])
            if isinstance(eval_resp, dict):
                return eval_resp
            else:
                raise ValueError
        except (ValueError, SyntaxError):
            logging.error(f'Error parsing response {response}')
            return {}


    def _planning_module(self, planning_image: list[np.array], previous_subtask, desciption: str, goal):
        """Determines if the agent should stop."""
        planning_prompt = self._construct_prompt(goal, 'planning', previous_subtask, desciption)
        planning_response = self.PlanVLM.call([planning_image], planning_prompt)
        planning_response = planning_response.replace('false', 'False').replace('true', 'True')
        dct = self._eval_response(planning_response)

        return dct

    def _descrip_module(self, descrip_image, goal):
        """Determines if the agent should stop."""
        descrip_prompt = self._construct_prompt(goal, 'descrip')
        descrip_response = self.descripVLM.call([descrip_image], descrip_prompt)
        dct = self._eval_response(descrip_response)

        return dct

    @staticmethod
    def _concat_panoramic(images, angles):
        try:
            height, width = images[0].shape[0], images[0].shape[1]
        except:
            height, width = 480, 640
        background_image = np.zeros((2 * height + 3 * 10, 3 * width + 4 * 10, 3), np.uint8)
        copy_images = np.array(images, dtype=np.uint8)
        for i in range(len(copy_images)):
            if i % 2 == 0:
                continue
            copy_images[i] = cv2.putText(copy_images[i], "Angle %d" % angles[i], (100, 100), cv2.FONT_HERSHEY_SIMPLEX,
                                         2, (255, 0, 0), 6, cv2.LINE_AA)
            row = i // 6
            col = (i // 2) % 3
            background_image[10 * (row + 1) + row * height:10 * (row + 1) + row * height + height:,
            10 * (col + 1) + col * width:10 * (col + 1) + col * width + width, :] = copy_images[i]
        return background_image

    def describe(self, response):
        import json
        description_str = json.dumps(response, ensure_ascii=False)

        return description_str

    def make_description(self, pano_images, goal):
        angles = (np.arange(len(pano_images))) * 30
        inference_image = cv2.cvtColor(self._concat_panoramic(pano_images, angles), cv2.COLOR_BGR2RGB)

        response = self._descrip_module(inference_image, goal)

        description = self.describe(response)

        return inference_image, description

    def make_plan(self, pano_images, previous_subtask, description, goal):
        response = self._planning_module(pano_images, previous_subtask, description, goal)

        try:
            direction, goal_flag, subtask = (int(response['Angle'] // 30)) % 12, response['Flag'], response['Subtask']
        except:
            print("planning failed!")
            print('response:', response)
            direction, goal_flag, subtask = np.random.randint(0, 12), False, '{}'

        return direction, goal_flag, subtask

    def _construct_prompt(self, goal: str, prompt_type:str, subtask: str='{}', descrip: str='{}', num_actions: int=0):
        if prompt_type == 'stopping':
            stopping_prompt = (f"The agent has been tasked with navigating to a {goal.upper()}. The agent has sent you an image taken from its current location. "
            f'Your job is to determine whether the agent is VERY CLOSE to a {goal}. Note a chair is NOT sofa(couch) which is NOT a bed. '
            f"First, tell me what you see in the image, and tell me if there is a {goal}. Second, return 1 if the agent is VERY CLOSE to the {goal} - make sure the object you see is ACTUALLY a {goal}, Return 0 if if there is no {goal}, or if it is far away, or if you are not sure. "
            "Format your answer in the json {{'done': <1 or 0>}}")
            return stopping_prompt
        if prompt_type == 'descrip':
            descrip_prompt = (f"The agent has been tasked with navigating to a {goal.upper()}. The agent has sent you the panoramic image describing your surrounding environment, each image contains a label indicating the relative rotation angle with blue fonts. "
            f'Your job is to understand the layout of your surrounding. '
            f'To help you describe the layout of your surrounding,  please follow my step-by-step instructions: '
            f'(1) For each direction, describe what kinds of objects it has and what type of room it belongs to. Note doors are also under consideration for objects(Determine whether a door is open or closed, use open door or closed door). Note each class of objects is listed only once. Note a chair is NOT sofa(couch) which is NOT a bed. '
            f'(2) Locate yourself in which room you are currently in.  '
            "Format your answer in the json {{'descriptions':[{'Type': <Room type>, 'Angle': <direction angle for the room type>, 'Objects': <Objects that the corresponding angle of the image has(Don't output the same class of objects over and over again)>}], 'self_localization':<The type of room you are in>}}. "
            "Answer Example: {{'descriptions':[{'Type': Dining Room, 'Angle': 30, 'Objects': [table, ...]}, {'Type': Living Room, 'Angle': 90, 'Objects': [sofa, ...]}, {'Type': Kitchen, 'Angle': 150, 'Objects': [sink, refrigerator, ...]}, ...], 'self_localization': Living Room}}")
            return descrip_prompt
        if prompt_type == 'planning':
            if descrip != '{}' and subtask != '{}':
                planning_prompt = (f"The agent has been tasked with navigating to a {goal.upper()}. The agent has sent you the following elements:"
                f"(1)<The panoramic image>: The image describes your surrounding environment, each image contains a label indicating the relative rotation angle with blue fonts. "
                f"(2){descrip}. This dictionary describes the room type and the objects within it corresponding to each orientation angle, as well as the room type where the agent is currently located. "
                f'Your job is to select the best direction and describe next place to go. '
                f'To help you select the best direction and plan your next step, I can give you some human suggestions:. '
                f'(1) For each direction, first confirm whether there are floor in the view, do not choose the directions without seen navigable areas. '
                f'(2) Try to avoid going backwards (selecting 150,210), unless all the forward directions do not meet the requirements of (1). Note you CANNOT GO THROUGH CLOSED DOORS, and you DO NOT NEED TO GO UP OR DOWN STAIRS. '
                f'(3) If the {goal} appears in the panoramic image, select the direction in which the {goal} appears. '
                f'(4) If the {goal} is not found and the previous subtask {subtask} has not completed, choose one direction from the direction of angle 30 and angle 330 to continue to complete the last subtask {subtask} that has not been completed.'
                f'(5) If the {goal} is not found and the previous subtask {subtask} has already been completed. Identify a new subtask by describing where you are going next to be more likely to find clues to the the {goal} and analyze the room type and think about whether the {goal} is likely to occur in that direction. Note you need to pay special attention to open doors and hallways, as they can lead to other unseen rooms. '
                "Format your answer in the json {{'Subtask': <Where you are going next>, 'Angle': <Your Select Angle>, 'Flag': <Whether the target is in your selected view, True or False>}}. "
                "Answer Example: {{'Subtask': 'Go to the hallway', 'Angle': 30, 'Flag': False}} or {{'Subtask': "+f"'Go to the {goal}'"+", 'Angle': 150, 'Flag': True}} or {{'Subtask': 'Go to the open door', 'Angle': 150, 'Flag': True}}")
            else:
                planning_prompt = (f"The agent has been tasked with navigating to a {goal.upper()}. The agent has sent you the panoramic image describing your surrounding environment, each image contains a label indicating the relative rotation angle with blue fonts. "
                f'Your job is to select the best direction and describe next place to go. '
                f'To help you select the best direction and plan your next step, I can give you some human suggestions:. '
                f'(1) For each direction, first confirm whether there are floor in the view, do not choose the directions without seen navigable areas. '
                f'(2) Try to avoid going backwards (selecting 150,210), unless all the forward directions do not meet the requirements of (1). Note you CANNOT GO THROUGH CLOSED DOORS, and you DO NOT NEED TO GO UP OR DOWN STAIRS. '
                f'(3) If the {goal} appears in the panoramic image, select the direction in which the {goal} appears. '
                f'(4) If the {goal} is not found, describe where you are going next to be more likely to find clues to the the {goal} and analyze the room type and think about whether the {goal} is likely to occur in that direction. Note you need to pay special attention to open doors and hallways, as they can lead to other unseen rooms. '
                "Format your answer in the json {{'Subtask': <Where you are going next>, 'Angle': <Your Select Angle>, 'Flag': <Whether the target is in your selected view, True or False>}}. "
                "Answer Example: {{'Subtask': 'Go to the hallway', 'Angle': 30, 'Flag': False}} or {{'Subtask': " + f"'Go to the {goal}'" + ", 'Angle': 150, 'Flag': True}} or {{'Subtask': 'Go to the open door', 'Angle': 150, 'Flag': True}}")
            return planning_prompt
        if prompt_type == 'action':
            if subtask != '{}':
                action_prompt = (
                f"TASK: {subtask}. Your final task is to NAVIGATE TO THE NEAREST {goal.upper()}, and get as close to it as possible. "
                f"There are {num_actions - 1} red arrows superimposed onto your observation, which represent potential actions. " 
                f"These are labeled with a number in a white circle, which represent the location you would move to if you took that action. {'NOTE: choose action 0 if you want to TURN AROUND or DONT SEE ANY GOOD ACTIONS. ' if self.step_ndx - self.turned >= self.cfg['turn_around_cooldown'] else ''}"
                f"In order to complete the subtask {subtask} and eventually the final task NAVIGATING TO THE NEAREST {goal.upper()}. Explain which action acheives that best. "
                "Return your answer as {{'action': <action_key>}}. Note you CANNOT GO THROUGH CLOSED DOORS, and you DO NOT NEED TO GO UP OR DOWN STAIRS"
                )
            else:
                action_prompt = (
                    f"TASK: NAVIGATE TO THE NEAREST {goal.upper()}, and get as close to it as possible. Use your prior knowledge about where items are typically located within a home. "
                    f"There are {num_actions - 1} red arrows superimposed onto your observation, which represent potential actions. "
                    f"These are labeled with a number in a white circle, which represent the location you would move to if you took that action. {'NOTE: choose action 0 if you want to TURN AROUND or DONT SEE ANY GOOD ACTIONS. ' if self.step_ndx - self.turned >= self.cfg['turn_around_cooldown'] else ''}"
                    f"First, tell me what you see in your sensor observation, and if you have any leads on finding the {goal.upper()}. Second, tell me which general direction you should go in. "
                    "Lastly, explain which action acheives that best, and return it as {{'action': <action_key>}}. Note you CANNOT GO THROUGH CLOSED DOORS, and you DO NOT NEED TO GO UP OR DOWN STAIRS"
                )
            return action_prompt

        raise ValueError('Prompt type must be stopping, descrip, planning, or action')

class d(VLMNavAgent):
    def reset(self):
        self.voxel_map = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)
        self.explored_map = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)
        self.cvalue_map = 10 * np.ones((self.map_size, self.map_size, 3), dtype=np.float16)
        self.panoramic_mask = {}
        self.effective_mask = {}
        self.stopping_calls = [-2]
        self.step_ndx = 0
        self.init_pos = None
        self.turned = -self.cfg['turn_around_cooldown']
        self.ActionVLM.reset()
        self.PlanVLM.reset()
        self.stoppingVLM.reset()
        self.PredictVLM.reset()

    def _initialize_vlms(self, cfg: dict):
        vlm_cls = globals()[cfg['model_cls']]
        system_instruction = (
            "You are an embodied robotic assistant, with an RGB image sensor. You observe the image and instructions "
            "given to you and output a textual response, which is converted into actions that physically move you "
            "within the environment. You cannot move through closed doors. "
        )
        self.ActionVLM: VLM = vlm_cls(**cfg['model_kwargs'], system_instruction=system_instruction)
        self.PlanVLM: VLM = vlm_cls(**cfg['model_kwargs'])
        self.stoppingVLM: VLM = vlm_cls(**cfg['model_kwargs'])
        self.PredictVLM: VLM = vlm_cls(**cfg['model_kwargs'])

    def _update_panoramic_voxel(self, r: float, theta: float, agent_state: habitat_sim.AgentState, clip_dist: float,
                                clip_frac: float):
        """Update the voxel map to mark actions as explored"""
        agent_coords = self._global_to_grid(agent_state.position)

        # Mark circle explored regions
        clipped = min(clip_frac * r, clip_dist)
        local_coords = np.array([clipped * np.sin(theta), 0, -clipped * np.cos(theta)])
        global_coords = local_to_global(agent_state.position, agent_state.rotation, local_coords)
        point = self._global_to_grid(global_coords)

        radius = int(np.linalg.norm(np.array(agent_coords) - np.array(point)))
        cv2.circle(self.explored_map, agent_coords, radius, self.explored_color, -1)  # -1表示实心圆

    def _run_threads(self, obs: dict, stopping_images: list[np.array], goal):
        """Concurrently runs the stopping thread to determine if the agent should stop, and the preprocessing thread to calculate potential actions."""
        called_stop = False
        stopping_response = None
        with concurrent.futures.ThreadPoolExecutor() as executor:
            preprocessing_thread = executor.submit(self._preprocessing_module, obs)
            if obs['goal_flag'] == True:
                stopping_thread = executor.submit(self._stopping_module, stopping_images, goal)

            a_final, images = preprocessing_thread.result()
            if obs['goal_flag'] == True:
                called_stop, stopping_response = stopping_thread.result()

        if called_stop:
            logging.info('Model called stop')
            self.stopping_calls.append(self.step_ndx)
            # If the model calls stop, turn off navigability and explore bias tricks
            if self.cfg['navigability_mode'] != 'none':
                new_image = obs['color_sensor'].copy()
                a_final = self._project_onto_image(
                    self._get_default_arrows(), new_image, obs['agent_state'],
                    obs['agent_state'].sensor_states['color_sensor'],
                    step=self.step_ndx,
                    goal=obs['goal']
                )
                images['color_sensor'] = new_image

        step_metadata = {
            'action_number': -10,
            'success': 1,
            'model': self.ActionVLM.name,
            'agent_location': obs['agent_state'].position,
            'called_stopping': called_stop
        }
        return a_final, images, step_metadata, stopping_response

    def _draw_direction_arrow(self, roomtrack_map, direction_vector, position, coords, angle_text='', arrow_length=1):
        # 箭头的终点
        arrow_end = np.array(
            [position[0] + direction_vector[0] * arrow_length, position[1],
             position[2] + direction_vector[2] * arrow_length])  # 假设 y 轴是高度轴，不变

        # 将世界坐标转换为网格坐标
        arrow_end_coords = self._global_to_grid(arrow_end)

        # 绘制箭头
        cv2.arrowedLine(roomtrack_map, (coords[0], coords[1]),
                        (arrow_end_coords[0], arrow_end_coords[1]), WHITE, 4, tipLength=0.1)

        # 绘制文本，表示角度（假设为 30°，你可以根据实际情况调整）
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = 1
        text_thickness = 2

        # 获取文本的宽度和高度，用来居中文本
        (text_width, text_height), _ = cv2.getTextSize(angle_text, font, text_size, text_thickness)

        # 设置文本的位置为箭头终点稍微偏移
        text_end_coords = self._global_to_grid(np.array(
            [position[0] + direction_vector[0] * arrow_length * 1.5, position[1],
             position[2] + direction_vector[2] * arrow_length * 1.5]))
        text_position = (text_end_coords[0] - text_width // 2, text_end_coords[1] + text_height // 2)

        # 绘制文本
        cv2.putText(roomtrack_map, angle_text, text_position, font, text_size, (255, 255, 255), text_thickness,
                    cv2.LINE_AA)

    def generate_voxel(self, agent_state: habitat_sim.AgentState = None, zoom: int = 9):
        """For visualization purposes, add the agent's position and actions onto the voxel map"""
        agent_coords = self._global_to_grid(agent_state.position)
        right = (agent_state.position[0] + zoom, 0, agent_state.position[2])
        right_coords = self._global_to_grid(right)
        delta = abs(agent_coords[0] - right_coords[0])

        topdown_map = self.voxel_map.copy()
        mask = np.all(self.explored_map == self.explored_color, axis=-1)
        topdown_map[mask] = self.explored_color

        # direction vector
        direction_vector = habitat_sim.utils.quat_rotate_vector(agent_state.rotation, habitat_sim.geo.FRONT)

        self._draw_direction_arrow(topdown_map, direction_vector, agent_state.position, agent_coords,
                                   angle_text="0")
        theta_60 = -np.pi / 3
        theta_30 = -np.pi / 6
        y_axis = np.array([0, 1, 0])
        quat_60 = habitat_sim.utils.quat_from_angle_axis(theta_60, y_axis)
        quat_30 = habitat_sim.utils.quat_from_angle_axis(theta_30, y_axis)
        direction_30_vector = habitat_sim.utils.quat_rotate_vector(quat_30, direction_vector)
        self._draw_direction_arrow(topdown_map, direction_30_vector, agent_state.position, agent_coords,
                                   angle_text="30")
        direction_60_vector = direction_30_vector.copy()
        for i in range(5):
            direction_60_vector = habitat_sim.utils.quat_rotate_vector(quat_60, direction_60_vector)
            angle = (i + 1) * 60 + 30
            self._draw_direction_arrow(topdown_map, direction_60_vector, agent_state.position, agent_coords,
                                       angle_text=str(angle))

        text_size = 1.25
        text_thickness = 1
        x, y = agent_coords
        font = cv2.FONT_HERSHEY_SIMPLEX

        text = str(self.step_ndx)
        (text_width, text_height), _ = cv2.getTextSize(text, font, text_size, text_thickness)
        circle_center = (agent_coords[0], agent_coords[1])
        circle_radius = max(text_width, text_height) // 2 + 15

        cv2.circle(topdown_map, circle_center, circle_radius, WHITE, -1)

        text_position = (circle_center[0] - text_width // 2, circle_center[1] + text_height // 2)
        cv2.circle(topdown_map, circle_center, circle_radius, RED, 1)
        cv2.putText(topdown_map, text, text_position, font, text_size, RED, text_thickness + 1)

        # Zoom the map
        max_x, max_y = topdown_map.shape[1], topdown_map.shape[0]
        x1 = max(0, x - delta)
        x2 = min(max_x, x + delta)
        y1 = max(0, y - delta)
        y2 = min(max_y, y + delta)

        zoomed_map = topdown_map[y1:y2, x1:x2]

        if self.step_ndx is not None:
            step_text = f'step {self.step_ndx}'
            cv2.putText(zoomed_map, step_text, (30, 90), font, 3, (255, 255, 255), 2, cv2.LINE_AA)

        return zoomed_map

    def update_voxel(self, r: float, theta: float, agent_state: habitat_sim.AgentState, temp_map: np.ndarray,
                     effective_dist: float = 3):
        agent_coords = self._global_to_grid(agent_state.position)

        # Mark unexplored regions
        unclipped = max(r, 0)
        local_coords = np.array([unclipped * np.sin(theta), 0, -unclipped * np.cos(theta)])
        global_coords = local_to_global(agent_state.position, agent_state.rotation, local_coords)
        point = self._global_to_grid(global_coords)
        cv2.line(self.voxel_map, agent_coords, point, self.unexplored_color, 40)

        # Mark directional regions
        cv2.line(temp_map, agent_coords, point, WHITE, 40)  # whole area
        unclipped = min(r, effective_dist)
        local_coords = np.array([unclipped * np.sin(theta), 0, -unclipped * np.cos(theta)])
        global_coords = local_to_global(agent_state.position, agent_state.rotation, local_coords)
        point = self._global_to_grid(global_coords)
        cv2.line(temp_map, agent_coords, point, GREEN, 40)  # effective area

    def navigability(self, obs: dict, direction_idx: int):
        """Generates the set of navigability actions and updates the voxel map accordingly."""
        agent_state: habitat_sim.AgentState = obs['agent_state']
        if self.step_ndx == 0:
            self.init_pos = agent_state.position
        sensor_state = agent_state.sensor_states['color_sensor']
        rgb_image = obs['color_sensor']
        depth_image = obs[f'depth_sensor']
        if self.cfg['navigability_mode'] == 'depth_estimate':
            depth_image = self.depth_estimator.call(rgb_image)
        if self.cfg['navigability_mode'] == 'segmentation':
            depth_image = None

        navigability_mask = self._get_navigability_mask(
            rgb_image, depth_image, agent_state, sensor_state
        )

        sensor_range = np.deg2rad(self.fov / 2) * 1.5

        all_thetas = np.linspace(-sensor_range, sensor_range, 120)
        start = agent_frame_to_image_coords(
            [0, 0, 0], agent_state, sensor_state,
            resolution=self.resolution, focal_length=self.focal_length
        )  # agent的原点在图像坐标系的位置（像素）

        temp_map = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)
        for theta_i in all_thetas:
            r_i, theta_i = self._get_radial_distance(start, theta_i, navigability_mask, agent_state, sensor_state,
                                                     depth_image)
            if r_i is not None:
                self.update_voxel(
                    r_i, theta_i, agent_state, temp_map
                )
        angle = str(int(direction_idx * 30))
        self.panoramic_mask[angle] = np.all(temp_map == WHITE, axis=-1) | np.all(temp_map == GREEN, axis=-1)
        self.effective_mask[angle] = np.all(temp_map == GREEN, axis=-1)

    def _update_voxel(self, r: float, theta: float, agent_state: habitat_sim.AgentState, clip_dist: float,
                      clip_frac: float):
        """Update the voxel map to mark actions as explored or unexplored"""
        agent_coords = self._global_to_grid(agent_state.position)

        # Mark explored regions
        clipped = min(r, clip_dist)
        local_coords = np.array([clipped * np.sin(theta), 0, -clipped * np.cos(theta)])
        global_coords = local_to_global(agent_state.position, agent_state.rotation, local_coords)
        point = self._global_to_grid(global_coords)
        cv2.line(self.explored_map, agent_coords, point, self.explored_color, self.voxel_ray_size)

    def _navigability(self, obs: dict):
        """Generates the set of navigability actions and updates the voxel map accordingly."""
        agent_state: habitat_sim.AgentState = obs['agent_state']
        sensor_state = agent_state.sensor_states['color_sensor']
        rgb_image = obs['color_sensor']
        depth_image = obs[f'depth_sensor']
        if self.cfg['navigability_mode'] == 'depth_estimate':
            depth_image = self.depth_estimator.call(rgb_image)
        if self.cfg['navigability_mode'] == 'segmentation':
            depth_image = None

        navigability_mask = self._get_navigability_mask(
            rgb_image, depth_image, agent_state, sensor_state
        )

        sensor_range = np.deg2rad(self.fov / 2) * 1.5

        all_thetas = np.linspace(-sensor_range, sensor_range, self.cfg['num_theta'])
        start = agent_frame_to_image_coords(
            [0, 0, 0], agent_state, sensor_state,
            resolution=self.resolution, focal_length=self.focal_length
        )  # agent的原点在图像坐标系的位置（像素）

        a_initial = []
        for theta_i in all_thetas:
            r_i, theta_i = self._get_radial_distance(start, theta_i, navigability_mask, agent_state, sensor_state,
                                                     depth_image)
            if r_i is not None:
                self._update_voxel(
                    r_i, theta_i, agent_state,
                    clip_dist=self.cfg['max_action_dist'], clip_frac=self.e_i_scaling
                )
                a_initial.append((r_i, theta_i))

        # draw explored circle
        if self.cfg['panoramic_padding'] == True:
            r_max, theta_max = max(a_initial, key=lambda x: x[0])
            self._update_panoramic_voxel(r_max, theta_max, agent_state,
                                         clip_dist=self.cfg['max_action_dist'], clip_frac=self.e_i_scaling)
        return a_initial

    def get_spend(self):
        # 可以修改原始逻辑或添加新的逻辑
        original_spend = super().get_spend()  # 调用父类的 get_spend 方法
        # 假设我们要将父类的返回值加上一个额外的花费
        return original_spend + self.PlanVLM.get_spend() + self.PredictVLM.get_spend()

    def _prompting(self, goal, a_final: list, images: dict, step_metadata: dict, subtask: str):
        """
        Prompting component of BASEV2. Constructs the textual prompt and calls the action model.
        Parses the response for the chosen action number.
        """
        prompt_type = 'action'
        action_prompt = self._construct_prompt(goal, prompt_type, subtask, num_actions=len(a_final))

        prompt_images = [images['color_sensor']]
        if 'goal_image' in images:
            prompt_images.append(images['goal_image'])

        response = self.ActionVLM.call_chat(self.cfg['context_history'], prompt_images, action_prompt)

        logging_data = {}
        try:
            response_dict = self._eval_response(response)
            step_metadata['action_number'] = int(response_dict['action'])
        except (IndexError, KeyError, TypeError, ValueError) as e:
            logging.error(f'Error parsing response {e}')
            step_metadata['success'] = 0
        finally:
            logging_data['ACTION_NUMBER'] = step_metadata.get('action_number')
            logging_data['ACTION_PROMPT'] = action_prompt
            logging_data['ACTION_RESPONSE'] = response

        return step_metadata, logging_data, response

    def _choose_action(self, obs: dict):
        agent_state = obs['agent_state']
        goal = obs['goal']

        a_final, images, step_metadata, stopping_response = self._run_threads(obs, [obs['color_sensor']], goal)
        step_metadata['object'] = goal

        # If the model calls stop two times in a row, terminate the episode
        if step_metadata['called_stopping']:
            step_metadata['action_number'] = -1
            agent_action = PolarAction.stop
            logging_data = {}
        else:
            step_metadata, logging_data, _ = self._prompting(goal, a_final, images, step_metadata, obs['subtask'])
            agent_action = self._action_number_to_polar(step_metadata['action_number'], list(a_final))

        logging_data['STOPPING RESPONSE'] = stopping_response
        metadata = {
            'step_metadata': step_metadata,
            'logging_data': logging_data,
            'a_final': a_final,
            'images': images,
            'step': self.step_ndx
        }
        return agent_action, metadata

    def _eval_response(self, response: str):
        """Converts the VLM response string into a dictionary, if possible"""
        import re
        result = re.sub(r"(?<=[a-zA-Z])'(?=[a-zA-Z])", "\\'", response)
        try:
            eval_resp = ast.literal_eval(result[result.index('{') + 1:result.rindex('}')])  # {{}}
            if isinstance(eval_resp, dict):
                return eval_resp
        except:
            try:
                eval_resp = ast.literal_eval(result[result.rindex('{'):result.rindex('}') + 1])  # {}
                if isinstance(eval_resp, dict):
                    return eval_resp
            except:
                try:
                    eval_resp = ast.literal_eval(result[result.index('{'):result.rindex('}') + 1])  # {{}, {}}
                    if isinstance(eval_resp, dict):
                        return eval_resp
                except:
                    logging.error(f'Error parsing response {response}')
                    return {}

    def _planning_module(self, planning_image: list[np.array], previous_subtask, goal_reason: str, goal):
        """Determines if the agent should stop."""
        planning_prompt = self._construct_prompt(goal, 'planning', previous_subtask, goal_reason)
        planning_response = self.PlanVLM.call([planning_image], planning_prompt)
        planning_response = planning_response.replace('false', 'False').replace('true', 'True')
        dct = self._eval_response(planning_response)

        return dct

    def _predicting_module(self, evaluator_image, goal):
        """Determines if the agent should stop."""
        evaluator_prompt = self._construct_prompt(goal, 'predicting')
        evaluator_response = self.PredictVLM.call([evaluator_image], evaluator_prompt)
        dct = self._eval_response(evaluator_response)

        return dct

    @staticmethod
    def _concat_panoramic(images, angles):
        try:
            height, width = images[0].shape[0], images[0].shape[1]
        except:
            height, width = 480, 640
        background_image = np.zeros((2 * height + 3 * 10, 3 * width + 4 * 10, 3), np.uint8)
        copy_images = np.array(images, dtype=np.uint8)
        for i in range(len(copy_images)):
            if i % 2 == 0:
                continue
            copy_images[i] = cv2.putText(copy_images[i], "Angle %d" % angles[i], (100, 100), cv2.FONT_HERSHEY_SIMPLEX,
                                         2, (255, 0, 0), 6, cv2.LINE_AA)
            row = i // 6
            col = (i // 2) % 3
            background_image[10 * (row + 1) + row * height:10 * (row + 1) + row * height + height:,
            10 * (col + 1) + col * width:10 * (col + 1) + col * width + width, :] = copy_images[i]
        return background_image

    def make_curiosity_value(self, pano_images, goal):
        angles = (np.arange(len(pano_images))) * 30
        inference_image = self._concat_panoramic(pano_images, angles)

        response = self._predicting_module(inference_image, goal)

        explorable_value = {}
        reason = {}

        try:
            for angle, values in response.items():
                explorable_value[angle] = values['Score']
                reason[angle] = values['Explanation']
        except:
            explorable_value, reason = None, None

        return inference_image, explorable_value, reason

    @staticmethod
    def _merge_evalue(arr, num):
        return np.minimum(arr, num)

    def update_curiosity_value(self, explorable_value, reason):
        try:
            final_score = {}
            for i in range(12):
                if i % 2 == 0:
                    continue
                last_angle = str(int((i - 2) * 30)) if i != 1 else '330'
                angle = str(int(i * 30))
                next_angle = str(int((i + 2) * 30)) if i != 11 else '30'
                if np.all(self.panoramic_mask[angle] == False):
                    continue
                intersection1 = self.effective_mask[last_angle] & self.effective_mask[angle]
                intersection2 = self.effective_mask[angle] & self.effective_mask[next_angle]

                mask_minus_intersection = self.effective_mask[angle] & ~intersection1 & ~intersection2

                self.cvalue_map[mask_minus_intersection] = self._merge_evalue(self.cvalue_map[mask_minus_intersection],
                                                                              explorable_value[
                                                                                  angle])  # update explorable value map
                if np.all(intersection2 == False):
                    continue
                self.cvalue_map[intersection2] = self._merge_evalue(self.cvalue_map[intersection2], (
                            explorable_value[angle] + explorable_value[next_angle]) / 2)  # update explorable value map
            for i in range(12):
                if i % 2 == 0:
                    continue
                angle = str(int(i * 30))
                if np.all(self.panoramic_mask[angle] == False):
                    final_score[i] = explorable_value[angle]
                else:
                    final_score[i] = np.mean(self.cvalue_map[self.panoramic_mask[angle]])

            idx = max(final_score, key=final_score.get)
            final_reason = reason[str(int(idx * 30))]
        except:
            idx = np.random.randint(0, 12)
            final_reason = ''
        return idx, final_reason

    def draw_cvalue_map(self, agent_state: habitat_sim.AgentState = None, zoom: int = 9):
        agent_coords = self._global_to_grid(agent_state.position)
        right = (agent_state.position[0] + zoom, 0, agent_state.position[2])
        right_coords = self._global_to_grid(right)
        delta = abs(agent_coords[0] - right_coords[0])

        cvalue_map = (self.cvalue_map / 10 * 255).astype(np.uint8)

        x, y = agent_coords
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Zoom the map
        max_x, max_y = cvalue_map.shape[1], cvalue_map.shape[0]
        x1 = max(0, x - delta)
        x2 = min(max_x, x + delta)
        y1 = max(0, y - delta)
        y2 = min(max_y, y + delta)

        zoomed_map = cvalue_map[y1:y2, x1:x2]

        if self.step_ndx is not None:
            step_text = f'step {self.step_ndx}'
            cv2.putText(zoomed_map, step_text, (30, 90), font, 3, (0, 0, 0), 2, cv2.LINE_AA)

        return zoomed_map

    def make_plan(self, pano_images, previous_subtask, goal_reason, goal):
        response = self._planning_module(pano_images, previous_subtask, goal_reason, goal)

        try:
            goal_flag, subtask = response['Flag'], '{}'

        except:
            print("planning failed!")
            print('response:', response)
            goal_flag, subtask = False, '{}'

        return goal_flag, subtask

    def _construct_prompt(self, goal: str, prompt_type: str, subtask: str = '{}', reason: str = '{}',
                          num_actions: int = 0):
        if prompt_type == 'stopping':
            stopping_prompt = (
                f"The agent has been tasked with navigating to a {goal.upper()}. The agent has sent you an image taken from its current location. "
                f'Your job is to determine whether the agent is VERY CLOSE to a {goal}(within 1 meter). Note a chair must have a backrest and a chair is not a stool. Note a chair is NOT sofa(couch) which is NOT a bed. '
                f"First, tell me what you see in the image, and tell me if there is a {goal}. Second, return 1 if the agent is VERY CLOSE to the {goal} - make sure the object you see is ACTUALLY a {goal}, Return 0 if if there is no {goal}, or if it is far away, or if you are not sure. "
                "Format your answer in the json {{'done': <1 or 0>}}")
            return stopping_prompt
        if prompt_type == 'predicting':
            evaluator_prompt = (
                f"The agent has been tasked with navigating to a {goal.upper()}. The agent has sent you the panoramic image describing your surrounding environment, each image contains a label indicating the relative rotation angle(30, 90, 150, 210, 270, 330) with red fonts. "
                f'Your job is to assign a score to each direction (ranging from 0 to 10), judging whether this direction is worth exploring. The following criteria should be used: '
                f'To help you describe the layout of your surrounding,  please follow my step-by-step instructions: '
                f'(1) If there is no visible way to move to other areas and it is clear that the target is not in sight, assign a score of 0. Note a chair must have a backrest and a chair is not a stool. Note a chair is NOT sofa(couch) which is NOT a bed. '
                f'(2) If the {goal} is found, assign a score of 10.  '
                f'(3) If there is a way to move to another area, assign a score based on your estimate of the likelihood of finding a {goal}, using your common sense. Moving to another area means there is a turn in the corner, an open door, a hallway, etc. Note you CANNOT GO THROUGH CLOSED DOORS. CLOSED DOORS and GOING UP OR DOWN STAIRS are not considered. '
                "For each direction, provide an explanation for your assigned score. Format your answer in the json {'30': {'Score': <The score(from 0 to 10) of angle 30>, 'Explanation': <An explanation for your assigned score.>}, '90': {...}, '150': {...}, '210': {...}, '270': {...}, '330': {...}}. "
                "Answer Example: {'30': {'Score': 0, 'Explanation': 'Dead end with a recliner. No sign of a bed or any other room.'}, '90': {'Score': 2, 'Explanation': 'Dining area. It is possible there is a doorway leading to other rooms, but bedrooms are less likely to be directly adjacent to dining areas.'}, ..., '330': {'Score': 2, 'Explanation': 'Living room area with a recliner.  Similar to 270, there is a possibility of other rooms, but no strong indication of a bedroom.'}}")
            return evaluator_prompt
        if prompt_type == 'planning':
            planning_prompt = (
                f"The agent has been tasked with navigating to a {goal.upper()}. The agent has sent you an image taken from its current location."
                f'Your job is to describe next place to go. '
                f'To help you plan your best next step, I can give you some human suggestions:. '
                f'(1) If the {goal} appears in the image, directly choose the target as the next step in the plan. Note a chair must have a backrest and a chair is not a stool. Note a chair is NOT sofa(couch) which is NOT a bed. '
                f'(2) If the {goal} is not found, describe where you are going next to be more likely to find clues to the the {goal} and analyze the room type and think about whether the {goal} is likely to occur in that direction. Note you need to pay special attention to open doors and hallways, as they can lead to other unseen rooms. Note GOING UP OR DOWN STAIRS is an option. '
                "Format your answer in the json {{'Flag': <Whether the target is in your view, True or False>}}. "
                "Answer Example: {{'Flag': False}} or {{'Flag': True}} or {{'Flag': True}}")
            return planning_prompt
        if prompt_type == 'action':
            action_prompt = (
                f"TASK: NAVIGATE TO THE NEAREST {goal.upper()}, and get as close to it as possible. Use your prior knowledge about where items are typically located within a home. "
                f"There are {num_actions - 1} red arrows superimposed onto your observation, which represent potential actions. "
                f"These are labeled with a number in a white circle, which represent the location you would move to if you took that action. {'NOTE: choose action 0 if you want to TURN AROUND or DONT SEE ANY GOOD ACTIONS. ' if self.step_ndx - self.turned >= self.cfg['turn_around_cooldown'] else ''}"
                f"First, tell me what you see in your sensor observation, and if you have any leads on finding the {goal.upper()}. Second, tell me which general direction you should go in. "
                "Lastly, explain which action acheives that best, and return it as {{'action': <action_key>}}. Note you CANNOT GO THROUGH CLOSED DOORS, and you DO NOT NEED TO GO UP OR DOWN STAIRS"
            )
            return action_prompt

        raise ValueError('Prompt type must be stopping, predicting, planning, or action')

class e(VLMNavAgent):
    def reset(self):
        self.voxel_map = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)
        self.explored_map = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)
        self.cvalue_map = 10 * np.ones((self.map_size, self.map_size, 3), dtype=np.float16)
        self.panoramic_mask = {}
        self.effective_mask = {}
        self.stopping_calls = [-2]
        self.step_ndx = 0
        self.init_pos = None
        self.turned = -self.cfg['turn_around_cooldown']
        self.ActionVLM.reset()
        self.PlanVLM.reset()
        self.stoppingVLM.reset()
        self.PredictVLM.reset()

    def _initialize_vlms(self, cfg: dict):
        vlm_cls = globals()[cfg['model_cls']]
        system_instruction = (
            "You are an embodied robotic assistant, with an RGB image sensor. You observe the image and instructions "
            "given to you and output a textual response, which is converted into actions that physically move you "
            "within the environment. You cannot move through closed doors. "
        )
        self.ActionVLM: VLM = vlm_cls(**cfg['model_kwargs'], system_instruction=system_instruction)
        self.PlanVLM: VLM = vlm_cls(**cfg['model_kwargs'])
        self.stoppingVLM: VLM = vlm_cls(**cfg['model_kwargs'])
        self.PredictVLM: VLM = vlm_cls(**cfg['model_kwargs'])

    def _update_panoramic_voxel(self, r: float, theta: float, agent_state: habitat_sim.AgentState, clip_dist: float,
                                clip_frac: float):
        """Update the voxel map to mark actions as explored"""
        agent_coords = self._global_to_grid(agent_state.position)

        # Mark circle explored regions
        clipped = min(clip_frac * r, clip_dist)
        local_coords = np.array([clipped * np.sin(theta), 0, -clipped * np.cos(theta)])
        global_coords = local_to_global(agent_state.position, agent_state.rotation, local_coords)
        point = self._global_to_grid(global_coords)

        radius = int(np.linalg.norm(np.array(agent_coords) - np.array(point)))
        cv2.circle(self.explored_map, agent_coords, radius, self.explored_color, -1)  # -1表示实心圆

    def _run_threads(self, obs: dict, stopping_images: list[np.array], goal):
        """Concurrently runs the stopping thread to determine if the agent should stop, and the preprocessing thread to calculate potential actions."""
        called_stop = False
        stopping_response = None
        with concurrent.futures.ThreadPoolExecutor() as executor:
            preprocessing_thread = executor.submit(self._preprocessing_module, obs)
            if obs['goal_flag'] == True:
                stopping_thread = executor.submit(self._stopping_module, stopping_images, goal)

            a_final, images = preprocessing_thread.result()
            if obs['goal_flag'] == True:
                called_stop, stopping_response = stopping_thread.result()

        if called_stop:
            logging.info('Model called stop')
            self.stopping_calls.append(self.step_ndx)
            # If the model calls stop, turn off navigability and explore bias tricks
            if self.cfg['navigability_mode'] != 'none':
                new_image = obs['color_sensor'].copy()
                a_final = self._project_onto_image(
                    self._get_default_arrows(), new_image, obs['agent_state'],
                    obs['agent_state'].sensor_states['color_sensor'],
                    step=self.step_ndx,
                    goal=obs['goal']
                )
                images['color_sensor'] = new_image

        step_metadata = {
            'action_number': -10,
            'success': 1,
            'model': self.ActionVLM.name,
            'agent_location': obs['agent_state'].position,
            'called_stopping': called_stop
        }
        return a_final, images, step_metadata, stopping_response

    def _draw_direction_arrow(self, roomtrack_map, direction_vector, position, coords, angle_text='', arrow_length=1):
        # 箭头的终点
        arrow_end = np.array(
            [position[0] + direction_vector[0] * arrow_length, position[1],
             position[2] + direction_vector[2] * arrow_length])  # 假设 y 轴是高度轴，不变

        # 将世界坐标转换为网格坐标
        arrow_end_coords = self._global_to_grid(arrow_end)

        # 绘制箭头
        cv2.arrowedLine(roomtrack_map, (coords[0], coords[1]),
                        (arrow_end_coords[0], arrow_end_coords[1]), WHITE, 4, tipLength=0.1)

        # 绘制文本，表示角度（假设为 30°，你可以根据实际情况调整）
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = 1
        text_thickness = 2

        # 获取文本的宽度和高度，用来居中文本
        (text_width, text_height), _ = cv2.getTextSize(angle_text, font, text_size, text_thickness)

        # 设置文本的位置为箭头终点稍微偏移
        text_end_coords = self._global_to_grid(np.array(
            [position[0] + direction_vector[0] * arrow_length * 1.5, position[1],
             position[2] + direction_vector[2] * arrow_length * 1.5]))
        text_position = (text_end_coords[0] - text_width // 2, text_end_coords[1] + text_height // 2)

        # 绘制文本
        cv2.putText(roomtrack_map, angle_text, text_position, font, text_size, (255, 255, 255), text_thickness,
                    cv2.LINE_AA)

    def generate_voxel(self, agent_state: habitat_sim.AgentState = None, zoom: int = 9):
        """For visualization purposes, add the agent's position and actions onto the voxel map"""
        agent_coords = self._global_to_grid(agent_state.position)
        right = (agent_state.position[0] + zoom, 0, agent_state.position[2])
        right_coords = self._global_to_grid(right)
        delta = abs(agent_coords[0] - right_coords[0])

        topdown_map = self.voxel_map.copy()
        mask = np.all(self.explored_map == self.explored_color, axis=-1)
        topdown_map[mask] = self.explored_color

        # direction vector
        direction_vector = habitat_sim.utils.quat_rotate_vector(agent_state.rotation, habitat_sim.geo.FRONT)

        self._draw_direction_arrow(topdown_map, direction_vector, agent_state.position, agent_coords,
                                   angle_text="0")
        theta_60 = -np.pi / 3
        theta_30 = -np.pi / 6
        y_axis = np.array([0, 1, 0])
        quat_60 = habitat_sim.utils.quat_from_angle_axis(theta_60, y_axis)
        quat_30 = habitat_sim.utils.quat_from_angle_axis(theta_30, y_axis)
        direction_30_vector = habitat_sim.utils.quat_rotate_vector(quat_30, direction_vector)
        self._draw_direction_arrow(topdown_map, direction_30_vector, agent_state.position, agent_coords,
                                   angle_text="30")
        direction_60_vector = direction_30_vector.copy()
        for i in range(5):
            direction_60_vector = habitat_sim.utils.quat_rotate_vector(quat_60, direction_60_vector)
            angle = (i + 1) * 60 + 30
            self._draw_direction_arrow(topdown_map, direction_60_vector, agent_state.position, agent_coords,
                                       angle_text=str(angle))

        text_size = 1.25
        text_thickness = 1
        x, y = agent_coords
        font = cv2.FONT_HERSHEY_SIMPLEX

        text = str(self.step_ndx)
        (text_width, text_height), _ = cv2.getTextSize(text, font, text_size, text_thickness)
        circle_center = (agent_coords[0], agent_coords[1])
        circle_radius = max(text_width, text_height) // 2 + 15

        cv2.circle(topdown_map, circle_center, circle_radius, WHITE, -1)

        text_position = (circle_center[0] - text_width // 2, circle_center[1] + text_height // 2)
        cv2.circle(topdown_map, circle_center, circle_radius, RED, 1)
        cv2.putText(topdown_map, text, text_position, font, text_size, RED, text_thickness + 1)

        # Zoom the map
        max_x, max_y = topdown_map.shape[1], topdown_map.shape[0]
        x1 = max(0, x - delta)
        x2 = min(max_x, x + delta)
        y1 = max(0, y - delta)
        y2 = min(max_y, y + delta)

        zoomed_map = topdown_map[y1:y2, x1:x2]

        if self.step_ndx is not None:
            step_text = f'step {self.step_ndx}'
            cv2.putText(zoomed_map, step_text, (30, 90), font, 3, (255, 255, 255), 2, cv2.LINE_AA)

        return zoomed_map

    def update_voxel(self, r: float, theta: float, agent_state: habitat_sim.AgentState, temp_map: np.ndarray,
                     effective_dist: float = 3):
        agent_coords = self._global_to_grid(agent_state.position)

        # Mark unexplored regions
        unclipped = max(r, 0)
        local_coords = np.array([unclipped * np.sin(theta), 0, -unclipped * np.cos(theta)])
        global_coords = local_to_global(agent_state.position, agent_state.rotation, local_coords)
        point = self._global_to_grid(global_coords)
        cv2.line(self.voxel_map, agent_coords, point, self.unexplored_color, 40)

        # Mark directional regions
        cv2.line(temp_map, agent_coords, point, WHITE, 40)  # whole area
        unclipped = min(r, effective_dist)
        local_coords = np.array([unclipped * np.sin(theta), 0, -unclipped * np.cos(theta)])
        global_coords = local_to_global(agent_state.position, agent_state.rotation, local_coords)
        point = self._global_to_grid(global_coords)
        cv2.line(temp_map, agent_coords, point, GREEN, 40)  # effective area

    def navigability(self, obs: dict, direction_idx: int):
        """Generates the set of navigability actions and updates the voxel map accordingly."""
        agent_state: habitat_sim.AgentState = obs['agent_state']
        if self.step_ndx == 0:
            self.init_pos = agent_state.position
        sensor_state = agent_state.sensor_states['color_sensor']
        rgb_image = obs['color_sensor']
        depth_image = obs[f'depth_sensor']
        if self.cfg['navigability_mode'] == 'depth_estimate':
            depth_image = self.depth_estimator.call(rgb_image)
        if self.cfg['navigability_mode'] == 'segmentation':
            depth_image = None

        navigability_mask = self._get_navigability_mask(
            rgb_image, depth_image, agent_state, sensor_state
        )

        sensor_range = np.deg2rad(self.fov / 2) * 1.5

        all_thetas = np.linspace(-sensor_range, sensor_range, 120)
        start = agent_frame_to_image_coords(
            [0, 0, 0], agent_state, sensor_state,
            resolution=self.resolution, focal_length=self.focal_length
        )  # agent的原点在图像坐标系的位置（像素）

        temp_map = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)
        for theta_i in all_thetas:
            r_i, theta_i = self._get_radial_distance(start, theta_i, navigability_mask, agent_state, sensor_state,
                                                     depth_image)
            if r_i is not None:
                self.update_voxel(
                    r_i, theta_i, agent_state, temp_map
                )
        angle = str(int(direction_idx * 30))
        self.panoramic_mask[angle] = np.all(temp_map == WHITE, axis=-1) | np.all(temp_map == GREEN, axis=-1)
        self.effective_mask[angle] = np.all(temp_map == GREEN, axis=-1)

    def _update_voxel(self, r: float, theta: float, agent_state: habitat_sim.AgentState, clip_dist: float,
                      clip_frac: float):
        """Update the voxel map to mark actions as explored or unexplored"""
        agent_coords = self._global_to_grid(agent_state.position)

        # Mark explored regions
        clipped = min(r, clip_dist)
        local_coords = np.array([clipped * np.sin(theta), 0, -clipped * np.cos(theta)])
        global_coords = local_to_global(agent_state.position, agent_state.rotation, local_coords)
        point = self._global_to_grid(global_coords)
        cv2.line(self.explored_map, agent_coords, point, self.explored_color, self.voxel_ray_size)

    def _navigability(self, obs: dict):
        """Generates the set of navigability actions and updates the voxel map accordingly."""
        agent_state: habitat_sim.AgentState = obs['agent_state']
        sensor_state = agent_state.sensor_states['color_sensor']
        rgb_image = obs['color_sensor']
        depth_image = obs[f'depth_sensor']
        if self.cfg['navigability_mode'] == 'depth_estimate':
            depth_image = self.depth_estimator.call(rgb_image)
        if self.cfg['navigability_mode'] == 'segmentation':
            depth_image = None

        navigability_mask = self._get_navigability_mask(
            rgb_image, depth_image, agent_state, sensor_state
        )

        sensor_range = np.deg2rad(self.fov / 2) * 1.5

        all_thetas = np.linspace(-sensor_range, sensor_range, self.cfg['num_theta'])
        start = agent_frame_to_image_coords(
            [0, 0, 0], agent_state, sensor_state,
            resolution=self.resolution, focal_length=self.focal_length
        )  # agent的原点在图像坐标系的位置（像素）

        a_initial = []
        for theta_i in all_thetas:
            r_i, theta_i = self._get_radial_distance(start, theta_i, navigability_mask, agent_state, sensor_state,
                                                     depth_image)
            if r_i is not None:
                self._update_voxel(
                    r_i, theta_i, agent_state,
                    clip_dist=self.cfg['max_action_dist'], clip_frac=self.e_i_scaling
                )
                a_initial.append((r_i, theta_i))

        # draw explored circle
        if self.cfg['panoramic_padding'] == True:
            r_max, theta_max = max(a_initial, key=lambda x: x[0])
            self._update_panoramic_voxel(r_max, theta_max, agent_state,
                                         clip_dist=self.cfg['max_action_dist'], clip_frac=self.e_i_scaling)
        return a_initial

    def get_spend(self):
        # 可以修改原始逻辑或添加新的逻辑
        original_spend = super().get_spend()  # 调用父类的 get_spend 方法
        # 假设我们要将父类的返回值加上一个额外的花费
        return original_spend + self.PlanVLM.get_spend() + self.PredictVLM.get_spend()

    def _prompting(self, goal, a_final: list, images: dict, step_metadata: dict, subtask: str):
        """
        Prompting component of BASEV2. Constructs the textual prompt and calls the action model.
        Parses the response for the chosen action number.
        """
        prompt_type = 'action'
        action_prompt = self._construct_prompt(goal, prompt_type, subtask, num_actions=len(a_final))

        prompt_images = [images['color_sensor']]
        if 'goal_image' in images:
            prompt_images.append(images['goal_image'])

        response = self.ActionVLM.call_chat(self.cfg['context_history'], prompt_images, action_prompt)

        logging_data = {}
        try:
            response_dict = self._eval_response(response)
            step_metadata['action_number'] = int(response_dict['action'])
        except (IndexError, KeyError, TypeError, ValueError) as e:
            logging.error(f'Error parsing response {e}')
            step_metadata['success'] = 0
        finally:
            logging_data['ACTION_NUMBER'] = step_metadata.get('action_number')
            logging_data['ACTION_PROMPT'] = action_prompt
            logging_data['ACTION_RESPONSE'] = response

        return step_metadata, logging_data, response

    def _choose_action(self, obs: dict):
        agent_state = obs['agent_state']
        goal = obs['goal']

        a_final, images, step_metadata, stopping_response = self._run_threads(obs, [obs['color_sensor']], goal)
        step_metadata['object'] = goal

        # If the model calls stop two times in a row, terminate the episode
        if step_metadata['called_stopping']:
            step_metadata['action_number'] = -1
            agent_action = PolarAction.stop
            logging_data = {}
        else:
            step_metadata, logging_data, _ = self._prompting(goal, a_final, images, step_metadata, obs['subtask'])
            agent_action = self._action_number_to_polar(step_metadata['action_number'], list(a_final))

        logging_data['STOPPING RESPONSE'] = stopping_response
        metadata = {
            'step_metadata': step_metadata,
            'logging_data': logging_data,
            'a_final': a_final,
            'images': images,
            'step': self.step_ndx
        }
        return agent_action, metadata

    def _eval_response(self, response: str):
        """Converts the VLM response string into a dictionary, if possible"""
        import re
        result = re.sub(r"(?<=[a-zA-Z])'(?=[a-zA-Z])", "\\'", response)
        try:
            eval_resp = ast.literal_eval(result[result.index('{') + 1:result.rindex('}')])  # {{}}
            if isinstance(eval_resp, dict):
                return eval_resp
        except:
            try:
                eval_resp = ast.literal_eval(result[result.rindex('{'):result.rindex('}') + 1])  # {}
                if isinstance(eval_resp, dict):
                    return eval_resp
            except:
                try:
                    eval_resp = ast.literal_eval(result[result.index('{'):result.rindex('}') + 1])  # {{}, {}}
                    if isinstance(eval_resp, dict):
                        return eval_resp
                except:
                    logging.error(f'Error parsing response {response}')
                    return {}

    def _planning_module(self, planning_image: list[np.array], previous_subtask, goal_reason: str, goal):
        """Determines if the agent should stop."""
        planning_prompt = self._construct_prompt(goal, 'planning', previous_subtask, goal_reason)
        planning_response = self.PlanVLM.call([planning_image], planning_prompt)
        planning_response = planning_response.replace('false', 'False').replace('true', 'True')
        dct = self._eval_response(planning_response)

        return dct

    def _predicting_module(self, evaluator_image, goal):
        """Determines if the agent should stop."""
        evaluator_prompt = self._construct_prompt(goal, 'predicting')
        evaluator_response = self.PredictVLM.call([evaluator_image], evaluator_prompt)
        dct = self._eval_response(evaluator_response)

        return dct

    @staticmethod
    def _concat_panoramic(images, angles):
        try:
            height, width = images[0].shape[0], images[0].shape[1]
        except:
            height, width = 480, 640
        background_image = np.zeros((2 * height + 3 * 10, 3 * width + 4 * 10, 3), np.uint8)
        copy_images = np.array(images, dtype=np.uint8)
        for i in range(len(copy_images)):
            if i % 2 == 0:
                continue
            copy_images[i] = cv2.putText(copy_images[i], "Angle %d" % angles[i], (100, 100), cv2.FONT_HERSHEY_SIMPLEX,
                                         2, (255, 0, 0), 6, cv2.LINE_AA)
            row = i // 6
            col = (i // 2) % 3
            background_image[10 * (row + 1) + row * height:10 * (row + 1) + row * height + height:,
            10 * (col + 1) + col * width:10 * (col + 1) + col * width + width, :] = copy_images[i]
        return background_image

    def make_curiosity_value(self, pano_images, goal):
        angles = (np.arange(len(pano_images))) * 30
        inference_image = self._concat_panoramic(pano_images, angles)

        response = self._predicting_module(inference_image, goal)

        explorable_value = {}
        reason = {}

        try:
            for angle, values in response.items():
                explorable_value[angle] = values['Score']
                reason[angle] = values['Explanation']
        except:
            explorable_value, reason = None, None

        return inference_image, explorable_value, reason

    @staticmethod
    def _merge_evalue(arr, num):
        return np.minimum(arr, num)

    def update_curiosity_value(self, explorable_value, reason):
        try:
            final_score = {}
            for i in range(12):
                if i % 2 == 0:
                    continue
                last_angle = str(int((i - 2) * 30)) if i != 1 else '330'
                angle = str(int(i * 30))
                next_angle = str(int((i + 2) * 30)) if i != 11 else '30'
                if np.all(self.panoramic_mask[angle] == False):
                    continue
                intersection1 = self.effective_mask[last_angle] & self.effective_mask[angle]
                intersection2 = self.effective_mask[angle] & self.effective_mask[next_angle]

                mask_minus_intersection = self.effective_mask[angle] & ~intersection1 & ~intersection2

                self.cvalue_map[mask_minus_intersection] = self._merge_evalue(self.cvalue_map[mask_minus_intersection],
                                                                              explorable_value[
                                                                                  angle])  # update explorable value map
                if np.all(intersection2 == False):
                    continue
                self.cvalue_map[intersection2] = self._merge_evalue(self.cvalue_map[intersection2], (
                            explorable_value[angle] + explorable_value[next_angle]) / 2)  # update explorable value map
            for i in range(12):
                if i % 2 == 0:
                    continue
                angle = str(int(i * 30))
                if np.all(self.panoramic_mask[angle] == False):
                    final_score[i] = explorable_value[angle]
                else:
                    final_score[i] = np.mean(self.cvalue_map[self.panoramic_mask[angle]])

            idx = max(final_score, key=final_score.get)
            final_reason = reason[str(int(idx * 30))]
        except:
            idx = np.random.randint(0, 12)
            final_reason = ''
        return idx, final_reason

    def draw_cvalue_map(self, agent_state: habitat_sim.AgentState = None, zoom: int = 9):
        agent_coords = self._global_to_grid(agent_state.position)
        right = (agent_state.position[0] + zoom, 0, agent_state.position[2])
        right_coords = self._global_to_grid(right)
        delta = abs(agent_coords[0] - right_coords[0])

        cvalue_map = (self.cvalue_map / 10 * 255).astype(np.uint8)

        x, y = agent_coords
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Zoom the map
        max_x, max_y = cvalue_map.shape[1], cvalue_map.shape[0]
        x1 = max(0, x - delta)
        x2 = min(max_x, x + delta)
        y1 = max(0, y - delta)
        y2 = min(max_y, y + delta)

        zoomed_map = cvalue_map[y1:y2, x1:x2]

        if self.step_ndx is not None:
            step_text = f'step {self.step_ndx}'
            cv2.putText(zoomed_map, step_text, (30, 90), font, 3, (0, 0, 0), 2, cv2.LINE_AA)

        return zoomed_map

    def make_plan(self, pano_images, previous_subtask, goal_reason, goal):
        response = self._planning_module(pano_images, previous_subtask, goal_reason, goal)

        try:
            goal_flag, subtask = response['Flag'], response['Subtask']
        except:
            print("planning failed!")
            print('response:', response)
            goal_flag, subtask = False, '{}'

        return goal_flag, subtask

    def _construct_prompt(self, goal: str, prompt_type: str, subtask: str = '{}', reason: str = '{}',
                          num_actions: int = 0):
        if prompt_type == 'stopping':
            stopping_prompt = (
                f"The agent has been tasked with navigating to a {goal.upper()}. The agent has sent you an image taken from its current location. "
                f'Your job is to determine whether the agent is VERY CLOSE to a {goal}(within 1 meter). Note a chair must have a backrest and a chair is not a stool. Note a chair is NOT sofa(couch) which is NOT a bed. '
                f"First, tell me what you see in the image, and tell me if there is a {goal}. Second, return 1 if the agent is VERY CLOSE to the {goal} - make sure the object you see is ACTUALLY a {goal}, Return 0 if if there is no {goal}, or if it is far away, or if you are not sure. "
                "Format your answer in the json {{'done': <1 or 0>}}")
            return stopping_prompt
        if prompt_type == 'predicting':
            evaluator_prompt = (
                f"The agent has been tasked with navigating to a {goal.upper()}. The agent has sent you the panoramic image describing your surrounding environment, each image contains a label indicating the relative rotation angle(30, 90, 150, 210, 270, 330) with red fonts. "
                f'Your job is to assign a score to each direction (ranging from 0 to 10), judging whether this direction is worth exploring. The following criteria should be used: '
                f'To help you describe the layout of your surrounding,  please follow my step-by-step instructions: '
                f'(1) If there is no visible way to move to other areas and it is clear that the target is not in sight, assign a score of 0. Note a chair must have a backrest and a chair is not a stool. Note a chair is NOT sofa(couch) which is NOT a bed. '
                f'(2) If the {goal} is found, assign a score of 10.  '
                f'(3) If there is a way to move to another area, assign a score based on your estimate of the likelihood of finding a {goal}, using your common sense. Moving to another area means there is a turn in the corner, an open door, a hallway, etc. Note you CANNOT GO THROUGH CLOSED DOORS. CLOSED DOORS and GOING UP OR DOWN STAIRS are not considered. '
                "For each direction, provide an explanation for your assigned score. Format your answer in the json {'30': {'Score': <The score(from 0 to 10) of angle 30>, 'Explanation': <An explanation for your assigned score.>}, '90': {...}, '150': {...}, '210': {...}, '270': {...}, '330': {...}}. "
                "Answer Example: {'30': {'Score': 0, 'Explanation': 'Dead end with a recliner. No sign of a bed or any other room.'}, '90': {'Score': 2, 'Explanation': 'Dining area. It is possible there is a doorway leading to other rooms, but bedrooms are less likely to be directly adjacent to dining areas.'}, ..., '330': {'Score': 2, 'Explanation': 'Living room area with a recliner.  Similar to 270, there is a possibility of other rooms, but no strong indication of a bedroom.'}}")
            return evaluator_prompt
        if prompt_type == 'planning':
            if reason != '' and subtask != '{}':
                planning_prompt = (
                            f"The agent has been tasked with navigating to a {goal.upper()}. The agent has sent you the following elements:"
                            f"(1)<The observed image>: The image taken from its current location. "
                            f"(2){reason}. This explains why you should go in this direction. "
                            f'Your job is to describe next place to go. '
                            f'To help you plan your best next step, I can give you some human suggestions:. '
                            f'(1) If the {goal} appears in the image, directly choose the target as the next step in the plan. Note a chair must have a backrest and a chair is not a stool. Note a chair is NOT sofa(couch) which is NOT a bed. '
                            f'(2) If the {goal} is not found and the previous subtask {subtask} has not completed, continue to complete the last subtask {subtask} that has not been completed.'
                            f'(3) If the {goal} is not found and the previous subtask {subtask} has already been completed. Identify a new subtask by describing where you are going next to be more likely to find clues to the the {goal} and think about whether the {goal} is likely to occur in that direction. Note you need to pay special attention to open doors and hallways, as they can lead to other unseen rooms. Note GOING UP OR DOWN STAIRS is an option. '
                            "Format your answer in the json {{'Subtask': <Where you are going next>, 'Flag': <Whether the target is in your view, True or False>}}. "
                            "Answer Example: {{'Subtask': 'Go to the hallway', 'Flag': False}} or {{'Subtask': " + f"'Go to the {goal}'" + ", 'Flag': True}} or {{'Subtask': 'Go to the open door', 'Flag': True}}")
            else:
                planning_prompt = (
                            f"The agent has been tasked with navigating to a {goal.upper()}. The agent has sent you an image taken from its current location."
                            f'Your job is to describe next place to go. '
                            f'To help you plan your best next step, I can give you some human suggestions:. '
                            f'(1) If the {goal} appears in the image, directly choose the target as the next step in the plan. Note a chair must have a backrest and a chair is not a stool. Note a chair is NOT sofa(couch) which is NOT a bed. '
                            f'(2) If the {goal} is not found, describe where you are going next to be more likely to find clues to the the {goal} and analyze the room type and think about whether the {goal} is likely to occur in that direction. Note you need to pay special attention to open doors and hallways, as they can lead to other unseen rooms. Note GOING UP OR DOWN STAIRS is an option. '
                            "Format your answer in the json {{'Subtask': <Where you are going next>, 'Flag': <Whether the target is in your view, True or False>}}. "
                            "Answer Example: {{'Subtask': 'Go to the hallway', 'Flag': False}} or {{'Subtask': " + f"'Go to the {goal}'" + ", 'Flag': True}} or {{'Subtask': 'Go to the open door', 'Flag': True}}")
            return planning_prompt
        if prompt_type == 'action':
            if subtask != '{}':
                action_prompt = (
                    f"TASK: {subtask}. Your final task is to NAVIGATE TO THE NEAREST {goal.upper()}, and get as close to it as possible. "
                    f"There are {num_actions - 1} red arrows superimposed onto your observation, which represent potential actions. "
                    f"These are labeled with a number in a white circle, which represent the location you would move to if you took that action. {'NOTE: choose action 0 if you want to TURN AROUND or DONT SEE ANY GOOD ACTIONS. ' if self.step_ndx - self.turned >= self.cfg['turn_around_cooldown'] else ''}"
                    f"In order to complete the subtask {subtask} and eventually the final task NAVIGATING TO THE NEAREST {goal.upper()}. Explain which action acheives that best. "
                    "Return your answer as {{'action': <action_key>}}. Note you CANNOT GO THROUGH CLOSED DOORS, and you DO NOT NEED TO GO UP OR DOWN STAIRS"
                )
            else:
                action_prompt = (
                    f"TASK: NAVIGATE TO THE NEAREST {goal.upper()}, and get as close to it as possible. Use your prior knowledge about where items are typically located within a home. "
                    f"There are {num_actions - 1} red arrows superimposed onto your observation, which represent potential actions. "
                    f"These are labeled with a number in a white circle, which represent the location you would move to if you took that action. {'NOTE: choose action 0 if you want to TURN AROUND or DONT SEE ANY GOOD ACTIONS. ' if self.step_ndx - self.turned >= self.cfg['turn_around_cooldown'] else ''}"
                    f"First, tell me what you see in your sensor observation, and if you have any leads on finding the {goal.upper()}. Second, tell me which general direction you should go in. "
                    "Lastly, explain which action acheives that best, and return it as {{'action': <action_key>}}. Note you CANNOT GO THROUGH CLOSED DOORS, and you DO NOT NEED TO GO UP OR DOWN STAIRS"
                )
            return action_prompt

        raise ValueError('Prompt type must be stopping, predicting, planning, or action')

class f(VLMNavAgent):
    def reset(self):
        self.voxel_map = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)
        self.explored_map = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)
        self.cvalue_map = 10 * np.ones((self.map_size, self.map_size, 3), dtype=np.float16)
        self.goal_position = []
        self.goal_mask = None
        self.panoramic_mask = {}
        self.effective_mask = {}
        self.stopping_calls = [-2]
        self.step_ndx = 0
        self.init_pos = None
        self.turned = -self.cfg['turn_around_cooldown']
        self.ActionVLM.reset()
        self.PlanVLM.reset()
        self.PredictVLM.reset()
        self.GoalVLM.reset()

    def _initialize_vlms(self, cfg: dict):
        vlm_cls = globals()[cfg['model_cls']]
        system_instruction = (
            "You are an embodied robotic assistant, with an RGB image sensor. You observe the image and instructions "
            "given to you and output a textual response, which is converted into actions that physically move you "
            "within the environment. You cannot move through closed doors. "
        )
        self.ActionVLM: VLM = vlm_cls(**cfg['model_kwargs'], system_instruction=system_instruction)
        self.PlanVLM: VLM = vlm_cls(**cfg['model_kwargs'])
        self.GoalVLM: VLM = vlm_cls(**cfg['model_kwargs'])
        self.PredictVLM: VLM = vlm_cls(**cfg['model_kwargs'])

    def _update_panoramic_voxel(self, r: float, theta: float, agent_state: habitat_sim.AgentState, clip_dist: float, clip_frac: float):
        """Update the voxel map to mark actions as explored"""
        agent_coords = self._global_to_grid(agent_state.position)

        # Mark circle explored regions
        clipped = min(clip_frac * r, clip_dist)
        local_coords = np.array([clipped * np.sin(theta), 0, -clipped * np.cos(theta)])
        global_coords = local_to_global(agent_state.position, agent_state.rotation, local_coords)
        point = self._global_to_grid(global_coords)

        radius = int(np.linalg.norm(np.array(agent_coords) - np.array(point)))
        cv2.circle(self.explored_map, agent_coords, radius, self.explored_color, -1)  # -1表示实心圆

    def _stopping_module(self, obs, threshold_dist=0.8):
        if self.goal_position:
            arr = np.array(self.goal_position)
            # 按列计算平均值
            avg_goal_position = np.mean(arr, axis=0)
            agent_state = obs['agent_state']
            current_position = np.array([agent_state.position[0], agent_state.position[2]])
            goal_position = np.array([avg_goal_position[0], avg_goal_position[2]])
            dist = np.linalg.norm(current_position - goal_position)

            if dist < threshold_dist:
                return True
        return False


    def _run_threads(self, obs: dict, stopping_images: list[np.array], goal):
        """Concurrently runs the stopping thread to determine if the agent should stop, and the preprocessing thread to calculate potential actions."""
        called_stop = False
        with concurrent.futures.ThreadPoolExecutor() as executor:
            preprocessing_thread = executor.submit(self._preprocessing_module, obs)
            stopping_thread = executor.submit(self._stopping_module, obs)

            a_final, images, a_goal, candidate_images = preprocessing_thread.result()
            called_stop = stopping_thread.result()

        if called_stop:
            logging.info('Model called stop')
            self.stopping_calls.append(self.step_ndx)
            # If the model calls stop, turn off navigability and explore bias tricks
            if self.cfg['navigability_mode'] != 'none':
                new_image = obs['color_sensor'].copy()
                a_final = self._project_onto_image(
                    self._get_default_arrows(), new_image, obs['agent_state'],
                    obs['agent_state'].sensor_states['color_sensor'],
                    step=self.step_ndx,
                    goal=obs['goal']
                )
                images['color_sensor'] = new_image

        step_metadata = {
            'action_number': -10,
            'success': 1,
            'model': self.ActionVLM.name,
            'agent_location': obs['agent_state'].position,
            'called_stopping': called_stop
        }
        return a_final, images, step_metadata, a_goal, candidate_images

    def _draw_direction_arrow(self, roomtrack_map, direction_vector, position, coords, angle_text='', arrow_length=1):
        # 箭头的终点
        arrow_end = np.array(
            [position[0] + direction_vector[0] * arrow_length, position[1],
             position[2] + direction_vector[2] * arrow_length])  # 假设 y 轴是高度轴，不变

        # 将世界坐标转换为网格坐标
        arrow_end_coords = self._global_to_grid(arrow_end)

        # 绘制箭头
        cv2.arrowedLine(roomtrack_map, (coords[0], coords[1]),
                        (arrow_end_coords[0], arrow_end_coords[1]), WHITE, 4, tipLength=0.1)

        # 绘制文本，表示角度（假设为 30°，你可以根据实际情况调整）
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = 1
        text_thickness = 2

        # 获取文本的宽度和高度，用来居中文本
        (text_width, text_height), _ = cv2.getTextSize(angle_text, font, text_size, text_thickness)

        # 设置文本的位置为箭头终点稍微偏移
        text_end_coords = self._global_to_grid(np.array(
            [position[0] + direction_vector[0] * arrow_length * 1.5, position[1],
             position[2] + direction_vector[2] * arrow_length * 1.5]))
        text_position = (text_end_coords[0] - text_width // 2, text_end_coords[1] + text_height // 2)

        # 绘制文本
        cv2.putText(roomtrack_map, angle_text, text_position, font, text_size, (255, 255, 255), text_thickness,
                    cv2.LINE_AA)

    def generate_voxel(self, agent_state: habitat_sim.AgentState=None, zoom: int=9):
        """For visualization purposes, add the agent's position and actions onto the voxel map"""
        agent_coords = self._global_to_grid(agent_state.position)
        right = (agent_state.position[0] + zoom, 0, agent_state.position[2])
        right_coords = self._global_to_grid(right)
        delta = abs(agent_coords[0] - right_coords[0])

        topdown_map = self.voxel_map.copy()
        mask = np.all(self.explored_map == self.explored_color, axis=-1)
        topdown_map[mask] = self.explored_color

        # direction vector
        direction_vector = habitat_sim.utils.quat_rotate_vector(agent_state.rotation, habitat_sim.geo.FRONT)

        self._draw_direction_arrow(topdown_map, direction_vector, agent_state.position, agent_coords,
                                   angle_text="0")
        theta_60 = -np.pi / 3
        theta_30 = -np.pi / 6
        y_axis = np.array([0, 1, 0])
        quat_60 = habitat_sim.utils.quat_from_angle_axis(theta_60, y_axis)
        quat_30 = habitat_sim.utils.quat_from_angle_axis(theta_30, y_axis)
        direction_30_vector = habitat_sim.utils.quat_rotate_vector(quat_30, direction_vector)
        self._draw_direction_arrow(topdown_map, direction_30_vector, agent_state.position, agent_coords,
                                   angle_text="30")
        direction_60_vector = direction_30_vector.copy()
        for i in range(5):
            direction_60_vector = habitat_sim.utils.quat_rotate_vector(quat_60, direction_60_vector)
            angle = (i + 1) * 60 + 30
            self._draw_direction_arrow(topdown_map, direction_60_vector, agent_state.position, agent_coords,
                                       angle_text=str(angle))

        text_size = 1.25
        text_thickness = 1
        x, y = agent_coords
        font = cv2.FONT_HERSHEY_SIMPLEX

        text = str(self.step_ndx)
        (text_width, text_height), _ = cv2.getTextSize(text, font, text_size, text_thickness)
        circle_center = (agent_coords[0], agent_coords[1])
        circle_radius = max(text_width, text_height) // 2 + 15

        cv2.circle(topdown_map, circle_center, circle_radius, WHITE, -1)

        text_position = (circle_center[0] - text_width // 2, circle_center[1] + text_height // 2)
        cv2.circle(topdown_map, circle_center, circle_radius, RED, 1)
        cv2.putText(topdown_map, text, text_position, font, text_size, RED, text_thickness + 1)


        # Zoom the map
        max_x, max_y = topdown_map.shape[1], topdown_map.shape[0]
        x1 = max(0, x - delta)
        x2 = min(max_x, x + delta)
        y1 = max(0, y - delta)
        y2 = min(max_y, y + delta)

        zoomed_map = topdown_map[y1:y2, x1:x2]

        if self.step_ndx is not None:
            step_text = f'step {self.step_ndx}'
            cv2.putText(zoomed_map, step_text, (30, 90), font, 3, (255, 255, 255), 2, cv2.LINE_AA)

        return zoomed_map

    def update_voxel(self, r: float, theta: float, agent_state: habitat_sim.AgentState, temp_map: np.ndarray, effective_dist: float=3):
        agent_coords = self._global_to_grid(agent_state.position)

        # Mark unexplored regions
        unclipped = max(r, 0)
        local_coords = np.array([unclipped * np.sin(theta), 0, -unclipped * np.cos(theta)])
        global_coords = local_to_global(agent_state.position, agent_state.rotation, local_coords)
        point = self._global_to_grid(global_coords)
        cv2.line(self.voxel_map, agent_coords, point, self.unexplored_color, 40)

        # Mark directional regions
        cv2.line(temp_map, agent_coords, point, WHITE, 40) # whole area
        unclipped = min(r, effective_dist)
        local_coords = np.array([unclipped * np.sin(theta), 0, -unclipped * np.cos(theta)])
        global_coords = local_to_global(agent_state.position, agent_state.rotation, local_coords)
        point = self._global_to_grid(global_coords)
        cv2.line(temp_map, agent_coords, point, GREEN, 40) # effective area

    def _goal_proposer(self, a_initial: list, agent_state: habitat_sim.AgentState):
        min_angle = self.fov / self.cfg['spacing_ratio']

        unique = {}
        for mag, theta in a_initial:
            if theta in unique:
                unique[theta].append(mag)
            else:
                unique[theta] = [mag]

        arrowData = []

        for theta, mags in unique.items():
            mag = min(mags)
            arrowData.append([mag, theta])

        arrowData.sort(key=lambda x: x[1])
        thetas = set()
        out = []
        filter_thresh = 0
        f = list(filter(lambda x: x[0] > filter_thresh, arrowData))

        f.sort(key=lambda x: x[1])
        if f == []:
            return []
        # Add unexplored actions with spacing, starting with the longest one
        if len(f) > 0:
            longest = max(f, key=lambda x: x[0])
            longest_theta = longest[1]
            smallest_theta = longest[1]
            longest_ndx = f.index(longest)

            out.append([longest[0], longest[1]])
            thetas.add(longest[1])
            for i in range(longest_ndx + 1, len(f)):
                if f[i][1] - longest_theta > (min_angle * 0.45):
                    out.append([f[i][0], f[i][1]])
                    thetas.add(f[i][1])
                    longest_theta = f[i][1]
            for i in range(longest_ndx - 1, -1, -1):
                if smallest_theta - f[i][1] > (min_angle * 0.45):
                    out.append([f[i][0], f[i][1]])
                    thetas.add(f[i][1])
                    smallest_theta = f[i][1]

        out.sort(key=lambda x: x[1])
        return [(mag, theta) for mag, theta in out]

    def _preprocessing_module(self, obs: dict):
        """Excutes the navigability, action_proposer and projection submodules."""
        agent_state = obs['agent_state']
        images = {'color_sensor': obs['color_sensor'].copy()}
        candidate_images = {'color_sensor': obs['color_sensor'].copy()}
        a_goal_projected = None

        if self.cfg['navigability_mode'] == 'none':
            a_final = [
                # Actions for the w/o nav baseline
                (self.cfg['max_action_dist'], -0.36 * np.pi),
                (self.cfg['max_action_dist'], -0.28 * np.pi),
                (self.cfg['max_action_dist'], 0),
                (self.cfg['max_action_dist'], 0.28 * np.pi),
                (self.cfg['max_action_dist'], 0.36 * np.pi)
            ]
        else:
            a_initial = self._navigability(obs)
            a_final = self._action_proposer(a_initial, agent_state)
            if obs['goal_flag']:
                a_goal = self._goal_proposer(a_initial, agent_state)

        a_final_projected = self._projection(a_final, images, agent_state, obs['goal'])
        if obs['goal_flag']:
            a_goal_projected = self._projection(a_goal, candidate_images, agent_state, obs['goal'], candidate_flag=True)
        images['voxel_map'] = self._generate_voxel(a_final_projected, agent_state=agent_state, step=self.step_ndx)
        return a_final_projected, images, a_goal_projected, candidate_images

    def navigability(self, obs: dict, direction_idx: int):
        """Generates the set of navigability actions and updates the voxel map accordingly."""
        agent_state: habitat_sim.AgentState = obs['agent_state']
        if self.step_ndx == 0:
            self.init_pos = agent_state.position
        sensor_state = agent_state.sensor_states['color_sensor']
        rgb_image = obs['color_sensor']
        depth_image = obs[f'depth_sensor']
        if self.cfg['navigability_mode'] == 'depth_estimate':
            depth_image = self.depth_estimator.call(rgb_image)
        if self.cfg['navigability_mode'] == 'segmentation':
            depth_image = None

        navigability_mask = self._get_navigability_mask(
            rgb_image, depth_image, agent_state, sensor_state
        )

        sensor_range = np.deg2rad(self.fov / 2) * 1.5

        all_thetas = np.linspace(-sensor_range, sensor_range, 120)
        start = agent_frame_to_image_coords(
            [0, 0, 0], agent_state, sensor_state,
            resolution=self.resolution, focal_length=self.focal_length
        )  # agent的原点在图像坐标系的位置（像素）

        temp_map = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)
        for theta_i in all_thetas:
            r_i, theta_i = self._get_radial_distance(start, theta_i, navigability_mask, agent_state, sensor_state,
                                                     depth_image)
            if r_i is not None:
                self.update_voxel(
                    r_i, theta_i, agent_state, temp_map
                )
        angle = str(int(direction_idx * 30))
        self.panoramic_mask[angle] = np.all(temp_map == WHITE, axis=-1) | np.all(temp_map == GREEN, axis=-1)
        self.effective_mask[angle] = np.all(temp_map == GREEN, axis=-1)

    def _update_voxel(self, r: float, theta: float, agent_state: habitat_sim.AgentState, clip_dist: float, clip_frac: float):
        """Update the voxel map to mark actions as explored or unexplored"""
        agent_coords = self._global_to_grid(agent_state.position)

        # Mark explored regions
        clipped = min(r, clip_dist)
        local_coords = np.array([clipped * np.sin(theta), 0, -clipped * np.cos(theta)])
        global_coords = local_to_global(agent_state.position, agent_state.rotation, local_coords)
        point = self._global_to_grid(global_coords)
        cv2.line(self.explored_map, agent_coords, point, self.explored_color, self.voxel_ray_size)

    def _navigability(self, obs: dict):
        """Generates the set of navigability actions and updates the voxel map accordingly."""
        agent_state: habitat_sim.AgentState = obs['agent_state']
        sensor_state = agent_state.sensor_states['color_sensor']
        rgb_image = obs['color_sensor']
        depth_image = obs[f'depth_sensor']
        if self.cfg['navigability_mode'] == 'depth_estimate':
            depth_image = self.depth_estimator.call(rgb_image)
        if self.cfg['navigability_mode'] == 'segmentation':
            depth_image = None

        navigability_mask = self._get_navigability_mask(
            rgb_image, depth_image, agent_state, sensor_state
        )

        sensor_range =  np.deg2rad(self.fov / 2) * 1.5

        all_thetas = np.linspace(-sensor_range, sensor_range, self.cfg['num_theta'])
        start = agent_frame_to_image_coords(
            [0, 0, 0], agent_state, sensor_state,
            resolution=self.resolution, focal_length=self.focal_length
        )  # agent的原点在图像坐标系的位置（像素）

        a_initial = []
        for theta_i in all_thetas:
            r_i, theta_i = self._get_radial_distance(start, theta_i, navigability_mask, agent_state, sensor_state, depth_image)
            if r_i is not None:
                self._update_voxel(
                    r_i, theta_i, agent_state,
                    clip_dist=self.cfg['max_action_dist'], clip_frac=self.e_i_scaling
                )
                a_initial.append((r_i, theta_i))

        # draw explored circle
        if self.cfg['panoramic_padding'] == True:
            r_max, theta_max = max(a_initial, key=lambda x: x[0])
            self._update_panoramic_voxel(r_max, theta_max, agent_state,
                    clip_dist=self.cfg['max_action_dist'], clip_frac=self.e_i_scaling)
        return a_initial

    def get_spend(self):
        return self.ActionVLM.get_spend()  + self.PlanVLM.get_spend() + self.PredictVLM.get_spend() + self.GoalVLM.get_spend()

    def _prompting(self, goal, a_final: list, images: dict, step_metadata: dict, subtask: str):
        """
        Prompting component of BASEV2. Constructs the textual prompt and calls the action model.
        Parses the response for the chosen action number.
        """
        prompt_type = 'action'
        action_prompt = self._construct_prompt(goal, prompt_type, subtask, num_actions=len(a_final))

        prompt_images = [images['color_sensor']]
        if 'goal_image' in images:
            prompt_images.append(images['goal_image'])

        response = self.ActionVLM.call_chat(prompt_images, action_prompt)

        logging_data = {}
        try:
            response_dict = self._eval_response(response)
            step_metadata['action_number'] = int(response_dict['action'])
        except (IndexError, KeyError, TypeError, ValueError) as e:
            logging.error(f'Error parsing response {e}')
            step_metadata['success'] = 0
        finally:
            logging_data['ACTION_NUMBER'] = step_metadata.get('action_number')
            logging_data['ACTION_PROMPT'] = action_prompt
            logging_data['ACTION_RESPONSE'] = response

        return step_metadata, logging_data, response

    def _goal_module(self, goal_image: np.array, a_goal, goal):
        """Determines if the agent should stop."""
        location_prompt = self._construct_prompt(goal, 'goal', num_actions=len(a_goal))
        location_response = self.GoalVLM.call([goal_image], location_prompt)
        dct = self._eval_response(location_response)

        try:
            number = int(dct['Number'])
        except:
            number = None

        return number, location_response

    def _get_goal_position(self, action_goal, idx, agent_state):
        for key, value in action_goal.items():
            if value == idx:
                r, theta = key
                break

        agent_coords = self._global_to_grid(agent_state.position)

        local_goal = np.array([r * np.sin(theta), 0, -r * np.cos(theta)])
        global_goal = local_to_global(agent_state.position, agent_state.rotation, local_goal)
        point = self._global_to_grid(global_goal)

        # get top down radius
        radius = 1  # real radius (m)
        local_radius = np.array([0, 0, -radius])
        global_radius = local_to_global(agent_state.position, agent_state.rotation, local_radius)
        radius_point = self._global_to_grid(global_radius)
        top_down_radius = int(np.linalg.norm(np.array(agent_coords) - np.array(radius_point)))

        temp_map = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)
        cv2.circle(temp_map, point, top_down_radius, WHITE, -1)
        goal_mask = np.all(temp_map == WHITE, axis=-1)

        return global_goal, goal_mask

    def _choose_action(self, obs: dict):
        agent_state = obs['agent_state']
        goal = obs['goal']

        a_final, images, step_metadata, a_goal, candidate_images = self._run_threads(obs, [obs['color_sensor']], goal)

        goal_image = candidate_images['color_sensor'].copy()
        if a_goal is not None:
            goal_number, location_response = self._goal_module(goal_image, a_goal, goal)
            images['goal_image'] = goal_image
            if goal_number is not None and goal_number != 0:
                goal_position, self.goal_mask = self._get_goal_position(a_goal, goal_number, agent_state)  # get goal position and update goal mask
                self.goal_position.append(goal_position)

        step_metadata['object'] = goal

        # If the model calls stop two times in a row, terminate the episode
        if step_metadata['called_stopping']:
            step_metadata['action_number'] = -1
            agent_action = PolarAction.stop
            logging_data = {}
        else:
            if a_goal is not None and goal_number is not None and goal_number != 0:
                logging_data = {}
                logging_data['ACTION_NUMBER'] = int(goal_number)
                step_metadata['action_number'] = goal_number
                a_final = a_goal
            else:
                step_metadata, logging_data, _ = self._prompting(goal, a_final, images, step_metadata, obs['subtask'])
            agent_action = self._action_number_to_polar(step_metadata['action_number'], list(a_final))
        if a_goal is not None:
            logging_data['LOCATOR_RESPONSE'] = location_response
        metadata = {
            'step_metadata': step_metadata,
            'logging_data': logging_data,
            'a_final': a_final,
            'images': images,
            'step': self.step_ndx
        }
        return agent_action, metadata

    def _eval_response(self, response: str):
        """Converts the VLM response string into a dictionary, if possible"""
        import re
        result = re.sub(r"(?<=[a-zA-Z])'(?=[a-zA-Z])", "\\'", response)
        try:
            eval_resp = ast.literal_eval(result[result.index('{') + 1:result.rindex('}')]) # {{}}
            if isinstance(eval_resp, dict):
                return eval_resp
        except:
            try:
                eval_resp = ast.literal_eval(result[result.rindex('{'):result.rindex('}') + 1]) # {}
                if isinstance(eval_resp, dict):
                    return eval_resp
            except:
                try:
                    eval_resp = ast.literal_eval(result[result.index('{'):result.rindex('}')+1]) # {{}, {}}
                    if isinstance(eval_resp, dict):
                        return eval_resp
                except:
                    logging.error(f'Error parsing response {response}')
                    return {}

    def _planning_module(self, planning_image: list[np.array], previous_subtask, goal_reason: str, goal):
        """Determines if the agent should stop."""
        planning_prompt = self._construct_prompt(goal, 'planning', previous_subtask, goal_reason)
        planning_response = self.PlanVLM.call([planning_image], planning_prompt)
        planning_response = planning_response.replace('false', 'False').replace('true', 'True')
        dct = self._eval_response(planning_response)

        return dct

    def _predicting_module(self, evaluator_image, goal):
        """Determines if the agent should stop."""
        evaluator_prompt = self._construct_prompt(goal, 'predicting')
        evaluator_response = self.PredictVLM.call([evaluator_image], evaluator_prompt)
        dct = self._eval_response(evaluator_response)

        return dct

    @staticmethod
    def _concat_panoramic(images, angles):
        try:
            height, width = images[0].shape[0], images[0].shape[1]
        except:
            height, width = 480, 640
        background_image = np.zeros((2 * height + 3 * 10, 3 * width + 4 * 10, 3), np.uint8)
        copy_images = np.array(images, dtype=np.uint8)
        for i in range(len(copy_images)):
            if i % 2 == 0:
                continue
            copy_images[i] = cv2.putText(copy_images[i], "Angle %d" % angles[i], (100, 100), cv2.FONT_HERSHEY_SIMPLEX,
                                         2, (255, 0, 0), 6, cv2.LINE_AA)
            row = i // 6
            col = (i // 2) % 3
            background_image[10 * (row + 1) + row * height:10 * (row + 1) + row * height + height:,
            10 * (col + 1) + col * width:10 * (col + 1) + col * width + width, :] = copy_images[i]
        return background_image

    def make_curiosity_value(self, pano_images, goal):
        angles = (np.arange(len(pano_images))) * 30
        inference_image = self._concat_panoramic(pano_images, angles)

        response = self._predicting_module(inference_image, goal)

        explorable_value = {}
        reason = {}

        try:
            for angle, values in response.items():
                explorable_value[angle] = values['Score']
                reason[angle] = values['Explanation']
        except:
            explorable_value, reason = None, None

        return inference_image, explorable_value, reason

    @staticmethod
    def _merge_evalue(arr, num):
        return np.minimum(arr, num)

    def update_curiosity_value(self, explorable_value, reason):
        try:
            final_score = {}
            for i in range(12):
                if i % 2 == 0:
                    continue
                last_angle = str(int((i-2)*30)) if i != 1 else '330'
                angle = str(int(i * 30))
                next_angle = str(int((i+2)*30)) if i != 11 else '30'
                if np.all(self.panoramic_mask[angle] == False):
                    continue
                intersection1 = self.effective_mask[last_angle] & self.effective_mask[angle]
                intersection2 = self.effective_mask[angle] & self.effective_mask[next_angle]

                mask_minus_intersection = self.effective_mask[angle] & ~intersection1 & ~intersection2

                self.cvalue_map[mask_minus_intersection] = self._merge_evalue(self.cvalue_map[mask_minus_intersection], explorable_value[angle]) # update explorable value map
                if np.all(intersection2 == False):
                    continue
                self.cvalue_map[intersection2] = self._merge_evalue(self.cvalue_map[intersection2], (explorable_value[angle] + explorable_value[next_angle]) / 2)  # update explorable value map
            if self.goal_mask is not None:
                self.cvalue_map[self.goal_mask] = 10.0
            for i in range(12):
                if i % 2 == 0:
                    continue
                angle = str(int(i * 30))
                if np.all(self.panoramic_mask[angle] == False):
                    final_score[i] = explorable_value[angle]
                else:
                    final_score[i] = np.mean(self.cvalue_map[self.panoramic_mask[angle]])

            idx = max(final_score, key=final_score.get)
            final_reason = reason[str(int(idx * 30))]
        except:
            idx = np.random.randint(0, 12)
            final_reason = ''
        return idx, final_reason

    def draw_cvalue_map(self, agent_state: habitat_sim.AgentState=None, zoom: int=9):
        agent_coords = self._global_to_grid(agent_state.position)
        right = (agent_state.position[0] + zoom, 0, agent_state.position[2])
        right_coords = self._global_to_grid(right)
        delta = abs(agent_coords[0] - right_coords[0])

        evalue_map = (self.evalue_map / 10 * 255).astype(np.uint8)


        x, y = agent_coords
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Zoom the map
        max_x, max_y = evalue_map.shape[1], evalue_map.shape[0]
        x1 = max(0, x - delta)
        x2 = min(max_x, x + delta)
        y1 = max(0, y - delta)
        y2 = min(max_y, y + delta)

        zoomed_map = evalue_map[y1:y2, x1:x2]

        if self.step_ndx is not None:
            step_text = f'step {self.step_ndx}'
            cv2.putText(zoomed_map, step_text, (30, 90), font, 3, (0, 0, 0), 2, cv2.LINE_AA)

        return zoomed_map

    def make_plan(self, pano_images, previous_subtask, goal_reason, goal):
        response = self._planning_module(pano_images, previous_subtask, goal_reason, goal)

        try:
            goal_flag, subtask = response['Flag'], response['Subtask']
        except:
            print("planning failed!")
            print('response:', response)
            goal_flag, subtask = False, '{}'

        return goal_flag, subtask

    def _construct_prompt(self, goal: str, prompt_type:str, subtask: str='{}', reason: str='{}', num_actions: int=0):
        if prompt_type == 'goal':
            location_prompt = (f"The agent has been tasked with navigating to a {goal.upper()}. The agent has sent you an image taken from its current location. "
            f"There are {num_actions} red arrows superimposed onto your observation, which represent potential positions. " 
            f"These are labeled with a number in a white circle, which represent the location you can move to. "
            f"First, tell me whether the {goal} is in the image, and make sure the object you see is ACTUALLY a {goal}, return number 0 if if there is no {goal}, or if you are not sure. Note a chair must have a backrest and a chair is not a stool. Note a chair is NOT sofa(couch) which is NOT a bed. "
            f'Second, if there is {goal} in the image, then determine which circle best represents the location of the {goal}(close enough to the target. If a person is standing in that position, they can easily touch the {goal}), and give the number and a reason. '
            f'If none of the circles represent the position of the {goal}, return number 0, and give a reason why you returned 0. '
            "Format your answer in the json {{'Number': <The number you choose>}}")
            return location_prompt
        if prompt_type == 'predicting':
            evaluator_prompt = (f"The agent has been tasked with navigating to a {goal.upper()}. The agent has sent you the panoramic image describing your surrounding environment, each image contains a label indicating the relative rotation angle(30, 90, 150, 210, 270, 330) with red fonts. "
            f'Your job is to assign a score to each direction (ranging from 0 to 10), judging whether this direction is worth exploring. The following criteria should be used: '
            f'To help you describe the layout of your surrounding,  please follow my step-by-step instructions: '
            f'(1) If there is no visible way to move to other areas and it is clear that the target is not in sight, assign a score of 0. Note a chair must have a backrest and a chair is not a stool. Note a chair is NOT sofa(couch) which is NOT a bed. '
            f'(2) If the {goal} is found, assign a score of 10.  ' 
            f'(3) If there is a way to move to another area, assign a score based on your estimate of the likelihood of finding a {goal}, using your common sense. Moving to another area means there is a turn in the corner, an open door, a hallway, etc. Note you CANNOT GO THROUGH CLOSED DOORS. CLOSED DOORS and GOING UP OR DOWN STAIRS are not considered. '
            "For each direction, provide an explanation for your assigned score. Format your answer in the json {'30': {'Score': <The score(from 0 to 10) of angle 30>, 'Explanation': <An explanation for your assigned score.>}, '90': {...}, '150': {...}, '210': {...}, '270': {...}, '330': {...}}. "
            "Answer Example: {'30': {'Score': 0, 'Explanation': 'Dead end with a recliner. No sign of a bed or any other room.'}, '90': {'Score': 2, 'Explanation': 'Dining area. It is possible there is a doorway leading to other rooms, but bedrooms are less likely to be directly adjacent to dining areas.'}, ..., '330': {'Score': 2, 'Explanation': 'Living room area with a recliner.  Similar to 270, there is a possibility of other rooms, but no strong indication of a bedroom.'}}")
            return evaluator_prompt
        if prompt_type == 'planning':
            if reason != '' and subtask != '{}':
                planning_prompt = (f"The agent has been tasked with navigating to a {goal.upper()}. The agent has sent you the following elements:"
                f"(1)<The observed image>: The image taken from its current location. "
                f"(2){reason}. This explains why you should go in this direction. "
                f'Your job is to describe next place to go. '
                f'To help you plan your best next step, I can give you some human suggestions:. '
                f'(1) If the {goal} appears in the image, directly choose the target as the next step in the plan. Note a chair must have a backrest and a chair is not a stool. Note a chair is NOT sofa(couch) which is NOT a bed. '
                f'(2) If the {goal} is not found and the previous subtask {subtask} has not completed, continue to complete the last subtask {subtask} that has not been completed.'
                f'(3) If the {goal} is not found and the previous subtask {subtask} has already been completed. Identify a new subtask by describing where you are going next to be more likely to find clues to the the {goal} and think about whether the {goal} is likely to occur in that direction. Note you need to pay special attention to open doors and hallways, as they can lead to other unseen rooms. Note GOING UP OR DOWN STAIRS is an option. '
                "Format your answer in the json {{'Subtask': <Where you are going next>, 'Flag': <Whether the target is in your view, True or False>}}. "
                "Answer Example: {{'Subtask': 'Go to the hallway', 'Flag': False}} or {{'Subtask': "+f"'Go to the {goal}'"+", 'Flag': True}} or {{'Subtask': 'Go to the open door', 'Flag': True}}")
            else:
                planning_prompt = (f"The agent has been tasked with navigating to a {goal.upper()}. The agent has sent you an image taken from its current location."
                f'Your job is to describe next place to go. '
                f'To help you plan your best next step, I can give you some human suggestions:. '
                f'(1) If the {goal} appears in the image, directly choose the target as the next step in the plan. Note a chair must have a backrest and a chair is not a stool. Note a chair is NOT sofa(couch) which is NOT a bed. '
                f'(2) If the {goal} is not found, describe where you are going next to be more likely to find clues to the the {goal} and analyze the room type and think about whether the {goal} is likely to occur in that direction. Note you need to pay special attention to open doors and hallways, as they can lead to other unseen rooms. Note GOING UP OR DOWN STAIRS is an option. '
                "Format your answer in the json {{'Subtask': <Where you are going next>, 'Flag': <Whether the target is in your view, True or False>}}. "
                "Answer Example: {{'Subtask': 'Go to the hallway', 'Flag': False}} or {{'Subtask': "+f"'Go to the {goal}'"+", 'Flag': True}} or {{'Subtask': 'Go to the open door', 'Flag': True}}")
            return planning_prompt
        if prompt_type == 'action':
            if subtask != '{}':
                action_prompt = (
                f"TASK: {subtask}. Your final task is to NAVIGATE TO THE NEAREST {goal.upper()}, and get as close to it as possible. "
                f"There are {num_actions - 1} red arrows superimposed onto your observation, which represent potential actions. " 
                f"These are labeled with a number in a white circle, which represent the location you would move to if you took that action. {'NOTE: choose action 0 if you want to TURN AROUND or DONT SEE ANY GOOD ACTIONS. ' if self.step_ndx - self.turned >= self.cfg['turn_around_cooldown'] else ''}"
                f"In order to complete the subtask {subtask} and eventually the final task NAVIGATING TO THE NEAREST {goal.upper()}. Explain which action acheives that best. "
                "Return your answer as {{'action': <action_key>}}. Note you CANNOT GO THROUGH CLOSED DOORS, and you DO NOT NEED TO GO UP OR DOWN STAIRS"
                )
            else:
                action_prompt = (
                    f"TASK: NAVIGATE TO THE NEAREST {goal.upper()}, and get as close to it as possible. Use your prior knowledge about where items are typically located within a home. "
                    f"There are {num_actions - 1} red arrows superimposed onto your observation, which represent potential actions. "
                    f"These are labeled with a number in a white circle, which represent the location you would move to if you took that action. {'NOTE: choose action 0 if you want to TURN AROUND or DONT SEE ANY GOOD ACTIONS. ' if self.step_ndx - self.turned >= self.cfg['turn_around_cooldown'] else ''}"
                    f"First, tell me what you see in your sensor observation, and if you have any leads on finding the {goal.upper()}. Second, tell me which general direction you should go in. "
                    "Lastly, explain which action acheives that best, and return it as {{'action': <action_key>}}. Note you CANNOT GO THROUGH CLOSED DOORS, and you DO NOT NEED TO GO UP OR DOWN STAIRS"
                )
            return action_prompt

        raise ValueError('Prompt type must be goal, predicting, planning, or action')

class CustomAgent(VLMNavAgent):
    def reset(self):
        self.voxel_map = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)
        self.explored_map = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)
        self.cvalue_map = 10 * np.ones((self.map_size, self.map_size, 3), dtype=np.float16)
        self.goal_position = []
        self.goal_mask = None
        self.panoramic_mask = {}
        self.effective_mask = {}
        self.stopping_calls = [-2]
        self.step_ndx = 0
        self.init_pos = None
        self.turned = -self.cfg['turn_around_cooldown']
        self.ActionVLM.reset()
        self.PlanVLM.reset()
        self.PredictVLM.reset()
        self.GoalVLM.reset()

    def _initialize_vlms(self, cfg: dict):
        vlm_cls = globals()[cfg['model_cls']]
        system_instruction = (
            "You are an embodied robotic assistant, with an RGB image sensor. You observe the image and instructions "
            "given to you and output a textual response, which is converted into actions that physically move you "
            "within the environment. You cannot move through closed doors. "
        )
        self.ActionVLM: VLM = vlm_cls(**cfg['model_kwargs'], system_instruction=system_instruction)
        self.PlanVLM: VLM = vlm_cls(**cfg['model_kwargs'])
        self.GoalVLM: VLM = vlm_cls(**cfg['model_kwargs'])
        self.PredictVLM: VLM = vlm_cls(**cfg['model_kwargs'])

    def _update_panoramic_voxel(self, r: float, theta: float, agent_state: habitat_sim.AgentState, clip_dist: float, clip_frac: float):
        """Update the voxel map to mark actions as explored"""
        agent_coords = self._global_to_grid(agent_state.position)

        # Mark circle explored regions
        clipped = min(clip_frac * r, clip_dist)
        local_coords = np.array([clipped * np.sin(theta), 0, -clipped * np.cos(theta)])
        global_coords = local_to_global(agent_state.position, agent_state.rotation, local_coords)
        point = self._global_to_grid(global_coords)

        radius = int(np.linalg.norm(np.array(agent_coords) - np.array(point)))
        cv2.circle(self.explored_map, agent_coords, radius, self.explored_color, -1)  # -1表示实心圆

    def _stopping_module(self, obs, threshold_dist=0.8):
        if self.goal_position:
            arr = np.array(self.goal_position)
            # 按列计算平均值
            avg_goal_position = np.mean(arr, axis=0)
            agent_state = obs['agent_state']
            current_position = np.array([agent_state.position[0], agent_state.position[2]])
            goal_position = np.array([avg_goal_position[0], avg_goal_position[2]])
            dist = np.linalg.norm(current_position - goal_position)

            if dist < threshold_dist:
                return True
        return False


    def _run_threads(self, obs: dict, stopping_images: list[np.array], goal):
        """Concurrently runs the stopping thread to determine if the agent should stop, and the preprocessing thread to calculate potential actions."""
        called_stop = False
        with concurrent.futures.ThreadPoolExecutor() as executor:
            preprocessing_thread = executor.submit(self._preprocessing_module, obs)
            stopping_thread = executor.submit(self._stopping_module, obs)

            a_final, images, a_goal, candidate_images = preprocessing_thread.result()
            called_stop = stopping_thread.result()

        if called_stop:
            logging.info('Model called stop')
            self.stopping_calls.append(self.step_ndx)
            # If the model calls stop, turn off navigability and explore bias tricks
            if self.cfg['navigability_mode'] != 'none':
                new_image = obs['color_sensor'].copy()
                a_final = self._project_onto_image(
                    self._get_default_arrows(), new_image, obs['agent_state'],
                    obs['agent_state'].sensor_states['color_sensor'],
                    step=self.step_ndx,
                    goal=obs['goal']
                )
                images['color_sensor'] = new_image

        step_metadata = {
            'action_number': -10,
            'success': 1,
            'model': self.ActionVLM.name,
            'agent_location': obs['agent_state'].position,
            'called_stopping': called_stop
        }
        return a_final, images, step_metadata, a_goal, candidate_images

    def _draw_direction_arrow(self, roomtrack_map, direction_vector, position, coords, angle_text='', arrow_length=1):
        # 箭头的终点
        arrow_end = np.array(
            [position[0] + direction_vector[0] * arrow_length, position[1],
             position[2] + direction_vector[2] * arrow_length])  # 假设 y 轴是高度轴，不变

        # 将世界坐标转换为网格坐标
        arrow_end_coords = self._global_to_grid(arrow_end)

        # 绘制箭头
        cv2.arrowedLine(roomtrack_map, (coords[0], coords[1]),
                        (arrow_end_coords[0], arrow_end_coords[1]), WHITE, 4, tipLength=0.1)

        # 绘制文本，表示角度（假设为 30°，你可以根据实际情况调整）
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = 1
        text_thickness = 2

        # 获取文本的宽度和高度，用来居中文本
        (text_width, text_height), _ = cv2.getTextSize(angle_text, font, text_size, text_thickness)

        # 设置文本的位置为箭头终点稍微偏移
        text_end_coords = self._global_to_grid(np.array(
            [position[0] + direction_vector[0] * arrow_length * 1.5, position[1],
             position[2] + direction_vector[2] * arrow_length * 1.5]))
        text_position = (text_end_coords[0] - text_width // 2, text_end_coords[1] + text_height // 2)

        # 绘制文本
        cv2.putText(roomtrack_map, angle_text, text_position, font, text_size, (255, 255, 255), text_thickness,
                    cv2.LINE_AA)

    def generate_voxel(self, agent_state: habitat_sim.AgentState=None, zoom: int=9):
        """For visualization purposes, add the agent's position and actions onto the voxel map"""
        agent_coords = self._global_to_grid(agent_state.position)
        right = (agent_state.position[0] + zoom, 0, agent_state.position[2])
        right_coords = self._global_to_grid(right)
        delta = abs(agent_coords[0] - right_coords[0])

        topdown_map = self.voxel_map.copy()
        mask = np.all(self.explored_map == self.explored_color, axis=-1)
        topdown_map[mask] = self.explored_color

        # direction vector
        direction_vector = habitat_sim.utils.quat_rotate_vector(agent_state.rotation, habitat_sim.geo.FRONT)

        self._draw_direction_arrow(topdown_map, direction_vector, agent_state.position, agent_coords,
                                   angle_text="0")
        theta_60 = -np.pi / 3
        theta_30 = -np.pi / 6
        y_axis = np.array([0, 1, 0])
        quat_60 = habitat_sim.utils.quat_from_angle_axis(theta_60, y_axis)
        quat_30 = habitat_sim.utils.quat_from_angle_axis(theta_30, y_axis)
        direction_30_vector = habitat_sim.utils.quat_rotate_vector(quat_30, direction_vector)
        self._draw_direction_arrow(topdown_map, direction_30_vector, agent_state.position, agent_coords,
                                   angle_text="30")
        direction_60_vector = direction_30_vector.copy()
        for i in range(5):
            direction_60_vector = habitat_sim.utils.quat_rotate_vector(quat_60, direction_60_vector)
            angle = (i + 1) * 60 + 30
            self._draw_direction_arrow(topdown_map, direction_60_vector, agent_state.position, agent_coords,
                                       angle_text=str(angle))

        text_size = 1.25
        text_thickness = 1
        x, y = agent_coords
        font = cv2.FONT_HERSHEY_SIMPLEX

        text = str(self.step_ndx)
        (text_width, text_height), _ = cv2.getTextSize(text, font, text_size, text_thickness)
        circle_center = (agent_coords[0], agent_coords[1])
        circle_radius = max(text_width, text_height) // 2 + 15

        cv2.circle(topdown_map, circle_center, circle_radius, WHITE, -1)

        text_position = (circle_center[0] - text_width // 2, circle_center[1] + text_height // 2)
        cv2.circle(topdown_map, circle_center, circle_radius, RED, 1)
        cv2.putText(topdown_map, text, text_position, font, text_size, RED, text_thickness + 1)


        # Zoom the map
        max_x, max_y = topdown_map.shape[1], topdown_map.shape[0]
        x1 = max(0, x - delta)
        x2 = min(max_x, x + delta)
        y1 = max(0, y - delta)
        y2 = min(max_y, y + delta)

        zoomed_map = topdown_map[y1:y2, x1:x2]

        if self.step_ndx is not None:
            step_text = f'step {self.step_ndx}'
            cv2.putText(zoomed_map, step_text, (30, 90), font, 3, (255, 255, 255), 2, cv2.LINE_AA)

        return zoomed_map

    def update_voxel(self, r: float, theta: float, agent_state: habitat_sim.AgentState, temp_map: np.ndarray, effective_dist: float=3):
        agent_coords = self._global_to_grid(agent_state.position)

        # Mark unexplored regions
        unclipped = max(r, 0)
        local_coords = np.array([unclipped * np.sin(theta), 0, -unclipped * np.cos(theta)])
        global_coords = local_to_global(agent_state.position, agent_state.rotation, local_coords)
        point = self._global_to_grid(global_coords)
        cv2.line(self.voxel_map, agent_coords, point, self.unexplored_color, 40)

        # Mark directional regions
        cv2.line(temp_map, agent_coords, point, WHITE, 40) # whole area
        unclipped = min(r, effective_dist)
        local_coords = np.array([unclipped * np.sin(theta), 0, -unclipped * np.cos(theta)])
        global_coords = local_to_global(agent_state.position, agent_state.rotation, local_coords)
        point = self._global_to_grid(global_coords)
        cv2.line(temp_map, agent_coords, point, GREEN, 40) # effective area

    def _goal_proposer(self, a_initial: list, agent_state: habitat_sim.AgentState):
        min_angle = self.fov / self.cfg['spacing_ratio']

        unique = {}
        for mag, theta in a_initial:
            if theta in unique:
                unique[theta].append(mag)
            else:
                unique[theta] = [mag]

        arrowData = []

        for theta, mags in unique.items():
            mag = min(mags)
            arrowData.append([mag, theta])

        arrowData.sort(key=lambda x: x[1])
        thetas = set()
        out = []
        filter_thresh = 0
        f = list(filter(lambda x: x[0] > filter_thresh, arrowData))

        f.sort(key=lambda x: x[1])
        if f == []:
            return []
        # Add unexplored actions with spacing, starting with the longest one
        if len(f) > 0:
            longest = max(f, key=lambda x: x[0])
            longest_theta = longest[1]
            smallest_theta = longest[1]
            longest_ndx = f.index(longest)

            out.append([longest[0], longest[1]])
            thetas.add(longest[1])
            for i in range(longest_ndx + 1, len(f)):
                if f[i][1] - longest_theta > (min_angle * 0.45):
                    out.append([f[i][0], f[i][1]])
                    thetas.add(f[i][1])
                    longest_theta = f[i][1]
            for i in range(longest_ndx - 1, -1, -1):
                if smallest_theta - f[i][1] > (min_angle * 0.45):
                    out.append([f[i][0], f[i][1]])
                    thetas.add(f[i][1])
                    smallest_theta = f[i][1]

        out.sort(key=lambda x: x[1])
        return [(mag, theta) for mag, theta in out]

    def _preprocessing_module(self, obs: dict):
        """Excutes the navigability, action_proposer and projection submodules."""
        agent_state = obs['agent_state']
        images = {'color_sensor': obs['color_sensor'].copy()}
        candidate_images = {'color_sensor': obs['color_sensor'].copy()}
        a_goal_projected = None

        if self.cfg['navigability_mode'] == 'none':
            a_final = [
                # Actions for the w/o nav baseline
                (self.cfg['max_action_dist'], -0.36 * np.pi),
                (self.cfg['max_action_dist'], -0.28 * np.pi),
                (self.cfg['max_action_dist'], 0),
                (self.cfg['max_action_dist'], 0.28 * np.pi),
                (self.cfg['max_action_dist'], 0.36 * np.pi)
            ]
        else:
            a_initial = self._navigability(obs)
            a_final = self._action_proposer(a_initial, agent_state)
            if obs['goal_flag']:
                a_goal = self._goal_proposer(a_initial, agent_state)

        a_final_projected = self._projection(a_final, images, agent_state, obs['goal'])
        if obs['goal_flag']:
            a_goal_projected = self._projection(a_goal, candidate_images, agent_state, obs['goal'], candidate_flag=True)
        images['voxel_map'] = self._generate_voxel(a_final_projected, agent_state=agent_state, step=self.step_ndx)
        return a_final_projected, images, a_goal_projected, candidate_images

    def navigability(self, obs: dict, direction_idx: int):
        """Generates the set of navigability actions and updates the voxel map accordingly."""
        agent_state: habitat_sim.AgentState = obs['agent_state']
        if self.step_ndx == 0:
            self.init_pos = agent_state.position
        sensor_state = agent_state.sensor_states['color_sensor']
        rgb_image = obs['color_sensor']
        depth_image = obs[f'depth_sensor']
        if self.cfg['navigability_mode'] == 'depth_estimate':
            depth_image = self.depth_estimator.call(rgb_image)
        if self.cfg['navigability_mode'] == 'segmentation':
            depth_image = None

        navigability_mask = self._get_navigability_mask(
            rgb_image, depth_image, agent_state, sensor_state
        )

        sensor_range = np.deg2rad(self.fov / 2) * 1.5

        all_thetas = np.linspace(-sensor_range, sensor_range, 120)
        start = agent_frame_to_image_coords(
            [0, 0, 0], agent_state, sensor_state,
            resolution=self.resolution, focal_length=self.focal_length
        )  # agent的原点在图像坐标系的位置（像素）

        temp_map = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)
        for theta_i in all_thetas:
            r_i, theta_i = self._get_radial_distance(start, theta_i, navigability_mask, agent_state, sensor_state,
                                                     depth_image)
            if r_i is not None:
                self.update_voxel(
                    r_i, theta_i, agent_state, temp_map
                )
        angle = str(int(direction_idx * 30))
        self.panoramic_mask[angle] = np.all(temp_map == WHITE, axis=-1) | np.all(temp_map == GREEN, axis=-1)
        self.effective_mask[angle] = np.all(temp_map == GREEN, axis=-1)

    def _update_voxel(self, r: float, theta: float, agent_state: habitat_sim.AgentState, clip_dist: float, clip_frac: float):
        """Update the voxel map to mark actions as explored or unexplored"""
        agent_coords = self._global_to_grid(agent_state.position)

        # Mark explored regions
        clipped = min(r, clip_dist)
        local_coords = np.array([clipped * np.sin(theta), 0, -clipped * np.cos(theta)])
        global_coords = local_to_global(agent_state.position, agent_state.rotation, local_coords)
        point = self._global_to_grid(global_coords)
        cv2.line(self.explored_map, agent_coords, point, self.explored_color, self.voxel_ray_size)

    def _navigability(self, obs: dict):
        """Generates the set of navigability actions and updates the voxel map accordingly."""
        agent_state: habitat_sim.AgentState = obs['agent_state']
        sensor_state = agent_state.sensor_states['color_sensor']
        rgb_image = obs['color_sensor']
        depth_image = obs[f'depth_sensor']
        if self.cfg['navigability_mode'] == 'depth_estimate':
            depth_image = self.depth_estimator.call(rgb_image)
        if self.cfg['navigability_mode'] == 'segmentation':
            depth_image = None

        navigability_mask = self._get_navigability_mask(
            rgb_image, depth_image, agent_state, sensor_state
        )

        sensor_range =  np.deg2rad(self.fov / 2) * 1.5

        all_thetas = np.linspace(-sensor_range, sensor_range, self.cfg['num_theta'])
        start = agent_frame_to_image_coords(
            [0, 0, 0], agent_state, sensor_state,
            resolution=self.resolution, focal_length=self.focal_length
        )  # agent的原点在图像坐标系的位置（像素）

        a_initial = []
        for theta_i in all_thetas:
            r_i, theta_i = self._get_radial_distance(start, theta_i, navigability_mask, agent_state, sensor_state, depth_image)
            if r_i is not None:
                self._update_voxel(
                    r_i, theta_i, agent_state,
                    clip_dist=self.cfg['max_action_dist'], clip_frac=self.e_i_scaling
                )
                a_initial.append((r_i, theta_i))

        # draw explored circle
        if self.cfg['panoramic_padding'] == True:
            r_max, theta_max = max(a_initial, key=lambda x: x[0])
            self._update_panoramic_voxel(r_max, theta_max, agent_state,
                    clip_dist=self.cfg['max_action_dist'], clip_frac=self.e_i_scaling)
        return a_initial

    def get_spend(self):
        return self.ActionVLM.get_spend()  + self.PlanVLM.get_spend() + self.PredictVLM.get_spend() + self.GoalVLM.get_spend()

    def _prompting(self, goal, a_final: list, images: dict, step_metadata: dict, subtask: str):
        """
        Prompting component of BASEV2. Constructs the textual prompt and calls the action model.
        Parses the response for the chosen action number.
        """
        prompt_type = 'action'
        action_prompt = self._construct_prompt(goal, prompt_type, subtask, num_actions=len(a_final))

        prompt_images = [images['color_sensor']]
        if 'goal_image' in images:
            prompt_images.append(images['goal_image'])

        response = self.ActionVLM.call_chat(prompt_images, action_prompt)

        logging_data = {}
        try:
            response_dict = self._eval_response(response)
            step_metadata['action_number'] = int(response_dict['action'])
        except (IndexError, KeyError, TypeError, ValueError) as e:
            logging.error(f'Error parsing response {e}')
            step_metadata['success'] = 0
        finally:
            logging_data['ACTION_NUMBER'] = step_metadata.get('action_number')
            logging_data['ACTION_PROMPT'] = action_prompt
            logging_data['ACTION_RESPONSE'] = response

        return step_metadata, logging_data, response

    def _goal_module(self, goal_image: np.array, a_goal, goal):
        """Determines if the agent should stop."""
        location_prompt = self._construct_prompt(goal, 'goal', num_actions=len(a_goal))
        location_response = self.GoalVLM.call([goal_image], location_prompt)
        dct = self._eval_response(location_response)

        try:
            number = int(dct['Number'])
        except:
            number = None

        return number, location_response

    def _get_goal_position(self, action_goal, idx, agent_state):
        for key, value in action_goal.items():
            if value == idx:
                r, theta = key
                break

        agent_coords = self._global_to_grid(agent_state.position)

        local_goal = np.array([r * np.sin(theta), 0, -r * np.cos(theta)])
        global_goal = local_to_global(agent_state.position, agent_state.rotation, local_goal)
        point = self._global_to_grid(global_goal)

        # get top down radius
        radius = 1  # real radius (m)
        local_radius = np.array([0, 0, -radius])
        global_radius = local_to_global(agent_state.position, agent_state.rotation, local_radius)
        radius_point = self._global_to_grid(global_radius)
        top_down_radius = int(np.linalg.norm(np.array(agent_coords) - np.array(radius_point)))

        temp_map = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)
        cv2.circle(temp_map, point, top_down_radius, WHITE, -1)
        goal_mask = np.all(temp_map == WHITE, axis=-1)

        return global_goal, goal_mask

    def _choose_action(self, obs: dict):
        agent_state = obs['agent_state']
        goal = obs['goal']

        a_final, images, step_metadata, a_goal, candidate_images = self._run_threads(obs, [obs['color_sensor']], goal)

        goal_image = candidate_images['color_sensor'].copy()
        if a_goal is not None:
            goal_number, location_response = self._goal_module(goal_image, a_goal, goal)
            images['goal_image'] = goal_image
            if goal_number is not None and goal_number != 0:
                goal_position, self.goal_mask = self._get_goal_position(a_goal, goal_number, agent_state)  # get goal position and update goal mask
                self.goal_position.append(goal_position)

        step_metadata['object'] = goal

        # If the model calls stop two times in a row, terminate the episode
        if step_metadata['called_stopping']:
            step_metadata['action_number'] = -1
            agent_action = PolarAction.stop
            logging_data = {}
        else:
            if a_goal is not None and goal_number is not None and goal_number != 0:
                logging_data = {}
                logging_data['ACTION_NUMBER'] = int(goal_number)
                step_metadata['action_number'] = goal_number
                a_final = a_goal
            else:
                step_metadata, logging_data, _ = self._prompting(goal, a_final, images, step_metadata, obs['subtask'])
            agent_action = self._action_number_to_polar(step_metadata['action_number'], list(a_final))
        if a_goal is not None:
            logging_data['LOCATOR_RESPONSE'] = location_response
        metadata = {
            'step_metadata': step_metadata,
            'logging_data': logging_data,
            'a_final': a_final,
            'images': images,
            'step': self.step_ndx
        }
        return agent_action, metadata

    def _eval_response(self, response: str):
        """Converts the VLM response string into a dictionary, if possible"""
        import re
        result = re.sub(r"(?<=[a-zA-Z])'(?=[a-zA-Z])", "\\'", response)
        try:
            eval_resp = ast.literal_eval(result[result.index('{') + 1:result.rindex('}')]) # {{}}
            if isinstance(eval_resp, dict):
                return eval_resp
        except:
            try:
                eval_resp = ast.literal_eval(result[result.rindex('{'):result.rindex('}') + 1]) # {}
                if isinstance(eval_resp, dict):
                    return eval_resp
            except:
                try:
                    eval_resp = ast.literal_eval(result[result.index('{'):result.rindex('}')+1]) # {{}, {}}
                    if isinstance(eval_resp, dict):
                        return eval_resp
                except:
                    logging.error(f'Error parsing response {response}')
                    return {}

    def _planning_module(self, planning_image: list[np.array], previous_subtask, goal_reason: str, goal):
        """Determines if the agent should stop."""
        planning_prompt = self._construct_prompt(goal, 'planning', previous_subtask, goal_reason)
        planning_response = self.PlanVLM.call([planning_image], planning_prompt)
        planning_response = planning_response.replace('false', 'False').replace('true', 'True')
        dct = self._eval_response(planning_response)

        return dct

    def _predicting_module(self, evaluator_image, goal):
        """Determines if the agent should stop."""
        evaluator_prompt = self._construct_prompt(goal, 'predicting')
        evaluator_response = self.PredictVLM.call([evaluator_image], evaluator_prompt)
        dct = self._eval_response(evaluator_response)

        return dct

    @staticmethod
    def _concat_panoramic(images, angles):
        try:
            height, width = images[0].shape[0], images[0].shape[1]
        except:
            height, width = 480, 640
        background_image = np.zeros((2 * height + 3 * 10, 3 * width + 4 * 10, 3), np.uint8)
        copy_images = np.array(images, dtype=np.uint8)
        for i in range(len(copy_images)):
            if i % 2 == 0:
                continue
            copy_images[i] = cv2.putText(copy_images[i], "Angle %d" % angles[i], (100, 100), cv2.FONT_HERSHEY_SIMPLEX,
                                         2, (255, 0, 0), 6, cv2.LINE_AA)
            row = i // 6
            col = (i // 2) % 3
            background_image[10 * (row + 1) + row * height:10 * (row + 1) + row * height + height:,
            10 * (col + 1) + col * width:10 * (col + 1) + col * width + width, :] = copy_images[i]
        return background_image

    def make_curiosity_value(self, pano_images, goal):
        angles = (np.arange(len(pano_images))) * 30
        inference_image = self._concat_panoramic(pano_images, angles)

        response = self._predicting_module(inference_image, goal)

        explorable_value = {}
        reason = {}

        try:
            for angle, values in response.items():
                explorable_value[angle] = values['Score']
                reason[angle] = values['Explanation']
        except:
            explorable_value, reason = None, None

        return inference_image, explorable_value, reason

    @staticmethod
    def _merge_evalue(arr, num):
        return np.minimum(arr, num)

    def update_curiosity_value(self, explorable_value, reason):
        try:
            final_score = {}
            for i in range(12):
                if i % 2 == 0:
                    continue
                last_angle = str(int((i-2)*30)) if i != 1 else '330'
                angle = str(int(i * 30))
                next_angle = str(int((i+2)*30)) if i != 11 else '30'
                if np.all(self.panoramic_mask[angle] == False):
                    continue
                intersection1 = self.effective_mask[last_angle] & self.effective_mask[angle]
                intersection2 = self.effective_mask[angle] & self.effective_mask[next_angle]

                mask_minus_intersection = self.effective_mask[angle] & ~intersection1 & ~intersection2

                self.cvalue_map[mask_minus_intersection] = self._merge_evalue(self.cvalue_map[mask_minus_intersection], explorable_value[angle]) # update explorable value map
                if np.all(intersection2 == False):
                    continue
                self.cvalue_map[intersection2] = self._merge_evalue(self.cvalue_map[intersection2], (explorable_value[angle] + explorable_value[next_angle]) / 2)  # update explorable value map
            if self.goal_mask is not None:
                self.cvalue_map[self.goal_mask] = 10.0
            for i in range(12):
                if i % 2 == 0:
                    continue
                angle = str(int(i * 30))
                if np.all(self.panoramic_mask[angle] == False):
                    final_score[i] = explorable_value[angle]
                else:
                    final_score[i] = np.mean(self.cvalue_map[self.panoramic_mask[angle]])

            idx = max(final_score, key=final_score.get)
            final_reason = reason[str(int(idx * 30))]
        except:
            idx = np.random.randint(0, 12)
            final_reason = ''
        return idx, final_reason

    def draw_cvalue_map(self, agent_state: habitat_sim.AgentState=None, zoom: int=9):
        agent_coords = self._global_to_grid(agent_state.position)
        right = (agent_state.position[0] + zoom, 0, agent_state.position[2])
        right_coords = self._global_to_grid(right)
        delta = abs(agent_coords[0] - right_coords[0])

        cvalue_map = (self.cvalue_map / 10 * 255).astype(np.uint8)


        x, y = agent_coords
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Zoom the map
        max_x, max_y = cvalue_map.shape[1], cvalue_map.shape[0]
        x1 = max(0, x - delta)
        x2 = min(max_x, x + delta)
        y1 = max(0, y - delta)
        y2 = min(max_y, y + delta)

        zoomed_map = cvalue_map[y1:y2, x1:x2]

        if self.step_ndx is not None:
            step_text = f'step {self.step_ndx}'
            cv2.putText(zoomed_map, step_text, (30, 90), font, 3, (0, 0, 0), 2, cv2.LINE_AA)

        return zoomed_map

    def make_plan(self, pano_images, previous_subtask, goal_reason, goal):
        response = self._planning_module(pano_images, previous_subtask, goal_reason, goal)

        try:
            goal_flag, subtask = response['Flag'], response['Subtask']
        except:
            print("planning failed!")
            print('response:', response)
            goal_flag, subtask = False, '{}'

        return goal_flag, subtask

    def _construct_prompt(self, goal: str, prompt_type:str, subtask: str='{}', reason: str='{}', num_actions: int=0):
        if prompt_type == 'goal':
            location_prompt = (f"The agent has been tasked with navigating to a {goal.upper()}. The agent has sent you an image taken from its current location. "
            f"There are {num_actions} red arrows superimposed onto your observation, which represent potential positions. " 
            f"These are labeled with a number in a white circle, which represent the location you can move to. "
            f"First, tell me whether the {goal} is in the image, and make sure the object you see is ACTUALLY a {goal}, return number 0 if if there is no {goal}, or if you are not sure. Note a chair must have a backrest and a chair is not a stool. Note a chair is NOT sofa(couch) which is NOT a bed. "
            f'Second, if there is {goal} in the image, then determine which circle best represents the location of the {goal}(close enough to the target. If a person is standing in that position, they can easily touch the {goal}), and give the number and a reason. '
            f'If none of the circles represent the position of the {goal}, return number 0, and give a reason why you returned 0. '
            "Format your answer in the json {{'Number': <The number you choose>}}")
            return location_prompt
        if prompt_type == 'predicting':
            evaluator_prompt = (f"The agent has been tasked with navigating to a {goal.upper()}. The agent has sent you the panoramic image describing your surrounding environment, each image contains a label indicating the relative rotation angle(30, 90, 150, 210, 270, 330) with red fonts. "
            f'Your job is to assign a score to each direction (ranging from 0 to 10), judging whether this direction is worth exploring. The following criteria should be used: '
            f'To help you describe the layout of your surrounding,  please follow my step-by-step instructions: '
            f'(1) If there is no visible way to move to other areas and it is clear that the target is not in sight, assign a score of 0. Note a chair must have a backrest and a chair is not a stool. Note a chair is NOT sofa(couch) which is NOT a bed. '
            f'(2) If the {goal} is found, assign a score of 10.  ' 
            f'(3) If there is a way to move to another area, assign a score based on your estimate of the likelihood of finding a {goal}, using your common sense. Moving to another area means there is a turn in the corner, an open door, a hallway, etc. Note you CANNOT GO THROUGH CLOSED DOORS. CLOSED DOORS and GOING UP OR DOWN STAIRS are not considered. '
            "For each direction, provide an explanation for your assigned score. Format your answer in the json {'30': {'Score': <The score(from 0 to 10) of angle 30>, 'Explanation': <An explanation for your assigned score.>}, '90': {...}, '150': {...}, '210': {...}, '270': {...}, '330': {...}}. "
            "Answer Example: {'30': {'Score': 0, 'Explanation': 'Dead end with a recliner. No sign of a bed or any other room.'}, '90': {'Score': 2, 'Explanation': 'Dining area. It is possible there is a doorway leading to other rooms, but bedrooms are less likely to be directly adjacent to dining areas.'}, ..., '330': {'Score': 2, 'Explanation': 'Living room area with a recliner.  Similar to 270, there is a possibility of other rooms, but no strong indication of a bedroom.'}}")
            return evaluator_prompt
        if prompt_type == 'planning':
            if reason != '' and subtask != '{}':
                planning_prompt = (f"The agent has been tasked with navigating to a {goal.upper()}. The agent has sent you the following elements:"
                f"(1)<The observed image>: The image taken from its current location. "
                f"(2){reason}. This explains why you should go in this direction. "
                f'Your job is to describe next place to go. '
                f'To help you plan your best next step, I can give you some human suggestions:. '
                f'(1) If the {goal} appears in the image, directly choose the target as the next step in the plan. Note a chair must have a backrest and a chair is not a stool. Note a chair is NOT sofa(couch) which is NOT a bed. '
                f'(2) If the {goal} is not found and the previous subtask {subtask} has not completed, continue to complete the last subtask {subtask} that has not been completed.'
                f'(3) If the {goal} is not found and the previous subtask {subtask} has already been completed. Identify a new subtask by describing where you are going next to be more likely to find clues to the the {goal} and think about whether the {goal} is likely to occur in that direction. Note you need to pay special attention to open doors and hallways, as they can lead to other unseen rooms. Note GOING UP OR DOWN STAIRS is an option. '
                "Format your answer in the json {{'Subtask': <Where you are going next>, 'Flag': <Whether the target is in your view, True or False>}}. "
                "Answer Example: {{'Subtask': 'Go to the hallway', 'Flag': False}} or {{'Subtask': "+f"'Go to the {goal}'"+", 'Flag': True}} or {{'Subtask': 'Go to the open door', 'Flag': True}}")
            else:
                planning_prompt = (f"The agent has been tasked with navigating to a {goal.upper()}. The agent has sent you an image taken from its current location."
                f'Your job is to describe next place to go. '
                f'To help you plan your best next step, I can give you some human suggestions:. '
                f'(1) If the {goal} appears in the image, directly choose the target as the next step in the plan. Note a chair must have a backrest and a chair is not a stool. Note a chair is NOT sofa(couch) which is NOT a bed. '
                f'(2) If the {goal} is not found, describe where you are going next to be more likely to find clues to the the {goal} and analyze the room type and think about whether the {goal} is likely to occur in that direction. Note you need to pay special attention to open doors and hallways, as they can lead to other unseen rooms. Note GOING UP OR DOWN STAIRS is an option. '
                "Format your answer in the json {{'Subtask': <Where you are going next>, 'Flag': <Whether the target is in your view, True or False>}}. "
                "Answer Example: {{'Subtask': 'Go to the hallway', 'Flag': False}} or {{'Subtask': "+f"'Go to the {goal}'"+", 'Flag': True}} or {{'Subtask': 'Go to the open door', 'Flag': True}}")
            return planning_prompt
        if prompt_type == 'action':
            if subtask != '{}':
                action_prompt = (
                f"TASK: {subtask}. Your final task is to NAVIGATE TO THE NEAREST {goal.upper()}, and get as close to it as possible. "
                f"There are {num_actions - 1} red arrows superimposed onto your observation, which represent potential actions. " 
                f"These are labeled with a number in a white circle, which represent the location you would move to if you took that action. {'NOTE: choose action 0 if you want to TURN AROUND or DONT SEE ANY GOOD ACTIONS. ' if self.step_ndx - self.turned >= self.cfg['turn_around_cooldown'] else ''}"
                f"In order to complete the subtask {subtask} and eventually the final task NAVIGATING TO THE NEAREST {goal.upper()}. Explain which action acheives that best. "
                "Return your answer as {{'action': <action_key>}}. Note you CANNOT GO THROUGH CLOSED DOORS, and you DO NOT NEED TO GO UP OR DOWN STAIRS"
                )
            else:
                action_prompt = (
                    f"TASK: NAVIGATE TO THE NEAREST {goal.upper()}, and get as close to it as possible. Use your prior knowledge about where items are typically located within a home. "
                    f"There are {num_actions - 1} red arrows superimposed onto your observation, which represent potential actions. "
                    f"These are labeled with a number in a white circle, which represent the location you would move to if you took that action. {'NOTE: choose action 0 if you want to TURN AROUND or DONT SEE ANY GOOD ACTIONS. ' if self.step_ndx - self.turned >= self.cfg['turn_around_cooldown'] else ''}"
                    f"First, tell me what you see in your sensor observation, and if you have any leads on finding the {goal.upper()}. Second, tell me which general direction you should go in. "
                    "Lastly, explain which action acheives that best, and return it as {{'action': <action_key>}}. Note you CANNOT GO THROUGH CLOSED DOORS, and you DO NOT NEED TO GO UP OR DOWN STAIRS"
                )
            return action_prompt

        raise ValueError('Prompt type must be goal, predicting, planning, or action')