import ast
import logging
import numpy as np
import math
import concurrent.futures
import cv2

from simWrapper import PolarAction
from utils import *
from vlm import VLM


class ActionDistribution2D:
    """
    2D action distribution for sampling and fitting actions
    """
    def __init__(self, image_width, image_height, mean=None, std_dev=None):
        """
        Initialize ActionDistribution2D with image dimensions and optional mean and standard deviation.
        """
        self.image_width = image_width
        self.image_height = image_height
        self.mean = mean if mean is not None else np.array([image_width / 2, image_height / 2])
        self.std_dev = std_dev if std_dev is not None else np.array([image_width / 4, image_height / 4])

    def sample(self, num_samples):
        """
        Sample from the action distribution.
        """
        return np.random.normal(self.mean, self.std_dev, (num_samples, 2))

    def fit_to_selected_actions(self, selected_actions):
        """
        Fit the distribution to a new set of selected actions.
        """
        if not selected_actions:
            return
        # Update mean and standard deviation based on selected actions
        self.mean = np.mean(selected_actions, axis=0)
        self.std_dev = np.std(selected_actions, axis=0)


class PIVOT:
    """
    PIVOT algorithm for vision-based navigation with a Vision-Language Model (VLM).
    """
    def __init__(self, vlm: VLM, fov, image_dim, max_action_length=1):
        """
        Initialize PIVOT with the VLM, field of view (FOV), and image dimensions.
        """
        self.image_width = image_dim[1]
        self.image_height = image_dim[0]
        self.vlm = vlm
        self.fov = fov

        fov_radians = np.deg2rad(fov)
        self.max_action_length = max_action_length
        self.focal_length = (self.image_width / 2) / np.tan(fov_radians / 2)

    def run(self, rgb_image, instruction, agent_state, sensor_state, 
            num_iter=3, num_parallel=3, num_samples=8, goal_image=None):
        """
        Execute the PIVOT algorithm by sampling actions, generating prompts for VLM, and selecting actions.
        """
        action_dist_2d = ActionDistribution2D(self.image_width, self.image_height)
        start_pt = [0, 0, 0]
        
        # Transform agent and sensor states
        global_p = local_to_global(agent_state.position, agent_state.rotation, start_pt)
        local_point = global_to_local(sensor_state.position, sensor_state.rotation, global_p)
        if local_point[2] > 0:
            return None

        # Project 3D points into 2D image coordinates
        point_3d = [local_point[0], -local_point[1], -local_point[2]]
        if point_3d[2] == 0:
            point_3d[2] = 0.0001

        x = self.focal_length * point_3d[0] / point_3d[2]
        y = self.focal_length * point_3d[1] / point_3d[2]
        x_pixel = int(self.image_width / 2 + x)
        y_pixel = int(self.image_height / 2 + y)
        self.start_pxl = [x_pixel, y_pixel]

        pivot_images = {}
        for itr in range(num_iter):
            ns = num_samples - 2 * itr
            sampled_actions = action_dist_2d.sample(ns)
            
            # Annotate sampled actions on the image
            actions, annotated_image = self._annotate_on_image(rgb_image, sampled_actions)
            pivot_images[f'pivot_itr_{itr}'] = annotated_image
 
            # Construct the VLM prompt
            K = max(4 - itr, 1)
            prompt = (
                'I am a wheeled robot that cannot go over objects. This is the image I’m seeing right '
                'now. I have annotated it with numbered circles. Each number represents a general '
                f'direction I can follow. Now you are a five-time world-champion navigation agent and '
                f'your task is to tell me which circle I should pick for the task of: \n{instruction}\n'
                f'Choose {K} of the best candidate numbers. Do NOT choose routes that go through objects. '
                'Skip analysis and provide your answer in the following format:\n{"points": [] }'
            )

            # Use VLM to evaluate the prompt
            ims = [annotated_image]
            if goal_image is not None:
                ims.append(goal_image)

            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(self.vlm.call, ims, prompt) for _ in range(num_parallel)]
                results = [future.result() for future in concurrent.futures.as_completed(futures)]

            # Parse the VLM responses
            all_points = set()
            for response in results:
                try:
                    eval_resp = ast.literal_eval(response[response.rindex('{'):response.rindex('}') + 1])
                    all_points.update(eval_resp['points'])
                except Exception as e:
                    logging.error(f'Error parsing VLM response: {e}')

            selected_actions = [actions[i-1] for i in all_points if i <= len(actions)]
            action_dist_2d.fit_to_selected_actions(selected_actions)

        if not selected_actions:
            # If VLM did not select a valid action
            return PolarAction.default, pivot_images
        
        _, final_image = self._annotate_on_image(rgb_image, [selected_actions[0]], chosen=True)
        pivot_images[f'pivot_itr_{num_iter}'] = final_image

        choosen_action = self._get_chosen_action(selected_actions[0])


        return choosen_action, pivot_images

    def _get_chosen_action(self, action_pixel):
        """
        Get the corresponding polar action for moving max_action_length in the direction of action_pixel.
        """

        action_pixel = [int(action_pixel[0]), int(action_pixel[1])]
        theta = math.atan2((action_pixel[0] - self.image_width / 2), self.focal_length)
        action = PolarAction(self.max_action_length, -theta)
        return action
    

    def _annotate_on_image(self, rgb_image, sampled_actions, chosen=False):
        """
        Annotate the image with sampled action points.
        """
        scale_factor = rgb_image.shape[0] / 1080
        annotated_image = rgb_image.copy()
        action_name = 1
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_color = BLACK
        text_size = 2 * scale_factor
        text_thickness = math.ceil(2 * scale_factor)

        annotated = []
        for action_pixel in sampled_actions:
            # Make sure actions are within the image bounds and not too close to each other
            if (0.05 * rgb_image.shape[1] <= action_pixel[0] <= 0.95 * rgb_image.shape[1] and 
                0.05 * rgb_image.shape[0] <= action_pixel[1] <= 0.95 * rgb_image.shape[0] and 
                all(np.linalg.norm(np.array(action_pixel) - np.array(ap)) > 100 * scale_factor for ap in annotated)):
                
                cv2.arrowedLine(annotated_image, tuple(self.start_pxl), (int(action_pixel[0]), int(action_pixel[1])), 
                                RED, math.ceil(5 * scale_factor), tipLength=0.)
                
                text = str(action_name)
                (text_width, text_height), _ = cv2.getTextSize(text, font, text_size, text_thickness)
                circle_center = (int(action_pixel[0]), int(action_pixel[1]))
                circle_radius = max(text_width, text_height) // 2 + math.ceil(15 * scale_factor)
                circle_color = WHITE if not chosen else GREEN
                cv2.circle(annotated_image, circle_center, circle_radius, (circle_color), -1)
                cv2.circle(annotated_image, circle_center, circle_radius, RED, math.ceil(2 * scale_factor))
                
                text_position = (circle_center[0] - text_width // 2, circle_center[1] + text_height // 2)
                cv2.putText(annotated_image, text, text_position, font, text_size, text_color, text_thickness)
                
                annotated.append(action_pixel)
                action_name += 1

        return annotated, annotated_image