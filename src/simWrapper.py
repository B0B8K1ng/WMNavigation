import logging
import habitat_sim
import numpy as np
import magnum as mn

from habitat_sim.utils.common import quat_from_angle_axis, quat_to_angle_axis
from utils import local_to_global


class PolarAction:
    
    default = None #Default action if the VLM response is not parsed
    stop = None #Stop action
    null = None #Null action to just get current observation

    def __init__(self, r, theta, type=None):
        self.theta = theta
        self.r = r
        self.type = type

PolarAction.stop = PolarAction(0, 0, 'stop')
PolarAction.null = PolarAction(0, 0, 'null')
    

class SimWrapper:
    """
    A wrapper for Habitat-Sim that initializes agents, sensors, and manages movement and sensor observations.
    """
    def __init__(self, cfg):
        """
        Initialize the simulator with the specified configurations.

        :param cfg: Dictionary with configurations for the simulator, agents, and sensors.
        """
        self.scene_id = cfg['scene_id']
        self.use_goal_image_agent = cfg['use_goal_image_agent']
        self.allow_slide = cfg['allow_slide']
        self.sensor_pitch = cfg['sensor_cfg']['pitch']
        self.fov = cfg['sensor_cfg']['fov']
        self.sensor_height = cfg['sensor_cfg']['height']
        self.resolution = (
            cfg['sensor_cfg']['img_height'],
            cfg['sensor_cfg']['img_width']
        )

        # Simulator configuration
        backend_cfg = habitat_sim.SimulatorConfiguration()
        backend_cfg.scene_id = cfg['scene_path']
        backend_cfg.scene_dataset_config_file = cfg['scene_config']
        backend_cfg.enable_physics = True

        # Agent configuration
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.radius = cfg['agent_radius']
        agent_cfg.height = cfg['agent_height']
        agent_cfg.sensor_specifications = self._create_sensor_specs()

        agents = [agent_cfg]
        if self.use_goal_image_agent:
            goal_cfg = self._create_goal_agent_config(fov=cfg['goal_image_agent_fov'])
            agents.append(goal_cfg)

        self.sim_cfg = habitat_sim.Configuration(backend_cfg, agents)

        try:
            self.sim = habitat_sim.Simulator(self.sim_cfg)
            # if cfg['dataset'] == 'hssd':
            #     self.sim.pathfinder.load_nav_mesh(cfg['scene_path'].replace('/stages/', '/navmesh/').replace('.glb', '.scene_instance.navmesh'))
            #self.sim.pathfinder.load_nav_mesh(navmesh_file)# if hssd, then load navmesh
            logging.info('Simulator initialized!')
        except Exception as e:
            logging.error(f"Error initializing simulator: {e}")
            raise RuntimeError(f"Could not initialize simulator: {e}")

    def _create_sensor_specs(self):
        """
        Create sensor specifications for the agent.

        :return: List of sensor specifications.
        """
        sensor_specs = []
        sensor_specs.append(self._create_rgb_sensor_spec())
        sensor_specs.append(self._create_depth_sensor_spec())
        return sensor_specs

    def _create_rgb_sensor_spec(self):
        """
        Create an RGB camera sensor specification.

        :return: RGB sensor specification.
        """
        rgb_sensor_spec = habitat_sim.CameraSensorSpec()
        rgb_sensor_spec.uuid = f"color_sensor"
        rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        rgb_sensor_spec.resolution = self.resolution
        rgb_sensor_spec.hfov = self.fov
        rgb_sensor_spec.position = mn.Vector3([0, self.sensor_height, 0])
        rgb_sensor_spec.orientation = mn.Vector3([self.sensor_pitch, 0, 0])
        return rgb_sensor_spec

    def _create_depth_sensor_spec(self):
        """
        Create a depth camera sensor specification.

        :return: Depth sensor specification.
        """
        depth_sensor_spec = habitat_sim.CameraSensorSpec()
        depth_sensor_spec.uuid = f"depth_sensor"
        depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
        depth_sensor_spec.resolution = self.resolution
        depth_sensor_spec.hfov = self.fov
        depth_sensor_spec.position = mn.Vector3([0, self.sensor_height, 0])
        depth_sensor_spec.orientation = mn.Vector3([self.sensor_pitch, 0, 0])
        return depth_sensor_spec

    def _create_goal_agent_config(self, fov):
        """
        Create a configuration for the goal image capture agent.

        :param fov: Field of view for the goal agent.
        :return: Agent configuration for the goal agent.
        """
        goal_cfg = habitat_sim.agent.AgentConfiguration()
        goal_sensor_spec = habitat_sim.CameraSensorSpec()
        goal_sensor_spec.uuid = "goal_sensor"
        goal_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        goal_sensor_spec.resolution = self.resolution
        goal_sensor_spec.hfov = fov
        goal_sensor_spec.orientation = mn.Vector3([0, 0, 0])
        goal_sensor_spec.position = mn.Vector3([0, 0, 0])
        goal_cfg.sensor_specifications = [goal_sensor_spec]
        return goal_cfg

    def step(self, action: PolarAction):
        """
        Move the agent based on the specified action and magnitude.

        :param action: The action to perform.
        """
        agent = self.sim.get_agent(0)
        curr_state = agent.get_state()

        if action is PolarAction.null:
            new_agent_state = curr_state
        else:
            new_agent_state = habitat_sim.AgentState()
            new_agent_state.position = np.copy(curr_state.position)
            new_agent_state.rotation = curr_state.rotation
            self._rotate_yaw(new_agent_state, new_agent_state.rotation, action.theta)
            self._move_forward(new_agent_state, new_agent_state.position, new_agent_state.rotation, action.r)
            agent.set_state(new_agent_state)

        observation = self.sim.get_sensor_observations(0)
        observation['agent_state'] = agent.get_state()
        return observation

    def _move_forward(self, agent_state, curr_position, curr_rotation, magnitude):
        """
        Move the agent_state forward by a specified magnitude.

        :param agent_state: The state of the agent to be updated.
        :param curr_position: Current position of the agent.
        :param curr_rotation: Current rotation of the agent.
        :param magnitude: Distance to move forward.
        """
        local_point = np.array([0, 0, -magnitude])
        global_point = local_to_global(curr_position, curr_rotation, local_point)
        delta = (global_point - curr_position) / 10
        new_position = np.copy(curr_position)

        for _ in range(10):
            if self.allow_slide:
                new_position = self.sim.pathfinder.try_step(new_position, new_position + delta)
            else:
                new_position = self.sim.pathfinder.try_step_no_sliding(new_position, new_position + delta)

        agent_state.position = new_position

    def _rotate_yaw(self, agent_state, curr_rotation, magnitude):
        """
        Rotate the agent_state by a specified angle.

        :param agent_state: The state of the agent to be updated.
        :param curr_rotation: Current rotation of the agent.
        :param magnitude: Angle in radians to rotate counterclockwise.
        """
        theta, axis = quat_to_angle_axis(curr_rotation)
        if axis[1] < 0:  # Ensure consistent rotation direction
            theta = 2 * np.pi - theta
        new_theta = theta + magnitude
        agent_state.rotation = quat_from_angle_axis(new_theta, np.array([0, 1, 0]))

    def get_goal_image(self, goal_position, goal_rotation):
        """
        Capture an image from the goal agent's perspective.

        :param goal_position: Position of the goal agent.
        :param goal_rotation: Rotation of the goal agent.
        :return: The captured image from the goal sensor.
        """
        assert self.use_goal_image_agent, "Goal image agent is not enabled."

        goal_agent = self.sim.get_agent(1)
        new_agent_state = habitat_sim.AgentState()
        new_agent_state.position = goal_position
        new_agent_state.rotation = goal_rotation
        goal_agent.set_state(new_agent_state)

        observations = self.sim.get_sensor_observations(1)
        return observations['goal_sensor']

    def set_state(self, pos, quat):
        """
        Set the agent's state to the specified position and orientation.

        :param pos: The position to set.
        :param quat: The quaternion orientation to set.
        """
        agent = self.sim.get_agent(0)
        agent_state = habitat_sim.AgentState()
        agent_state.position = pos
        agent_state.rotation = quat
        agent.set_state(agent_state)

    def get_path(self, path):
        """
        Find a path to the specified target position.

        :param path: Target position for pathfinding.
        :return: The path found by the pathfinder.
        """
        if self.sim.pathfinder.find_path(path):
            return path.geodesic_distance
        else:
            logging.info('NO PATH FOUND')
            return 1000

    def reset(self):
        """
        Close the simulator to clean up memory.
        """
        try:
            self.sim.close()
        except:
            pass