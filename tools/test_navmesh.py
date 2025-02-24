import habitat_sim
import os
import argparse
def main(scene_id):
    # 设置场景和NavMesh路径
    navmesh_file = "/mnt/data/navmesh/" + scene_id + ".scene_instance.navmesh"  # 替换为你的NavMesh文件路径

    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_dataset_config_file = (
        "/mnt/data/hssd/hssd-hab.scene_dataset_config.json"
            )
    backend_cfg.scene_id = (
        "/mnt/data/hssd/scenes/" + scene_id + ".scene_instance.json"
     )


    # First, let's create a stereo RGB agent
    left_rgb_sensor = habitat_sim.bindings.CameraSensorSpec()
    # Give it the uuid of left_sensor, this will also be how we
    # index the observations to retrieve the rendering from this sensor
    left_rgb_sensor.uuid = "left_sensor"
    left_rgb_sensor.resolution = [540, 960]
    left_rgb_sensor.hfov = 90
    # The left RGB sensor will be 1.5 meters off the ground
    # and 0.25 meters to the left of the center of the agent
    left_rgb_sensor.position = 0.8 * habitat_sim.geo.UP + 0.1 * habitat_sim.geo.LEFT

    # Same deal with the right sensor
    right_rgb_sensor = habitat_sim.CameraSensorSpec()
    right_rgb_sensor.uuid = "right_sensor"
    right_rgb_sensor.resolution = [540, 960]
    right_rgb_sensor.hfov = 90
    # The left RGB sensor will be 1.5 meters off the ground
    # The right RGB sensor will be 1.5 meters off the ground
    # and 0.25 meters to the right of the center of the agent
    right_rgb_sensor.position = 0.8 * habitat_sim.geo.UP + 0.1 * habitat_sim.geo.RIGHT

    agent_config = habitat_sim.AgentConfiguration()
    # Now we simply set the agent's list of sensor specs to be the two specs for our two sensors
    agent_config.sensor_specifications = [left_rgb_sensor, right_rgb_sensor]

    # 初始化模拟器
    sim = habitat_sim.Simulator(habitat_sim.Configuration(backend_cfg, [agent_config]))

    # 加载预生成的NavMesh
    if os.path.exists(navmesh_file):
        success = sim.pathfinder.load_nav_mesh(navmesh_file)
        if success:
            print(f"NavMesh loaded successfully from {navmesh_file}")
        else:
            print(f"Failed to load NavMesh from {navmesh_file}")
    else:
        print(f"NavMesh file not found: {navmesh_file}")

    # 使用模拟器...
    # 例如获取随机可导航点
    agent_state = sim.get_agent(0).state
    agent_state.position = sim.pathfinder.get_random_navigable_point()
    print("Agent position:", agent_state.position)

    # 关闭模拟器
    sim.close()
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some parameters.")
    parser.add_argument('--scene_id', type=str, required=True, help='The first parameter')

    args = parser.parse_args()
    main(args.scene_id)
