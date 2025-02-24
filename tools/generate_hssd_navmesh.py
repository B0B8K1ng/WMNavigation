import habitat_sim
from habitat_sim.nav import NavMeshSettings
import os
import magnum as mn

def _create_rgb_sensor_spec():
    """
    Create an RGB camera sensor specification.

    :return: RGB sensor specification.
    """
    rgb_sensor_spec = habitat_sim.CameraSensorSpec()
    rgb_sensor_spec.uuid = f"color_sensor"
    rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor_spec.resolution = [480, 640]
    rgb_sensor_spec.hfov = 79
    rgb_sensor_spec.position = mn.Vector3([0, 0.88, 0])
    rgb_sensor_spec.orientation = mn.Vector3([-0.25, 0, 0])
    return rgb_sensor_spec

def _create_depth_sensor_spec():
    """
    Create a depth camera sensor specification.

    :return: Depth sensor specification.
    """
    depth_sensor_spec = habitat_sim.CameraSensorSpec()
    depth_sensor_spec.uuid = f"depth_sensor"
    depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_spec.resolution = [480, 640]
    depth_sensor_spec.hfov = 79
    depth_sensor_spec.position = mn.Vector3([0, 0.88, 0])
    depth_sensor_spec.orientation = mn.Vector3([-0.25, 0, 0])
    return depth_sensor_spec

def _create_sensor_specs():
    """
    Create sensor specifications for the agent.

    :return: List of sensor specifications.
    """
    sensor_specs = []
    sensor_specs.append(_create_rgb_sensor_spec())
    sensor_specs.append(_create_depth_sensor_spec())
    return sensor_specs


# 设置HSSD数据集路径和输出目录
hssd_dataset_path = "/file_system/vepfs/algorithm/dujun.nie/data/hssd/scenes"  # 替换为HSSD数据集路径
output_navmesh_dir = "/file_system/vepfs/algorithm/dujun.nie/data/hssd/navmesh"  # 替换为保存NavMesh文件的输出目录



# 检查并创建输出目录
if not os.path.exists(output_navmesh_dir):
    os.makedirs(output_navmesh_dir)

# NavMesh生成配置
navmesh_settings = NavMeshSettings()
navmesh_settings.agent_radius = 0.18  # 可根据需要调整
navmesh_settings.agent_height = 0.88  # 可根据需要调整

# 获取HSSD数据集中所有场景文件
scene_files = [f for f in os.listdir(hssd_dataset_path) if f.endswith('.json')]

# 创建Habitat Simulator的配置
backend_cfg = habitat_sim.SimulatorConfiguration()
backend_cfg.scene_dataset_config_file = (
    "/file_system/vepfs/algorithm/dujun.nie/data/hssd/hssd-hab.scene_dataset_config.json"
        )

agent_config = habitat_sim.AgentConfiguration()
agent_config.radius = 0.18
agent_config.height = 0.88
agent_config.sensor_specifications = _create_sensor_specs()

# 遍历并生成NavMesh
for i, scene_file in enumerate(scene_files):
    scene_path = os.path.join(hssd_dataset_path, scene_file)
    backend_cfg.scene_id = scene_path
    # 输出NavMesh文件路径
    navmesh_output_path = os.path.join(output_navmesh_dir, scene_file.replace('.json', '.navmesh'))

    # 检查NavMesh文件是否已经存在
    if os.path.exists(navmesh_output_path):
        print(f"NavMesh already exists for {scene_file} ({i+1}/{len(scene_files)})")
        continue
    
    # 初始化模拟器
    try:
        sim = habitat_sim.Simulator(habitat_sim.Configuration(backend_cfg, [agent_config]))
   
    except Exception as e:
        print(f"Failed to generate NavMesh for {scene_file} ({i+1}/{len(scene_files)})")
        continue

    sim.recompute_navmesh(sim.pathfinder, navmesh_settings)
    print(f"NavMesh generated successfully for {scene_file} ({i+1}/{len(scene_files)})")
    # 保存NavMesh到输出目录
    sim.pathfinder.save_nav_mesh(navmesh_output_path)
    print(f"NavMesh saved to {navmesh_output_path}")
    sim.close()

    # 销毁当前模拟器实例以释放内存


print("NavMesh generation and saving complete.")

