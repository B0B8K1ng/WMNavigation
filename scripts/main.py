import argparse
import yaml

from dotenv import load_dotenv
from api import *
from WMNav_agent import *
from WMNav_env import *
from custom_agent import *
from custom_env import *

load_dotenv()
def set_seed(seed: int):
    # 设置 random 模块的种子
    random.seed(seed)
    # 设置 numpy 模块的种子
    np.random.seed(seed)

def main():
    parser = argparse.ArgumentParser(description="Run a dynamic envmark")

    parser.add_argument('--config', type=str, default='WMNav', help='name of the YAML config file')
    parser.add_argument('-n', '--name', type=str, help='Name for the run (optional)')
    parser.add_argument('-lf', '--log_freq', type=int, help='Logging frequency (optional)')
    parser.add_argument('-ms', '--max_steps', type=int, help='Max steps per episode (optional)')
    parser.add_argument('-ne', '--num_episodes', type=int, help='Number of episodes to run (optional)')
    parser.add_argument('-pa', '--parallel', action='store_true', help='Enable parallel execution')
    parser.add_argument('--instances', type=int, help='Number of instances for parallel execution (optional)')
    parser.add_argument('--instance', type=int, help='Instance number for parallel execution (optional)')
    parser.add_argument('--port', type=int, help='port number for Flask server parallel execution (optional)')
    parser.add_argument('--dataset', type=str, default='hm3d_v0.2', choices=['hm3d_v0.1', 'hm3d_v0.2', 'mp3d'], help='Type of dataset')

    args = parser.parse_args()

    # Load configuration from YAML file
    with open(f'config/{args.config}.yaml', 'r') as file:
        config = yaml.safe_load(file)
    # Override YAML config with command-line arguments if provided
    if args.name is not None:
        config['env_cfg']['name'] = args.name
    if args.log_freq is not None:
        config['env_cfg']['log_freq'] = args.log_freq
    if args.max_steps is not None:
        config['env_cfg']['max_steps'] = args.max_steps
    if args.num_episodes is not None:
        config['env_cfg']['num_episodes'] = args.num_episodes
    if args.instances is not None:
        config['env_cfg']['instances'] = args.instances
    if args.instance is not None:
        config['env_cfg']['instance'] = args.instance
    if args.port is not None:
        config['env_cfg']['port'] = args.port
    if args.parallel:
        config['env_cfg']['parallel'] = True
    if args.dataset:
        config['env_cfg']['dataset'] = args.dataset
    env_cls = globals()[config['env_cls']]
    env = env_cls(cfg=config)
    env.run_experiment()

if __name__ == '__main__':
    set_seed(48)
    main()
