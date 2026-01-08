import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

# Add project root to sys.path to import logger
sys.path.append(str(Path(__file__).resolve().parent.parent))
from logger import logger


def get_args():
    parent_parser = argparse.ArgumentParser()
    parent_parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        default=None,
        help="The directory of images for generation and used for generation",
    )
    parent_parser.add_argument("--method", type=str, default="CAAT")
    parent_parser.add_argument(
        "--device", required=True, type=str, help="device for training, e.g., cuda:0"
    )
    parent_parser.add_argument("--purification", type=str, default="")
    args = parent_parser.parse_args()
    return args


def load_config(config_file):
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config


args = get_args()
input_dir = args.input_dir
method = args.method
device = args.device
purification = args.purification
assert purification in set(["naive", "gridpure", ""])

config_path = Path("configs/generation.yaml").resolve()
config = load_config(config_path)
device_id = "".join(filter(str.isdigit, device))
instance_paths = list(Path(input_dir).iterdir())
dir_suffix = config[method]["dir_suffix"]

try:
    for each in instance_paths:
        i_name = each.name
        if ".pt" in i_name:
            continue
        logger.info(f"Processing: [bold green]{i_name}[/bold green]")
        train_dir = Path(input_dir) / i_name / dir_suffix

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = device_id

        if purification == "":
            script = "generation/scripts/train_DB.sh"
        elif purification == "naive":
            script = "generation/scripts/train_DB_withPurified.sh"
        elif purification == "gridpure":
            script = "generation/scripts/train_DB_gridpure.sh"
        else:
            raise Exception("check purification type")

        subprocess.run(["bash", script, str(train_dir)], env=env, check=True)

except KeyboardInterrupt:
    logger.warning("收到中断信号，正在停止当前任务...")
    sys.exit(130)
except subprocess.CalledProcessError as e:
    logger.error(f"脚本执行失败: {e}")
    sys.exit(1)
except Exception as e:
    logger.exception(f"发生错误: {e}")
    sys.exit(1)
