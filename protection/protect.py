import subprocess
import sys
from glob import glob
import os
from pathlib import Path
import yaml
import argparse

# Add project root to sys.path to import logger
sys.path.append(str(Path(__file__).resolve().parent.parent))
from logger import logger


def get_args():
    parent_parser = argparse.ArgumentParser()
    parent_parser.add_argument("--method", type=str, default="CAAT")
    parent_parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        default=None,
        help="The directory of authorized images",
    )
    parent_parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        default=None,
        help="The directory of output ATP images",
    )
    parent_parser.add_argument(
        "--device", required=True, type=str, help="device for training, e.g., cuda:0"
    )
    args = parent_parser.parse_args()
    return args


def load_config(config_file):
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config


args = get_args()
method = args.method
input_dir = args.input_dir
output_dir = args.output_dir
device = args.device

assert method in set(["CAAT", "ANTIDB", "ADVDM", "METACLOAK"])

config_path = Path("configs/protection.yaml").resolve()
config = load_config(config_path)
device_id = "".join(filter(str.isdigit, device))
Path(output_dir).mkdir(parents=True, exist_ok=True)
input_paths = list(Path(input_dir).iterdir())

for key in config[method]:
    logger.info(f"{key}: [bold cyan]{config[method][key]}[/bold cyan]")

try:
    for each in input_paths:
        i_name = each.name
        if ".pt" in i_name:
            continue
        logger.info(f"Processing: [bold green]{i_name}[/bold green]")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = device_id

        cmd = []
        if method == "CAAT":
            cmd = [
                "bash",
                "protection/CAAT/train_CAAT_freq.sh",
                str(each / "set_B"),
                str(Path(output_dir) / i_name),
                str(config_path),
            ]
        elif method == "METACLOAK":
            project_path = Path("protection/Metacloak").resolve()
            cmd = [
                "bash",
                "protection/Metacloak/script/gen_and_eval_freq.sh",
                str(each),
                str(Path(output_dir) / i_name),
                str(project_path),
            ]
        elif method == "ADVDM":
            cmd = [
                "bash",
                "protection/ADVDM/scripts/train_adv_freq.sh",
                str(each / "set_B"),
                str(Path(output_dir) / i_name),
                str(config_path),
            ]
        elif method == "ANTIDB":
            cmd = [
                "bash",
                "protection/ANTIDB/scripts/gen_aspl_freq.sh",
                str(each),
                str(Path(output_dir) / i_name),
                str(config_path),
            ]
        else:
            raise ValueError("Method Unknown")

        if cmd:
            subprocess.run(cmd, env=env, check=True)

except KeyboardInterrupt:
    logger.warning("收到中断信号，正在停止当前任务...")
    sys.exit(130)
except subprocess.CalledProcessError as e:
    logger.error(f"任务执行失败: {e}")
    sys.exit(1)
except Exception as e:
    logger.exception(f"发生错误: {e}")
    sys.exit(1)
