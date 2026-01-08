import subprocess
import sys
from pathlib import Path
import yaml
import argparse


def get_args():
    parent_parser = argparse.ArgumentParser()
    parent_parser.add_argument(
        "--input_dir",
        type=str,
        required=False,
        default="data/CelebA-HQ",
        help="The directory of input images",
    )
    parent_parser.add_argument(
        "--output_dir",
        type=str,
        required=False,
        default="data/perturbed",
        help="The directory of output authorized images",
    )
    parent_parser.add_argument(
        "--device",
        required=False,
        type=str,
        default="cuda:0",
        help="device for training, e.g., cuda:0",
    )
    parent_parser.add_argument(
        "--input-mask", default=None, type=str, help="input-mask path"
    )
    parent_parser.add_argument("--method", type=str, default="CAAT")
    args = parent_parser.parse_args()
    return args


def load_config(config_file):
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config


args = get_args()
input_dir = args.input_dir
output_dir = args.output_dir
device = args.device
method = args.method


try:
    print("###Authorization Start###")
    subprocess.run(
        [
            sys.executable,
            "authorization/infer.py",
            "--device",
            device,
            "--input_dir",
            input_dir,
            "--output_dir",
            str(Path(output_dir) / "Authorized"),
        ],
        check=True,
    )
    print("Output at: %s" % output_dir)
    print("###Authorization End###")

    print()

    print("###Protection Start###")
    subprocess.run(
        [
            sys.executable,
            "protection/protect.py",
            "--device",
            device,
            "--method",
            method,
            "--input_dir",
            str(Path(output_dir) / "Authorized"),
            "--output_dir",
            str(Path(output_dir) / f"ATP_{method}"),
        ],
        check=True,
    )
    print("###Protection End###")
except KeyboardInterrupt:
    print("\n[!] 收到中断信号，正在退出...")
    sys.exit(130)
except subprocess.CalledProcessError as e:
    print(f"\n[!] 执行出错: {e}")
    sys.exit(1)
