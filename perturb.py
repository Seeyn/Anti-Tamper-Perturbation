import os
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


print("###Authorization Start###")
os.system(
    "python authorization/infer.py --device %s --input_dir  %s --output_dir %s"
    % (device, input_dir, Path(output_dir) / "Authorized")
)
print("Output at: %s" % output_dir)
print("###Authorization End###")

print()

print("###Protection Start###")
os.system(
    "python protection/protect.py --device %s --method %s --input_dir  %s --output_dir %s"
    % (
        device,
        method,
        Path(output_dir) / "Authorized",
        Path(output_dir) / f"ATP_{method}",
    )
)
print("###Protection End###")
