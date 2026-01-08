from glob import glob
import os
from pathlib import Path
import argparse
import yaml


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
device = "".join(filter(str.isdigit, device))
instance_paths = list(Path(input_dir).iterdir())
dir_suffix = config[method]["dir_suffix"]
for each in instance_paths:
    i_name = each.name
    if ".pt" in i_name:
        continue
    print(i_name)
    train_dir = Path(input_dir) / i_name / dir_suffix
    if purification == "":
        os.system(
            "CUDA_VISIBLE_DEVICES=%s bash generation/scripts/train_DB.sh %s "
            % (device, train_dir)
        )
    elif purification == "naive":
        os.system(
            "CUDA_VISIBLE_DEVICES=%s bash generation/scripts/train_DB_withPurified.sh %s "
            % (device, train_dir)
        )
    elif purification == "gridpure":
        os.system(
            "CUDA_VISIBLE_DEVICES=%s bash generation/scripts/train_DB_gridpure.sh %s "
            % (device, train_dir)
        )
    else:
        raise Exception("check purification type")
