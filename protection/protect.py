from glob import glob
import os
from pathlib import Path
import yaml
import argparse


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
device = "".join(filter(str.isdigit, device))
Path(output_dir).mkdir(parents=True, exist_ok=True)
input_paths = list(Path(input_dir).iterdir())
for key in config[method]:
    print(key, config[method][key])

if method == "CAAT":
    for each in input_paths:
        i_name = each.name
        if ".pt" in i_name:
            continue
        print(i_name)
        os.system(
            "CUDA_VISIBLE_DEVICES=%s bash protection/CAAT/train_CAAT_freq.sh %s %s %s"
            % (device, each / "set_B", Path(output_dir) / i_name, config_path)
        )

elif method == "METACLOAK":
    for each in input_paths:
        i_name = each.name
        if ".pt" in i_name:
            continue
        print(i_name)
        project_path = Path("protection/Metacloak").resolve()
        os.system(
            "CUDA_VISIBLE_DEVICES=%s bash protection/Metacloak/script/gen_and_eval_freq.sh %s %s %s"
            % (device, each, Path(output_dir) / i_name, project_path)
        )

elif method == "ADVDM":
    for each in input_paths:
        i_name = each.name
        if ".pt" in i_name:
            continue
        print(i_name)
        os.system(
            "CUDA_VISIBLE_DEVICES=%s bash protection/ADVDM/scripts/train_adv_freq.sh %s %s %s"
            % (device, each / "set_B", Path(output_dir) / i_name, config_path)
        )

elif method == "ANTIDB":
    for each in input_paths:
        i_name = each.name
        if ".pt" in i_name:
            continue
        print(i_name)
        os.system(
            "CUDA_VISIBLE_DEVICES=%s bash protection/ANTIDB/scripts/gen_aspl_freq.sh %s %s %s"
            % (device, each, Path(output_dir) / i_name, config_path)
        )

else:
    raise ValueError("Method Unknown")
