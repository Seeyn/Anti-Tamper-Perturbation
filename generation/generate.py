from glob import glob
import os
import argparse
import yaml

def get_args():
    parent_parser = argparse.ArgumentParser()
    parent_parser.add_argument('--input_dir', type=str, required=True,default=None,
                                help='The directory of images for generation and used for generation')
    parent_parser.add_argument('--method', type=str, default='CAAT')
    parent_parser.add_argument('--device', required=True, type=str,
                                help='device for training, e.g., cuda:0')
    parent_parser.add_argument('--purification', type=str, default='')
    args = parent_parser.parse_args()
    return args


def load_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

args = get_args()
locals().update(vars(args))
assert purification in set(['naive','gridpure',''])

config_path = os.path.abspath(f'../configs/generation.yaml')
config = load_config(config_path)
device = device[-1]
instance_dir = glob(f'{input_dir}/*')
dir_suffix = config[method]['dir_suffix']
for each in instance_dir:
    i_name = each.split('/')[-1]
    if('.pt'in i_name):
        continue
    print(i_name)
    if purification == '':
        os.system('CUDA_VISIBLE_DEVICES=%s bash scripts/train_DB.sh %s '%(device, input_dir+'/'+i_name+f'/{dir_suffix}'))
    elif purification == 'naive':
        os.system('CUDA_VISIBLE_DEVICES=%s bash scripts/train_DB_withPurified.sh %s '%(device, input_dir+'/'+i_name+f'/{dir_suffix}'))
    elif purification == 'gridpure':
        os.system('CUDA_VISIBLE_DEVICES=%s bash scripts/train_DB_gridpure.sh %s '%(device, input_dir+'/'+i_name+f'/{dir_suffix}'))
    else:
        raise Exception("check purification type")

