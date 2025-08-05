from glob import glob
import os
import yaml
import argparse


def get_args():
    parent_parser = argparse.ArgumentParser()
    parent_parser.add_argument('--method', type=str, default='CAAT')
    parent_parser.add_argument('--input_dir', type=str, required=True,default=None,
                                help='The directory of authorized images')
    parent_parser.add_argument('--output_dir', type=str,required=True, default=None,
                                help='The directory of output ATP images')
    parent_parser.add_argument('--device', required=True, type=str,
                                help='device for training, e.g., cuda:0')
    args = parent_parser.parse_args()
    return args

def load_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

args = get_args()
locals().update(vars(args))

assert method in set(['CAAT','ANTIDB', 'ADVDM' ,'METACLOAK'])

config_path = os.path.abspath('../configs/protection.yaml')
config = load_config(config_path)
device = device[-1]
os.makedirs(output_dir,exist_ok=True)
input_dir = glob(f'{input_dir}/*')
for key in config[method]:
    print(key,config[method][key])

if method == 'CAAT':
    for each in input_dir:
        i_name = each.split('/')[-1] 
        if '.pt' in i_name:
            continue
        print(i_name)
        os.system('cd CAAT; CUDA_VISIBLE_DEVICES=%s bash train_CAAT_freq.sh %s %s %s'%(device, each+'/set_B',output_dir+'/'+i_name,config_path))
        
elif method == 'METACLOAK':
    for each in input_dir:
        i_name = each.split('/')[-1] 
        if '.pt' in i_name:
            continue
        print(i_name)
        project_path = os.path.abspath('./Metacloak')
        os.system('cd Metacloak; CUDA_VISIBLE_DEVICES=%s bash script/gen_and_eval_freq.sh %s %s %s'%(device, each,output_dir+'/'+i_name,project_path))

elif method == 'ADVDM':
    for each in input_dir:
        i_name = each.split('/')[-1] 
        if '.pt' in i_name:
            continue
        print(i_name)
        os.system('cd ADVDM; CUDA_VISIBLE_DEVICES=%s bash scripts/train_adv_freq.sh %s %s %s'%(device, each+'/set_B',output_dir+'/'+i_name,config_path))

elif method == 'ANTIDB':
    for each in input_dir:
        i_name = each.split('/')[-1] 
        if '.pt' in i_name:
            continue
        print(i_name)
        os.system('cd ANTIDB; CUDA_VISIBLE_DEVICES=%s bash scripts/gen_aspl_freq.sh %s %s %s'%(device, each,output_dir+'/'+i_name,config_path))

else:
    raise "Method Unknown"