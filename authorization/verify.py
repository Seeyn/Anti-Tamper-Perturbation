import os
import time
import torch
import numpy as np
import utils
import logging
from networks.HIDNet import HIDNet
from dataset import *
import sys
import re
from tqdm import tqdm
import yaml
import argparse
import cv2


def sorted_nicely(l):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

def last_checkpoint_from_folder(folder: str):
    last_file = sorted_nicely(os.listdir(folder))[-1]
    last_file = os.path.join(folder, last_file)
    return last_file

def model_from_checkpoint(hidden_net, checkpoint):
    """ Restores the hidden_net object from a checkpoint object """
    hidden_net.encoder_decoder.load_state_dict(checkpoint['enc-dec-model'],strict=False)
    hidden_net.optimizer_enc_dec.load_state_dict(checkpoint['enc-dec-optim'])
    hidden_net.discriminator.load_state_dict(checkpoint['discrim-model'],strict=False)
    hidden_net.optimizer_discrim.load_state_dict(checkpoint['discrim-optim'])

def load_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_args():
    parent_parser = argparse.ArgumentParser()
    parent_parser.add_argument('--input_dir', type=str, required=True,default=None,
                                help='The directory of input images')
    parent_parser.add_argument('--message_dir', type=str,required=True, default=None,
                                help='The directory of authorization messages')
    parent_parser.add_argument('--method', type=str, default='CAAT')
    parent_parser.add_argument('--device', default='cuda:0', type=str,
                                help='device for training, e.g., cuda:0')
    parent_parser.add_argument('--gamma', default=0.5, type=float,
                                help='deciding the 0-1 mask percentage')
    parent_parser.add_argument("--input-mask",
                        default=None,type=str,
                        help="input-mask path")
    parent_parser.add_argument("--noise-type",
                        default=None,type=str,
                        help="implement noise type")
    parent_parser.add_argument("--noise-args",
                        default=0.5, type=float,
                        help="noise args")
    parent_parser.add_argument("--pixel-space",
                        action="store_true",
                        help="pixel space or not")
    args = parent_parser.parse_args()
    return args

config = load_config('../configs/verification.yaml')

logging.basicConfig(level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ])

seed = config['seed']
utils.set_seed(seed)
message_length = config['message_length']
print('seed:', seed)
print('message_length:', message_length)

args = get_args()
locals().update(vars(args))
dir_suffix = config[method]['dir_suffix']
img_prefix = config[method]['img_prefix']

model = HIDNet(device,gamma=gamma,input_mask=input_mask,pixel_space=pixel_space)
weights = config['weights']
if weights is not None:
    model_from_checkpoint(model,torch.load(weights,map_location=device))
        
###########Verification###########
random_mask = model.random_mask
model = model.encoder_decoder
model.eval()
bitwise_avg_err = []

paths = glob(f'{input_dir}/*/{dir_suffix}/*.png')

suffix_length = len(dir_suffix.split('/'))

transform = transforms.Compose([
            transforms.Resize((512,512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

test_set = message_dir
print('noise_type',noise_type)
print('noise_args',noise_args)
bit_error_threshold = 3/32

if noise_type is not None:
    degrader = utils.Degradation(noise_type=noise_type,noise_args=noise_args)
unauthorized_image_num = 0
id_set = {}
for path in paths:
    id_dir = path.split('/')[-(1+suffix_length)]
    id_set[id_dir] = 0

for path in tqdm(paths):
    # print(path)
    image_path = path
    if noise_type is None:
        image = transform(Image.open(image_path)).unsqueeze(0)
    else:
        image = cv2.imread(image_path).astype(np.float32)/255.
        image = degrader.degrade(image).unsqueeze(0)
    image = image.to(device)
    image_name = image_path.split('/')[-1]
    id_dir = image_path.split('/')[-(1+suffix_length)]
    torch_set = torch.load(os.path.join(test_set,id_dir+'_'+image_name.replace('.png','.pt').replace(img_prefix,'')))
    message = torch_set['message'].to(device)
    random_mask = torch_set['random_mask'].to(device)
    img_dct_masked, img_dct, random_mask = model.dctlayer(image,random_mask=random_mask)
    encoded_dct, encoded_image = model.idctlayer(img_dct_masked,img_dct,random_mask)
    decoded_message = model.decoder(encoded_dct)
    decoded_rounded = decoded_message.detach().cpu().numpy().round().clip(0, 1)
    bitwise_error = np.sum(np.abs(decoded_rounded - message.round().detach().cpu().numpy())) / (
            message.shape[0])
    if bitwise_error>bit_error_threshold:
        unauthorized_image_num += 1
        id_set[id_dir] += 1
    bitwise_avg_err.append(bitwise_error)

Pass_num = 0
for each in id_set:
    if id_set[each] == 0:
        Pass_num += 1

print('Average Bit-Error:',np.array(bitwise_avg_err).mean(),' / Unauthorized image Ratio:',unauthorized_image_num/len(paths),' / Pass Verification Identity Num:',Pass_num)





          

        



        