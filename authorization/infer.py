import os
import torch
import numpy as np
import utils
import logging
from networks.HIDNet import HIDNet
import options
from dataset import *
import sys
import re
from tqdm import tqdm
import yaml

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

config = load_config('../configs/authorization.yaml')

logging.basicConfig(level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ])

seed = config['seed']
utils.set_seed(seed)
message_length = config['message_length']
args = options.get_args()
locals().update(vars(args))

print('seed:', seed)
print('message_length:', message_length)
print('block_size',config['block_size'])
print('input_folder:', input_dir)
print('running_folder:', output_dir)

model = HIDNet(device,gamma=gamma,input_mask=input_mask,pixel_space=pixel_space)
val_dataloader = torch.utils.data.DataLoader(TestDataset(train=False,get_path=True,path=input_dir), batch_size=10, shuffle=False)
weights = config['test']['weights']
if weights is not None:
    model_from_checkpoint(model,torch.load(weights,map_location=device))

###########Authorization###########


val_losses = {}
transform = transforms.Compose([
            transforms.Resize((512,512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
os.makedirs(output_dir,exist_ok=True)
os.makedirs(os.path.join(output_dir,'Authorization_messages'),exist_ok=True)

for batch_id, (path,image) in tqdm(enumerate(val_dataloader)):
    image = image.to(device)
    message = utils.get_phis(phi_dimension=message_length,batch_size=image.shape[0]).to(device)
    losses, (encoded_images, random_mask, decoded_messages) = model.val_one_batch([image,message])
    # print('bitwise-error:',losses['bitwise-error'])
    for key in losses:
        if key not in val_losses:
            val_losses[key] = 0
        val_losses[key] += losses[key]
    
    for id, e_img in enumerate(encoded_images):
        image_name = path[id].split('/')[-1].split('.')[0]
        id_dir = path[id].split('/')[-3]
        set_dir = path[id].split('/')[-2]
        dir_path = os.path.join(output_dir,id_dir,set_dir)
        os.makedirs(dir_path,exist_ok=True)
        transforms.ToPILImage()(((e_img.clip(-1,1)+1)/2).clip(0,1).cpu()).save('%s/%s.png'%(dir_path,image_name))
        torch_set ={'random_mask':random_mask.cpu(),'message':message[id].cpu()}
        torch.save(torch_set,'%s/Authorization_messages/%s.pt'%(output_dir,id_dir+'_'+image_name))
        





          

        



        