
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import torch
from piq import CLIPIQA
from brisque import BRISQUE
obj = BRISQUE(url=False)
clipiqa = CLIPIQA()
from LIQE.LIQE import LIQE
ckpt = './LIQE/checkpoints/LIQE.pt'
lieq_model = LIQE(ckpt, device = 'cuda' if torch.cuda.is_available() else 'cpu')
# import ImageReward as RM
# RMmodel = RM.load("ImageReward-v1.0")
import clip
clip_model, clip_preprocess = clip.load("ViT-B/32")
import glob
import os 
from ism_fdfr import matching_score_id
from tqdm import tqdm
from torchvision.transforms import ToTensor
import torch
from torchvision import transforms
import pickle
import argparse
import yaml

def get_args():
    parent_parser = argparse.ArgumentParser()
    parent_parser.add_argument('--dataset', type=str, default='CelebA-HQ')
    parent_parser.add_argument('--method', type=str, default='CAAT')
    args = parent_parser.parse_args()
    return args


def load_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

args = get_args()
locals().update(vars(args))
config_path = os.path.abspath(f'../configs/metrics.yaml')
config = load_config(config_path)

if type(config['instance_dir']) is list:
    for instance_dir in config['instance_dir']:
        print(instance_dir)
        # method = instance_dir.split('/')[-1].split('_')[1]
        # dataset= instance_dir.split('/')[-1].split('_')[0]
        
        if dataset == 'CelebAHQ':
            dataset = 'CelebA-HQ'

        dir_suffix = config[method]['dir_suffix']
        dir_suffix_length = len(dir_suffix.split('/'))
        img_prefix = config[method]['img_prefix']
        prompt = config['prompt']
        output_path = config['output_dir']
        os.makedirs(output_path,exist_ok=True)
        generated_img_paths = glob.glob(f'{instance_dir}/*/{dir_suffix}/DREAMBOOTH/checkpoint-1000/dreambooth/{prompt}/*')

        id_emb_path = config[f'{dataset}']['id_emb_path']
        with open(f'{id_emb_path}','rb') as f:
            avg_id_set = pickle.load(f)

        item_list = ['checkpoint-1000'] 
        # item_list = ['checkpoint-1000-JPEG-50','checkpoint-1000-JPEG-70','checkpoint-1000-Resize-4','checkpoint-1000-Resize-2']
        trans = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        for item in item_list:
            liqe, clip_iqac, clip_iqa, brisque_scores,ism_scores, RM_scores= [], [], [], [],[],[]
            image_paths = []
            for input_path in tqdm(generated_img_paths):

                input = input_path.replace('checkpoint-1000',item)

                avg_id_embedding = avg_id_set[input.split('/')[-(5+dir_suffix_length)]]
                img = Image.open(input).convert("RGB")

                liqe_img = ToTensor()(img).unsqueeze(0)
                q1, s1, d1 = lieq_model(liqe_img)
                liqe_score = q1.item()
                # RM_score = RMmodel.score(f"{prompt}", [f"{input_path}"])
                gen_img = img
                type_name = 'face'
                image = clip_preprocess(gen_img).unsqueeze(0).to('cuda')
                text = clip.tokenize(["a good photo of " + type_name, "a bad photo of " + type_name]).to('cuda')
                similarity_matrix = None
                with torch.no_grad():
                    image_features = clip_model.encode_image(image)
                    text_features = clip_model.encode_text(text)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    similarity_matrix = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                iqac_score = similarity_matrix[0][0].item() - similarity_matrix[0][1].item()
                
                brisque_score = obj.score(img)            
                with torch.no_grad():
                    iqa_socre = clipiqa(trans(img).unsqueeze(0)).item()
                ism = matching_score_id(input,avg_id_embedding)
                if ism is None:
                    ism = -1
                ism_scores.append(ism)
                image_paths.append(input)
                liqe.append(liqe_score)
                clip_iqac.append(iqac_score)
                clip_iqa.append(iqa_socre)
                brisque_scores.append(brisque_score)
                # RM_scores.append(RM_score)

            results_dict = {
            'ism':np.array(ism_scores),
            'brisque':np.array(brisque_scores),
            'clip_iqa':np.array(clip_iqa),
            'clip_iqac':np.array(clip_iqac),
            'liqe':np.array(liqe),
            # 'RM':np.array(RM_scores),
            'path':image_paths
            }

            dir_name = instance_dir.split('/')[-1]
            with open(output_path+f'/{dataset}_{dir_name}_{item}_{prompt}','wb') as f:
                pickle.dump(results_dict,f)

else:
    raise Exception("Please input a list, the list can contain one element")



