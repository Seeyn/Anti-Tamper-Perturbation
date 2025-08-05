import pickle
import numpy as np
import argparse

def get_args():
    parent_parser = argparse.ArgumentParser()
    parent_parser.add_argument('--path', type=str, default=None)
    args = parent_parser.parse_args()
    return args

def get_ism(results):
    ism_result = np.array(results)
    # print(ism_result)
    ism_score = ism_result[ism_result!=-1].mean()
    fdfr_rate = (ism_result==-1).mean()
    return ism_score,fdfr_rate
#CLIP LIQE ISM FDFR

def print_results(path):
    with open(path,'rb') as f:
        results = pickle.load(f)
    if results['ism'].shape[0] != 800:
        print('Warning',results['ism'].shape[0])
    # print(results['ism'])
    ism_score,fdfr_rate = get_ism(results['ism'])
    metric_threshold = 0.1318359375 # metric threshold to define whether the protection succeed after generation
    metric = 'clip_iqac'
    print('%f	%f	%f	%f	%f'%((results['clip_iqac'].mean()),results['liqe'].mean(),ism_score,fdfr_rate,sum(results[metric]<metric_threshold)/len(results[metric])))

args = get_args()
locals().update(vars(args))
print('CLIP-IQAC	LIQE            ISM             FDFR             PSR')
print_results(path)