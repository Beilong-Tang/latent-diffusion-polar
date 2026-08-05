# This script combines all the data and save them as a single npy file
import fire
from omegaconf import OmegaConf
import argparse
import torch as th
from tqdm import tqdm
import numpy as np

import os
import os.path as op
import sys
sys.path = [os.getcwd()] + sys.path

from main import DataModuleFromConfig
from ldm.data.lsun import LSUNChurchesTrain, LSUNChurchesValidation
from ldm.util import instantiate_from_config

def main(config_path: str = "configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml", 
         out_path:str = "workspace/lsun/lsun_churches_train.npy",
         mode:str = 'train',
         resolution=256):

    conf = OmegaConf.load(config_path)
    data_conf = conf['data']
    data_conf['params']['train']['params']['return_uint8'] = True
    data_conf['params']['validation']['params']['return_uint8'] = True
    data_module = instantiate_from_config(data_conf)
    data_module.setup()

    # Training
    if mode == 'train':
        data_loader = data_module.train_dataloader()
    elif mode == 'val':
        data_loader = data_module.val_dataloader()
    else:
        raise ValueError(f"unknown mode {mode}")
    dataset_length = len(data_loader.dataset)
    print(f"dataset length: {dataset_length}")
    batch_size = data_module.batch_size

    output = th.empty(dataset_length, resolution, resolution, 3, dtype = th.uint8)

    cursor = 0
    for index, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
        image = batch['image']
        assert image.dtype == th.uint8, f"data dtype is {image.dtype}"
        assert image.size(1) == resolution

        batch_size = len(image)
        output[cursor: cursor+batch_size] = image
        cursor += batch_size

    output = output.numpy()

    if mode =='train':
        np.save(out_path, output)
    elif mode == "val":
        np.save(op.join(op.dirname(out_path), 'lsun_churches_val.npy'), output)
    else:
        raise ValueError(f"unknown mode {mode}")
    

if __name__ == "__main__":
    fire.Fire(main)

print("done")


"""
Example usage

python workspace/lsun/extract_data_to_numpy.py --mode train # This is for the training

python workspace/lsun/extract_data_to_numpy.py --mode val 
"""