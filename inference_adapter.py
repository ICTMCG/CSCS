import sys

import os
import argparse
import numpy as np

import torch
import torch.nn.functional as F

from utils.utils import setup_seed

# import ID Embedder
from model.arcface.iresnet import iresnet100

# import ID Embedder Adapter
from model.arcface.iresnet_adapter import iresnet100_adapter

# import Generator
from model.faceshifter.layers.faceshifter.layers_arcface import AEI_Net

import glob

from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms

def parse_args():
    parser = argparse.ArgumentParser(description='testing')
    parser.add_argument('--device', default='cuda:0', type=str, help='cuda:n or cpu')
    
    parser.add_argument('--weight_path', type=str, required=True)
    parser.add_argument('--ID_emb_model_path', type=str, help='ID_Embedder_model_path', default=None, required=True)
    
    parser.add_argument('--src_path', type=str, required=True)
    parser.add_argument('--tgt_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)

    parser.add_argument('--adapter_type', type=str, default='add')

    parser.add_argument('--seed', type=int, default=999)
    
    args = parser.parse_args()
    return args


resize_T = transforms.Resize(size=(256, 256))
norm_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0.5, .5)
        ])


def load_sample(img_path):
    img = Image.open(img_path).convert('RGB')
    img = resize_T(img)
    img = norm_transform(img)

    return img



if __name__ == '__main__':

    # load configs
    opt = parse_args()

    # set random seed
    setup_seed(opt.seed)

    torch.cuda.set_device(opt.device)
    device = torch.device(opt.device)

    # set data path
    model_dir = opt.output_dir
    
    os.makedirs(model_dir,exist_ok=True)

    model_weight = torch.load(opt.weight_path, map_location='cpu')

    # load ID Embedder
    ID_emb = iresnet100()
    ID_emb.load_state_dict(torch.load(opt.ID_emb_model_path, map_location='cpu'))
    
    # load ID adapter
    ID_adapter = iresnet100_adapter(type=opt.adapter_type)
    ID_adapter.load_state_dict(model_weight['adapter'])

    # build Generator
    G = AEI_Net(512)
    G.load_state_dict(model_weight['G'])
    
    ID_emb = ID_emb.to(device)
    G = G.to(device)
    ID_adapter = ID_adapter.to(device)

    ID_emb.eval()
    G.eval()
    ID_adapter.eval()

    src = opt.src_path
    tgt = opt.tgt_path
    output_dir = opt.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        try:
            img_a = load_sample(src)
        except Exception as e:
            print('src image load error')
            print(src)
            print(e)
            raise e
        try:
            img_b = load_sample(tgt)
        except Exception as e:
            print('tgt image load error')
            print(tgt)
            print(e)
            raise e

        # convert numpy to tensor
        img_a = img_a.cuda()
        img_b = img_b.cuda()

        img_a = img_a.unsqueeze(0)
        img_b = img_b.unsqueeze(0)


        src_id = F.normalize(
                ID_emb(F.interpolate(img_a, size=112, mode="bilinear")),
                dim=-1,
                p=2,
            )
        src_id_adapt = F.normalize(
                ID_adapter(F.interpolate(img_a, size=112, mode="bilinear")),
                dim=-1,
                p=2,
            )
        if opt.adapter_type=='concat': 
            src_id = torch.cat([src_id, src_id_adapt], dim=1) 
        elif opt.adapter_type=='add':
            src_id = src_id + src_id_adapt
        elif opt.adapter_type=='replace':
            src_ID_emb_input = src_id_adapt
        swapped, attr, m = G(img_b, src_id)

        output_path = os.path.join(output_dir, '{}_{}.jpg'.format(os.path.basename(src.split('.')[0]), os.path.basename(tgt.split('.')[0])))

        swapped = swapped[0].cpu()*0.5 + 0.5
        swapped = swapped.numpy().transpose(1,2,0)
        swapped=np.clip(255*swapped, 0, 255)
        swapped = np.cast[np.uint8](swapped)

        Image.fromarray(swapped).save(output_path)
