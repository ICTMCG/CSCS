import sys
import itertools

from torch.nn.modules import loss
sys.path.append('../')
import os
import argparse
import json
import numpy as np

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
from torch.cuda.amp import GradScaler

from torch.nn import SyncBatchNorm
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from utils.utils import load_config, setup_seed

from utils.logger import create_logger, Progbar, AverageMeter

# import ID Embedder
from model.arcface.iresnet import iresnet100

# import ID Embedder Adapter
from model.arcface.iresnet_adapter import iresnet100_adapter

# import Generator
from model.faceshifter.layers.faceshifter.layers_arcface import AEI_Net

# import Discriminator
from model.stylegan2.networks import Discriminator

# import losses
from model.stylegan2.loss import StyleGAN2Loss
from model.losses import face_cos_loss, pixel_level_change_loss

from data.dataset import SMSwap_dataset_arcface as SMSwap_dataset
from data.dataset import SMSwap_dataset_ssl_Aug_arcface as SMSwap_dataset_ssl_Aug
import glob

def parse_args():
    parser = argparse.ArgumentParser(description='training')
    parser.add_argument('--local_rank', type=int, help='local rank for dist')
    parser.add_argument('--data_path', type=str, help='path to datasets', required=True)
    parser.add_argument('--data_ssl_path', type=str, help='path to ssl datasets', required=True)
    parser.add_argument('--expr_path', type=str, help='path to expr dir', required=True)
    parser.add_argument('--config_name', type=str, help='model configuration file')
    parser.add_argument('--adapter_type', type=str, help='add, concat or replace')
    parser.add_argument('--run_id', type=str, default=None, help='run_id')
    parser.add_argument('--resume', default=0, type=int, help='resume from an existing checkpoint (default: 0)')
    parser.add_argument('--device', default='cuda:0', type=str, help='cuda:n or cpu')
    parser.add_argument('--overwrite', default=0, type=int, help='overwrite existing files (default: 0)')
    parser.add_argument('--epoch', default=None, type=int, help='resume model epoch')
    parser.add_argument('--save_models', default=0, type=int, help='whether to save model')

    parser.add_argument('--ID_emb_model_path', type=str, help='ID_Embedder_model_path', default=None)

    parser.add_argument('--pretrain_model_path', type=str, help='pretrain_model_path', default=None)
    parser.add_argument('--pretrain_id', type=str, default=None)
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    # load configs
    opt = parse_args()
    config = load_config('configs.{}'.format(opt.config_name))
    config_attr = dir(config)
    config_params = {config_attr[i]: getattr(config, config_attr[i]) for i in range(len(config_attr)) if config_attr[i][:2] != '__'}

    opt.run_id = '{}{}{}{}'.format(
        opt.run_id,
        '_seed'+str(config.seed),
        '_pretraind' if opt.pretrain_model_path is not None else '',
        '{}'.format(opt.pretrain_id) if opt.pretrain_id is not None else '',
        )

    train_params = {
        'data_path': opt.data_path,
        'model_dir': opt.expr_path,
        'config_name': opt.config_name,
        'run_id': opt.run_id
    }

    # set random seed
    setup_seed(config.seed)

    
    # set dist
    word_size=torch.cuda.device_count()
    local_rank  = opt.local_rank

    dist.init_process_group(backend='nccl')
    print('dist init done')
    # set gpu device
    torch.cuda.set_device(local_rank)
    device = torch.device(local_rank)

    # set data path
    model_dir = opt.expr_path
    train_data_path = opt.data_path
    train_data_ssl_path = opt.data_ssl_path
    
    os.makedirs(model_dir,exist_ok=True)

    
    model_path = os.path.join(model_dir, "model.pth.tar")

    if local_rank==0:
        logger = create_logger(model_dir)

        if os.path.exists(model_path):
            if opt.overwrite:
                logger.info('%s exists. overwrite' % model_path)
            else:
                logger.info('%s exists. stop' % model_path)
                sys.exit(0)
        logger.info('model save dir: %s' % model_dir)

        # save configs
        train_params_file = os.path.join(model_dir, 'train_params.json')
        with open(train_params_file, 'w') as fp:
            json.dump(train_params, fp, indent=4)
        config_file = os.path.join(model_dir, 'configs.json')
        with open(config_file, 'w') as fp:
            json.dump(config_params, fp, indent=4)

        logger.info('train_params: %s',train_params)
        logger.info('config_params: %s',config_params)

    # dataset
    train_set = SMSwap_dataset(train_data_path, config.img_size)
    train_sampler = DistributedSampler(train_set)
    
    train_ssl_set = SMSwap_dataset_ssl_Aug(train_data_ssl_path, config.img_size, aug_type=config.aug_type, aug_prob=.5)
    train_ssl_sampler = DistributedSampler(train_ssl_set)

    if local_rank==0:
        train_loader = DataLoader(
            dataset=train_set,
            num_workers=config.num_workers,
            batch_size=config.batch_size-config.batch_size//config.rec_scale,
            sampler=train_sampler,
            pin_memory=True,
            shuffle=False, 
            drop_last=True
        )
        train_self_recon_set = SMSwap_dataset(train_data_path, config.img_size, self_recons=True)
        train_self_recon_loader = DataLoader(
            dataset=train_self_recon_set,
            num_workers=config.num_workers,
            batch_size=config.batch_size//config.rec_scale,
            pin_memory=True,
            shuffle=False, 
            drop_last=True
        )
    else:
        train_loader = DataLoader(
            dataset=train_set,
            num_workers=config.num_workers,
            batch_size=config.batch_size,
            sampler=train_sampler,
            pin_memory=True,
            shuffle=False, 
            drop_last=True
        )
        
    train_ssl_loader = DataLoader(
        dataset=train_ssl_set,
        num_workers=config.num_workers,
        batch_size=config.batch_size,
        sampler=train_ssl_sampler,
        pin_memory=True,
        shuffle=False, 
        drop_last=True
    )

    if local_rank==0:
        writer = SummaryWriter(logdir=model_dir)


    # load ID Embedder
    ID_emb = iresnet100()
    ID_emb.load_state_dict(torch.load(opt.ID_emb_model_path, map_location='cpu'))

    ID_adapter = iresnet100_adapter(type=opt.adapter_type)
    ID_adapter.load_state_dict(torch.load(opt.ID_emb_model_path, map_location='cpu'), strict=False)
    
    # build Generator
    if config.backbone == 'faceshifter':
        if opt.adapter_type=='concat':
            G = AEI_Net(1024)
        elif opt.adapter_type=='add' or opt.adapter_type=='replace':
            G = AEI_Net(512)
        
    D = Discriminator(0, config.img_size[0], 3)
    

    perf_log_file = os.path.join(model_dir, "%s.txt" % config.metric)
    board_num = 0
    best_loss = 999999999999999
    no_impr_counter = 0
    start_epoch = 0
    epoch=0


    if opt.epoch is not None:
        print(model_dir)
        model_path = glob.glob(model_dir+ "*/model_"+str(opt.epoch)+'*')[0]

    if opt.resume:
        if opt.pretrain_model_path is not None and os.path.exists(opt.pretrain_model_path):
            logger.info('pretrain_model_path %s' % opt.pretrain_model_path)
            pretrained_dict = torch.load(opt.pretrain_model_path, map_location='cpu')
            G.load_state_dict(pretrained_dict['G'])
            D.load_state_dict(pretrained_dict['D'])
            ID_adapter.load_state_dict(pretrained_dict['adapter'])
            logger.info('pretrain model load success!')

        elif os.path.exists(model_path):
            logger.info('resume from %s' % model_path)
            pretrained_dict = torch.load(model_path, map_location='cpu')
            G.load_state_dict(pretrained_dict['G'])
            D.load_state_dict(pretrained_dict['D'])
            ID_emb.load_state_dict(pretrained_dict['ID_emb'])
            ID_adapter.load_state_dict(pretrained_dict['adapter'])
            epoch = pretrained_dict['epoch']
            board_num, best_loss = open(perf_log_file).read().split()[:2]
            board_num = int(board_num)
            best_loss = float(best_loss)

    ID_emb = ID_emb.to(device)

    ID_adapter = ID_adapter.to(device)
    ID_adapter = SyncBatchNorm.convert_sync_batchnorm(ID_adapter)
    ID_adapter = DistributedDataParallel(ID_adapter, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False, find_unused_parameters=True)
    G = G.to(device)
    G = SyncBatchNorm.convert_sync_batchnorm(G)
    G = DistributedDataParallel(G, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False, find_unused_parameters=True)
    D = D.to(device)
    D = SyncBatchNorm.convert_sync_batchnorm(D)
    D = DistributedDataParallel(D, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False, find_unused_parameters=True)
    
    loss_attr_clac = nn.MSELoss()
    
    id_loss = face_cos_loss()
    change_loss = pixel_level_change_loss()
    if config.adv_type=='stylegan2':
        adv_loss = StyleGAN2Loss(device, G, D)
        D_reg_interval = 16


    optimG = torch.optim.Adam([
        {'params':itertools.chain(G.parameters(), ID_adapter.parameters()), 'lr':config.G_init_lr}
        ])
    
    
    optimD = torch.optim.Adam([
        {'params':D.parameters(), 'lr':config.D_init_lr}
        ])


    if local_rank==0:
        logger.info("begin to train!")
    s_time = time.time()
    scaler = torch.cuda.amp.GradScaler()

    ID_emb.eval()
    ID_emb.requires_grad_(False)
    ID_adapter.train()
    G.train()
    D.train()

    while True:
        if local_rank==0:
            progbar = Progbar(len(train_set), stateful_metrics=['epoch', 'config'])
        batch_time = AverageMeter()
        end = time.time()

        train_sampler.set_epoch(epoch)

        if local_rank==0:
            self_recon_iter = iter(train_self_recon_loader)
            
        ssl_iter = iter(train_ssl_loader)
            
        skip_adv = -1

        for batch_idx, batch in enumerate(train_loader):
            board_num += 1

            src_batch, tgt_batch= batch

            src_batch = src_batch.reshape((-1, 3, src_batch.size(-2), src_batch.size(-1)))
            tgt_batch = tgt_batch.reshape((-1, 3, tgt_batch.size(-2), tgt_batch.size(-1)))
            
            if local_rank==0:
                try:
                    batch_selfrecon = next(self_recon_iter)
                except StopIteration:
                    self_recon_iter = iter(train_self_recon_loader)
                    batch_selfrecon = next(self_recon_iter)

                
                src_batch_srecon, tgt_batch_srecon = batch_selfrecon
                src_batch_srecon = src_batch_srecon.reshape((-1, 3, src_batch_srecon.size(-2), src_batch_srecon.size(-1)))
                tgt_batch_srecon = tgt_batch_srecon.reshape((-1, 3, tgt_batch_srecon.size(-2), tgt_batch_srecon.size(-1)))

                src_batch = torch.cat([src_batch, src_batch_srecon], dim=0)
                tgt_batch = torch.cat([tgt_batch, tgt_batch_srecon], dim=0)

            src_batch = src_batch.to(device)
            tgt_batch = tgt_batch.to(device)


            src_ID_embedding = F.normalize(
                    ID_emb(F.interpolate(src_batch, size=112, mode="bilinear")),
                    dim=-1,
                    p=2,
                )

            src_ID_emb_adapt = F.normalize(
                ID_adapter(F.interpolate(src_batch, size=112, mode="bilinear")),
                dim=-1,
                p=2,
            )
            if opt.adapter_type=='concat': 
                src_ID_emb_input = torch.cat([src_ID_embedding, src_ID_emb_adapt], dim=1) 
            elif opt.adapter_type=='add':
                src_ID_emb_input = src_ID_embedding + src_ID_emb_adapt
            elif opt.adapter_type=='replace':
                src_ID_emb_input = src_ID_emb_adapt

            
            # train D
            if board_num > skip_adv:
                optimD.zero_grad()
                if config.adv_type=='stylegan2':
                    D_reg = board_num % D_reg_interval == 0
                    if config.backbone[:11]=='faceshifter':
                        loss_D_fake, loss_D_real = adv_loss.get_Dloss_faceshifter(tgt_batch, src_ID_emb_input, Dreg = D_reg)
                    else:
                        loss_D_fake, loss_D_real = adv_loss.get_Dloss(tgt_batch, src_ID_emb_input, Dreg = D_reg)
                    loss_D_fake = loss_D_fake.mean()
                    loss_D_real = loss_D_real.mean()
                    if D_reg:
                        loss_D_real = D_reg_interval * loss_D_real
                    loss_D = loss_D_real+loss_D_fake

                loss_D.backward(retain_graph=True)
                if config.clip_grad:
                    nn.utils.clip_grad_norm_(D.parameters(), 2.0)
                optimD.step()

            # train G
            optimG.zero_grad()
            if config.backbone[:11]=='faceshifter':
                swapped, tgt_attr, m = G(tgt_batch, src_ID_emb_input)
            else:
                swapped = G(tgt_batch, src_ID_emb_input)
            
            swapped_ID_embedding = F.normalize(
                    ID_emb(F.interpolate(swapped, size=112, mode="bilinear")),
                    dim=-1,
                    p=2,
                )
            loss_id = id_loss(src_ID_embedding, swapped_ID_embedding)

            chg_type='only_selfrecon'
            if chg_type=='only_selfrecon':
                loss_chg = change_loss(tgt_batch_srecon.to(device), swapped[-config.batch_size//config.rec_scale:])
            else:
                loss_chg = change_loss(tgt_batch, swapped)
            if config.adv_type=='stylegan2':
                loss_G = adv_loss.get_Gloss(swapped).mean()
                
            swap_attr = G.module.get_attr(swapped)
            
            
            if board_num > skip_adv:
                loss = config.l_id * loss_id + config.l_chg * loss_chg + config.l_adv * loss_G
            else:
                loss = config.l_chg * loss_chg
            loss.backward()
            if config.clip_grad:
                nn.utils.clip_grad_norm_(ID_emb.parameters(), 2.0)
                nn.utils.clip_grad_norm_(G.parameters(), 2.0)
                nn.utils.clip_grad_norm_(D.parameters(), 2.0)
            optimG.step()

            
            ''' get ssl data '''
            try:
                batch_ssl  =  next(ssl_iter)
            except StopIteration:
                ssl_iter   =  iter(train_ssl_loader)
                batch_ssl  =  next(ssl_iter)
                
            gt_ssl_batch, src_ssl_batch, tgt_ssl_batch = batch_ssl
            
            gt_ssl_batch = gt_ssl_batch.to(device)
            src_ssl_batch = src_ssl_batch.to(device)
            tgt_ssl_batch = tgt_ssl_batch.to(device)
            
            src_ssl_ID = F.normalize(
                    ID_emb(F.interpolate(src_ssl_batch, size=112, mode="bilinear")),
                    dim=-1,
                    p=2,
                )
            src_ssl_ID_adapt = F.normalize(
                ID_adapter(F.interpolate(src_ssl_batch, size=112, mode="bilinear")),
                dim=-1,
                p=2,
            )
                
            if opt.adapter_type=='concat':
                src_ssl_ID_input = torch.cat([src_ssl_ID, src_ssl_ID_adapt], dim=1) 
            elif opt.adapter_type=='add':
                src_ssl_ID_input = src_ssl_ID + src_ssl_ID_adapt
            elif opt.adapter_type=='replace':
                src_ssl_ID_input = src_ssl_ID_adapt


            # train D ssl
            optimD.zero_grad()
            if config.adv_type=='stylegan2':
                D_reg = board_num % D_reg_interval == 0
                if config.backbone[:11]=='faceshifter':
                    loss_D_fake_ssl, loss_D_real_ssl = adv_loss.get_Dloss_faceshifter(tgt_ssl_batch, src_ssl_ID_input, x_real = gt_ssl_batch, Dreg = D_reg)
                else:
                    loss_D_fake_ssl, loss_D_real_ssl = adv_loss.get_Dloss(tgt_ssl_batch, src_ssl_ID_input, x_real = gt_ssl_batch, Dreg = D_reg)
                loss_D_fake_ssl = loss_D_fake_ssl.mean()
                loss_D_real_ssl = loss_D_real_ssl.mean()
                if D_reg:
                    loss_D_real_ssl = D_reg_interval * loss_D_real_ssl
                loss_D_ssl = loss_D_fake_ssl + loss_D_real_ssl


            loss_D_ssl.backward(retain_graph=True)
            if config.clip_grad:
                nn.utils.clip_grad_norm_(D.parameters(), 2.0)
            optimD.step()
                
            # train G ssl 
            optimG.zero_grad()
            if config.backbone[:11]=='faceshifter':
                swapped_ssl, tgt_ssl_attr, m = G(tgt_ssl_batch, src_ssl_ID_input)

            swapped_ssl_ID = F.normalize(
                    ID_emb(F.interpolate(swapped_ssl, size=112, mode="bilinear")),
                    dim=-1,
                    p=2,
                )
            gt_ssl_ID = F.normalize(
                    ID_emb(F.interpolate(gt_ssl_batch, size=112, mode="bilinear")),
                    dim=-1,
                    p=2,
                )
            loss_id_ssl = id_loss(src_ssl_ID, swapped_ssl_ID)
            src_gt_sim = 1 - id_loss(src_ssl_ID, gt_ssl_ID)
            
            loss_id_ssl = (src_gt_sim**config.ssl_id_sim_beta) * loss_id_ssl
            
            loss_chg_ssl = change_loss(gt_ssl_batch, swapped_ssl)

            if config.adv_type=='stylegan2':
                loss_G_ssl = adv_loss.get_Gloss(swapped_ssl).mean()


            
            if board_num > skip_adv:
                loss_ssl = config.l_id_ssl * loss_id_ssl + config.l_chg_ssl * loss_chg_ssl + config.l_adv_ssl * loss_G_ssl
            else:
                loss_ssl = config.l_chg_ssl * loss_chg_ssl
            loss_ssl.backward()
            if config.clip_grad:
                nn.utils.clip_grad_norm_(G.parameters(), 2.0)
            optimG.step()
            
            
            losses = {'l':loss.item(), 'l_id':loss_id.item(), 'l_chg':loss_chg.item(),  'l_G': loss_G.item(), 'l_D':loss_D.item(), 'l_D_re':loss_D_real.item(), 'l_D_fa':loss_D_fake.item(),
                      'l_s':loss_ssl.item(), 'l_id_s':loss_id_ssl.item(), 'l_chg_s':loss_chg_ssl.item(), 'l_G_s': loss_G_ssl.item(), 'l_D_s':loss_D_ssl.item(), 'l_D_re_s':loss_D_real_ssl.item(), 'l_D_fa_s':loss_D_fake_ssl.item()
                     }

            if local_rank==0:
                for loss_key in losses.keys():
                    writer.add_scalars(loss_key, {'loss_key': losses[loss_key]}, board_num)

                if board_num % config.vis_interval == 0:
                    print()
                    info = 'step %d -> ' % (board_num)
                    for loss_key in losses.keys():
                        info += loss_key + ' %.4f, ' % (losses[loss_key])
                    logger.info(info)
                    writer.add_image('src_img', vutils.make_grid(src_batch[:config.vis_num, :, :, :], normalize=True,scale_each=True), board_num)
                    writer.add_image('tgt_img', vutils.make_grid(tgt_batch[:config.vis_num, :, :, :], normalize=True,scale_each=True), board_num)
                    writer.add_image('swapped_img', vutils.make_grid(swapped[:config.vis_num, :, :, :], normalize=True,scale_each=True), board_num)
                    show_img = torch.cat([src_batch[:config.vis_num, :, :, :], tgt_batch[:config.vis_num, :, :, :], swapped[:config.vis_num, :, :, :]], dim=2)
                    vutils.save_image(vutils.make_grid(show_img, normalize=True,scale_each=True), os.path.join(model_dir,'show_{:06}.jpg'.format(board_num)))
                    
                    writer.add_image('src_img_ssl', vutils.make_grid(src_ssl_batch[:config.vis_num, :, :, :], normalize=True,scale_each=True), board_num)
                    writer.add_image('tgt_img_ssl', vutils.make_grid(tgt_ssl_batch[:config.vis_num, :, :, :], normalize=True,scale_each=True), board_num)
                    writer.add_image('swapped_img_ssl', vutils.make_grid(swapped_ssl[:config.vis_num, :, :, :], normalize=True,scale_each=True), board_num)
                    writer.add_image('gt_img_ssl', vutils.make_grid(gt_ssl_batch[:config.vis_num, :, :, :], normalize=True,scale_each=True), board_num)
                    show_img_ssl = torch.cat([src_ssl_batch[:config.vis_num, :, :, :], tgt_ssl_batch[:config.vis_num, :, :, :], gt_ssl_batch[:config.vis_num, :, :, :], swapped_ssl[:config.vis_num, :, :, :]], dim=2)
                    vutils.save_image(vutils.make_grid(show_img_ssl, normalize=True,scale_each=True), os.path.join(model_dir,'show_ssl_{:06}.jpg'.format(board_num)))
                    
                progbar.add(src_batch.size(0),
                            values=[('epoch', epoch)]+[(loss_key,losses[loss_key]) for loss_key in losses.keys()])

            batch_time.update(time.time() - end)
            end = time.time()
            if batch_idx == len(train_loader)-1 and local_rank==0:
                logger.info('epoch: %d ',epoch )
                logger.info('iters: %s',str(board_num))
                logger.info('losses: %s',str(losses))


        epoch+=1
            
        if local_rank==0:
            if loss.item() < best_loss:
                best_loss = loss.item()
                no_impr_counter = 0
                torch.save({
                    'epoch': epoch,
                    'ID_emb': ID_emb.state_dict(),
                    'G': G.module.state_dict(),
                    'D': D.module.state_dict(),
                    'adapter': ID_adapter.module.state_dict(),
                    config.metric: best_loss
                }, os.path.join(model_dir,"model.pth.tar"))
            else:
                is_best = False
                no_impr_counter += 1

            if (epoch+1) % config.save_interval == 0: 
                torch.save({
                    'epoch': epoch,
                    'ID_emb': ID_emb.state_dict(),
                    'G': G.module.state_dict(),
                    'D': D.module.state_dict(),
                    'adapter': ID_adapter.module.state_dict(),
                    config.metric: loss.item()
                }, os.path.join(model_dir, 'model_' + str(epoch) + '_' + config.metric + '_' + str(
                    round(loss.item(),4)) 
                    + '.pth.tar'))

                logger.info('epoch %d -> val %s: %.4f, best loss %s: %.4f' % (epoch, 
                                                                                    config.metric, loss.item(), config.metric, best_loss))

        else:
            if local_rank==0:
                logger.info('epoch %d -> val %s: %.4f, best %s: %.4f' % (epoch, config.metric, loss.item(), config.metric, best_loss))

        if local_rank==0:
            open(perf_log_file, 'w').write('{} {}'.format(board_num, best_loss))

        break_state=False

        if board_num > config.max_steps:
            if local_rank==0:
                logger.info("touch the max iteration {}".format(config.max_steps))
            break_state=True

        if break_state:
            if local_rank==0:
                torch.save({
                    'epoch': epoch,
                    'ID_emb': ID_emb.state_dict(),
                    'G': G.module.state_dict(),
                    'D': D.module.state_dict(),
                    'adapter': ID_adapter.module.state_dict(),
                    config.metric: best_loss
                }, os.path.join(model_dir,"model_final.pth.tar"))


                time_span = time.time() - s_time                
                logger.info("training done in {} minutes".format(time_span / 60.0))
            break
