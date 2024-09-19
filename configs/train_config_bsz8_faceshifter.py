class Config(object):
    G_init_lr = 4e-4
    D_init_lr = 4e-4

    max_steps = 1600000

    max_epochs = 50
    step_size = 500
    gamma = 0.9
    batch_size = 8
    num_workers = 8
    board_num = 0
    early_stop_bar = 99999
    vis_interval=100
    metric = 'loss'

    img_size = (256,256)

    gray_channel=False
    
    l_id=1
    l_chg=10
    l_adv=1
    l_att=0
    l_percep=1
    l_cx=1

    l_id_ssl=1
    l_chg_ssl=20
    l_adv_ssl=1
    
    l_bak=0
    l_bak_ssl=0
    l_land_ssl=0
    
    
    ssl_id_sim_beta=0
    
    rec_scale=2
    
    seed=0
    backbone='faceshifter'
    
    aug_type='all'

    save_interval=1
    vis_num=8

    clip_grad=True

    adv_type='stylegan2'