from __future__ import print_function

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

torch.backends.cudnn.benchmark = True

from disparity.dataloader.messytable import MessytableDataset



# from disparity.dataloader import KITTILoader3D as ls
# from disparity.dataloader import KITTILoader_dataset3d as DA
from disparity.models import *
from disparity.Losses import get_losses
from tools.env_utils import *
from disparity.util.config import cfg
from disparity.util.metrics import EPE_metric
from disparity.dataloader.warp_ops import apply_disparity_cu
from disparity.util.cascade_metrics import compute_err_metric
from disparity.util.util import disp_error_img, save_scalars, save_images,setup_logger
import tensorboardX


# multiprocessing distributed training
import torch.distributed as dist
import torch.utils.data.distributed
import torch.multiprocessing as mp



def get_parser():
    parser = argparse.ArgumentParser(description='ActiveStereoNet')
    parser.add_argument('-cfg', '--cfg', '--config', default='', help='config path')
    # parser.add_argument('--data_path', default='./data/kitti/training/', help='data_path')
    parser.add_argument('--datapath', default='./data/kitti2015/training/',
                    help='datapath')
    parser.add_argument('--datapath12', default='./data/kitti2012/training/',
                    help='datapath')
    parser.add_argument('--epochs', type=int, default=5, help='number of epochs to train')
    parser.add_argument('--loadmodel', default=None, help='load model')
    parser.add_argument('--debug', action='store_true', default=False, help='debug mode')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--devices', '-d', type=str, default=None)
    parser.add_argument('--lr_scale', type=int, default=40, metavar='S', help='lr scale')
    parser.add_argument('--split_file', default='./data/kitti/train.txt', help='split file')
    parser.add_argument('--btrain', '-btrain', type=int, default=None)
    parser.add_argument('--start_epoch', type=int, default=None)
    parser.add_argument('--logdir', required=True, help='Directory to save logs and checkpoints')
    parser.add_argument('--log_freq', type=int, default=500, help='Frequency of saving temporary results')

    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    ## for distributed training
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    args = parser.parse_args()
    #print(args)
    cfg.merge_from_file(args.cfg)

    os.makedirs(args.logdir, exist_ok=True)

    if not args.devices:
        args.devices = str(np.argmin(mem_info()))

    if args.devices is not None and '-' in args.devices:
        gpus = args.devices.split('-')
        gpus[0] = 0 if not gpus[0].isdigit() else int(gpus[0])
        gpus[1] = len(mem_info()) if not gpus[1].isdigit() else int(gpus[1]) + 1
        args.devices = ','.join(map(lambda x: str(x), list(range(*gpus))))

    if not args.dist_url:
        args.dist_url = "tcp://127.0.0.1:{}".format(random_int() % 30000)

    print('Using GPU:{}'.format(args.devices))
    os.environ['CUDA_VISIBLE_DEVICES'] = args.devices

    return args, cfg

def main():
    args, cfg = get_parser()

    if args.debug:
        args.savemodel = './outputs/debug/'
        args.btrain = 1
        args.workers = 0

    logger = setup_logger("ActiveStereoNet", distributed_rank=args.rank, save_dir=args.logdir)

    
    reset_seed(args.seed)


    ### distributed training ###
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    ngpus_per_node = torch.cuda.device_count()
    print('ngpus_per_node: {}'.format(ngpus_per_node))
    args.ngpus_per_node = ngpus_per_node

    args.distributed = ngpus_per_node > 0 and (args.world_size > 1 or args.multiprocessing_distributed)
    args.multiprocessing_distributed = args.distributed

    if args.distributed and args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args, cfg, logger))
    else:
        # Simply call main_worker function
        main_worker(0, ngpus_per_node, args, cfg, logger)

def main_process(args):
    return not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)

def main_worker(gpu, ngpus_per_node, args, cfg, logger):
    summary_writer = tensorboardX.SummaryWriter(logdir=args.logdir)
    print("Using GPU: {} for training".format(gpu))
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    #------------------- Model -----------------------
    # model = hitnet(cfg)

    model = Active_StereoNet()
    optimizer = optim.Adam(model.parameters(), lr=0.1, betas=(0.9, 0.999))

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        torch.cuda.set_device(gpu)
        model.cuda(gpu)
        # When using a single GPU per process and per
        # DistributedDataParallel, we need to divide the batch size
        # ourselves based on the total number of GPUs we have
        args.btrain = int(args.btrain / ngpus_per_node)
        args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu], find_unused_parameters=True)
    elif ngpus_per_node > 1:
        model = torch.nn.DataParallel(model).cuda()
    else:
        torch.cuda.set_device(gpu)
        model = model.cuda(gpu)

    #------------------- Data Loader -----------------------
    # all_left_img, all_right_img, all_left_disp, = ls.dataloader(args.data_path,
    #                                                             args.split_file,
    #                                                             depth_disp=True,
    #                                                             cfg=cfg,
    #                                                             is_train=True)

    #train_left_img, train_right_img, train_left_disp,train_left_norm, test_left_img, test_right_img, test_left_disp, test_left_norm = ls_2015.dataloader(args.datapath)
    #train_left_img12, train_right_img12, train_left_disp12,train_left_norm12, test_left_img12, test_right_img12, test_left_disp12, test_left_norm12 = ls_2012.dataloader(args.datapath12)

    train_dataset = MessytableDataset(cfg.SPLIT.TRAIN, gaussian_blur=False, color_jitter=False, debug=args.debug, sub=600)
    val_dataset = MessytableDataset(cfg.SPLIT.VAL, gaussian_blur=False, color_jitter=False, debug=args.debug, sub=100)
    #print(args.distributed)
    if args.distributed:
        train_sampler = torch.utils.data.DistributedSampler(train_dataset, num_replicas=dist.get_world_size(),
                                                            rank=dist.get_rank())
        val_sampler = torch.utils.data.DistributedSampler(val_dataset, num_replicas=dist.get_world_size(),
                                                          rank=dist.get_rank())

        TrainImgLoader = torch.utils.data.DataLoader(train_dataset, cfg.SOLVER.BATCH_SIZE, sampler=train_sampler,
                                                     num_workers=cfg.SOLVER.NUM_WORKER, drop_last=True, pin_memory=True)
        ValImgLoader = torch.utils.data.DataLoader(val_dataset, cfg.SOLVER.BATCH_SIZE, sampler=val_sampler,
                                                   num_workers=cfg.SOLVER.NUM_WORKER, drop_last=False, pin_memory=True)
    else:
        TrainImgLoader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.SOLVER.BATCH_SIZE,
                                                     shuffle=True, num_workers=cfg.SOLVER.NUM_WORKER, drop_last=True)

        ValImgLoader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.SOLVER.BATCH_SIZE,
                                                   shuffle=False, num_workers=cfg.SOLVER.NUM_WORKER, drop_last=False)



    # ImageFloader = DA.myImageFloder(all_left_img, all_right_img, all_left_disp, True, split=args.split_file, cfg=cfg)

    # ImageFloader = torch.utils.data.DataLoader(
    #     DA.myImageFloder(train_left_img+train_left_img12, train_right_img+train_right_img12, train_left_disp+train_left_disp12,train_left_norm+train_left_norm12, True),
    #     batch_size=args.btrain, shuffle=True, num_workers=2, drop_last=False,pin_memory=True)

    #ImageFloader = DA.myImageFloder(train_left_img+train_left_img12, train_right_img+train_right_img12, train_left_disp+train_left_disp12,train_left_norm+train_left_norm12, True)
    

    #if args.distributed:
    #    train_sampler = torch.utils.data.distributed.DistributedSampler(ImageFloader)
    #else:
    #    train_sampler = None

    #TrainImgLoader = torch.utils.data.DataLoader(
    #    ImageFloader, 
    #    batch_size=args.btrain, shuffle=(train_sampler is None), num_workers=args.workers, drop_last=True,
    #    collate_fn=BatchCollator(cfg), 
    #    sampler=train_sampler)

    args.max_warmup_step = min(len(TrainImgLoader), 500)

    #------------------ Logger -------------------------------------
    if main_process(args):
        logger.info('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
        #writer = exp.writer

    # ------------------------ Resume ------------------------------
    if args.loadmodel is not None:
        if main_process(args):
            logger.info('load model ' + args.loadmodel)
        state_dict = torch.load(args.loadmodel)
        model.load_state_dict(state_dict['state_dict'], strict=False)
        if 'optimizer' in state_dict:
            try:
                optimizer.load_state_dict(state_dict['optimizer'])
                if main_process(args):
                    logger.info('Optimizer Restored.')
            except Exception as e:
                if main_process(args):
                    logger.error(str(e))
                    logger.info('Failed to restore Optimizer')
        else:
            if main_process(args):
                logger.info('No saved optimizer.')
        if args.start_epoch is None:
            args.start_epoch = state_dict['epoch'] + 1

    if args.start_epoch is None:
        args.start_epoch = 1


    critiron = get_losses('xtloss', max_disp=cfg.ARGS.MAX_DISP)

    # ------------------------ Training ------------------------------
    for epoch in range(args.start_epoch, args.epochs + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        total_train_loss = 0
        adjust_learning_rate(optimizer, epoch, args=args)

        best_loss = 1000000000000


        for batch_idx, sample in enumerate(TrainImgLoader):
            start_time = time.time()
            if epoch == 1 and batch_idx < args.max_warmup_step:
                adjust_learning_rate(optimizer, epoch, batch_idx, args=args)

            scalar_outputs_psmnet, img_outputs_psmnet = train(model, cfg, args, optimizer, sample, critiron)
            loss = scalar_outputs_psmnet['epe loss']
            
            if batch_idx%args.log_freq == 0:
                    global_step = batch_idx * epoch
                    # Update PSMNet images
                    save_images(summary_writer, 'train_ActiveStereo', img_outputs_psmnet, global_step)
                    # Update PSMNet losses
                    save_scalars(summary_writer, 'train_ActiveStereo', scalar_outputs_psmnet, global_step)

            if main_process(args):
                #logger.info('%s: %s' % (args.savemodel.strip('/').split('/')[-1], args.devices))
                if batch_idx%args.log_freq == 0:
                    logger.info('Epoch %d Iter %d/%d training loss = %.3f , time = %.2f; Epoch time: %.3fs, Left time: %.3fs, lr: %.6f' % (
                        epoch, 
                        batch_idx, len(TrainImgLoader), loss, time.time() - start_time, (time.time() - start_time) * len(TrainImgLoader), 
                        (time.time() - start_time) * (len(TrainImgLoader) * (args.epochs - epoch) - batch_idx), optimizer.param_groups[0]["lr"]) )
                #logger.info('losses: {}'.format(list(losses.items())))
                #for lk, lv in losses.items():
                #    writer.add_scalar(lk, lv, epoch * len(TrainImgLoader) + batch_idx)
                total_train_loss += loss

        #----------------------------validation--------------------------
        total_val_loss = 0
        for batch_idx, sample in enumerate(ValImgLoader):
            start_time = time.time()

            scalar_outputs_psmnet, img_outputs_psmnet = test(model, cfg, args, optimizer, sample, critiron)
            loss = scalar_outputs_psmnet['epe loss']
            
            if batch_idx%args.log_freq == 0:
                    global_step = batch_idx * epoch
                    # Update PSMNet images
                    save_images(summary_writer, 'test_ActiveStereo', img_outputs_psmnet, global_step)
                    # Update PSMNet losses
                    save_scalars(summary_writer, 'test_ActiveStereo', scalar_outputs_psmnet, global_step)

            if main_process(args):
                #logger.info('%s: %s' % (args.savemodel.strip('/').split('/')[-1], args.devices))
                if batch_idx%args.log_freq == 0:
                    logger.info('Epoch %d Iter %d/%d validation loss = %.3f , time = %.2f; Epoch time: %.3fs, Left time: %.3fs, lr: %.6f' % (
                        epoch, 
                        batch_idx, len(TrainImgLoader), loss, time.time() - start_time, (time.time() - start_time) * len(TrainImgLoader), 
                        (time.time() - start_time) * (len(TrainImgLoader) * (args.epochs - epoch) - batch_idx), optimizer.param_groups[0]["lr"]) )
                #logger.info('losses: {}'.format(list(losses.items())))
                #for lk, lv in losses.items():
                #    writer.add_scalar(lk, lv, epoch * len(TrainImgLoader) + batch_idx)
                total_val_loss += loss


        if main_process(args):
            logger.info('epoch %d total training loss = %.3f' % (epoch, total_train_loss / len(TrainImgLoader)))
            savefilename = args.logdir + '/finetune_' + str(epoch) + '.tar'
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'train_loss': total_train_loss / len(TrainImgLoader),
                'optimizer': optimizer.state_dict()
            }, savefilename)
            logger.info('Snapshot {} epoch in {}'.format(epoch, args.logdir))

            if total_val_loss/len(ValImgLoader) < best_loss:
                best_loss = total_val_loss/len(ValImgLoader)
                savefilename = args.logdir + '/finetune_best.tar'
                torch.save({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'train_loss': total_val_loss / len(ValImgLoader),
                    'optimizer': optimizer.state_dict()
                }, savefilename)
                logger.info('Best Snapshot {} epoch in {}'.format(epoch, args.logdir))







def train(model, cfg, args, optimizer, sample, critiron):
    model.train()
    
    img_L = sample['img_L'].cuda()    # [bs, 1, H, W]
    img_R = sample['img_R'].cuda()   # [bs, 1, H, W]
    disp_gt = sample['img_disp_l'].cuda()
    depth_gt = sample['img_depth_l'].cuda()  # [bs, 1, H, W]
    img_focal_length = sample['focal_length'].cuda()
    img_baseline = sample['baseline'].cuda()
    # Resize the 2x resolution disp and depth back to H * W
    # Note: This step should go before the apply_disparity_cu
    disp_gt = F.interpolate(disp_gt, scale_factor=0.5, mode='nearest',
                             recompute_scale_factor=False)  # [bs, 1, H, W]
    depth_gt = F.interpolate(depth_gt, scale_factor=0.5, mode='nearest',
                             recompute_scale_factor=False)  # [bs, 1, H, W]


    img_disp_r = sample['img_disp_r'].cuda()
    img_disp_r = F.interpolate(img_disp_r, scale_factor=0.5, mode='nearest',
                                recompute_scale_factor=False)
    disp_gt = apply_disparity_cu(img_disp_r, img_disp_r.type(torch.int))  # [bs, 1, H, W]
    del img_disp_r

    mask = (disp_gt < cfg.ARGS.MAX_DISP) * (disp_gt > 0)  # Note in training we do not exclude bg


    #losses = dict()

    # outputs = model(imgL, imgR, disp_L)

    optimizer.zero_grad()
    disp_pred_L, disp_pred_R = model(img_L, img_R)
    #outputs = [torch.unsqueeze(output, 1) for output in outputs]

    loss = critiron(img_L, img_R, disp_pred_L[:,0,:,:])
    loss.backward()
    optimizer.step()

    train_EPE_left = EPE_metric(disp_pred_L[0], disp_gt[0], mask[0])

    scalar_outputs_psmnet = {'epe loss': train_EPE_left.item(), 'network loss': loss.item()}

    err_metrics = compute_err_metric(disp_gt, depth_gt, disp_pred_L, img_focal_length, img_baseline, mask)
    scalar_outputs_psmnet.update(err_metrics)

    pred_disp_err_np = disp_error_img(disp_pred_L[[0]], disp_gt[[0]], mask[[0]])
    pred_disp_err_tensor = torch.from_numpy(np.ascontiguousarray(pred_disp_err_np[None].transpose([0, 3, 1, 2])))

    img_outputs_psmnet = {
        'img_L': img_L,
        'img_R': img_R,
        'disp_gt': disp_gt[[0]].repeat([1, 3, 1, 1]),
        'disp_pred': disp_pred_L[[0]].repeat([1, 3, 1, 1]),
        'disp_err': pred_disp_err_tensor
    }

        # loss = 0.
        # if getattr(cfg, 'DispVolume', True) and cfg.loss_disp:
        #     pass
            # # depth_preds = [torch.squeeze(o, 1) for o in outputs['depth_preds']]

            # disp_loss = 0.
            # # weight = [0.5, 0.7, 1.0]
            # # for i, o in enumerate(depth_preds):
            # #     disp_loss += weight[3 - len(depth_prTrueeds) + i]  * F.smooth_l1_loss(o[mask], disp_true[mask], size_average=True)
            # losses.update(disp_loss=disp_loss)
            # loss += disp_loss
    #losses.update(loss=loss)

    """
    if args.multiprocessing_distributed:
        with torch.no_grad():
            loss_names = []
            all_losses = []
            for k in sorted(losses.keys()):
                loss_names.append(k)
                all_losses.append(losses[k])
            all_losses = torch.stack(all_losses, dim=0)
            dist.all_reduce(all_losses)
            all_losses /= args.ngpus_per_node
            reduced_losses = {k: v.item() for k, v in zip(loss_names, all_losses)}
    else:
        reduced_losses = {k: v.item() for k, v in losses.items()}
    """

    return scalar_outputs_psmnet, img_outputs_psmnet

def test(model, cfg, args, optimizer, sample, critiron):
    model.eval()
    
    img_L = sample['img_L'].cuda()    # [bs, 1, H, W]
    img_R = sample['img_R'].cuda()   # [bs, 1, H, W]
    disp_gt = sample['img_disp_l'].cuda()
    depth_gt = sample['img_depth_l'].cuda()  # [bs, 1, H, W]
    img_focal_length = sample['focal_length'].cuda()
    img_baseline = sample['baseline'].cuda()
    # Resize the 2x resolution disp and depth back to H * W
    # Note: This step should go before the apply_disparity_cu
    disp_gt = F.interpolate(disp_gt, scale_factor=0.5, mode='nearest',
                             recompute_scale_factor=False)  # [bs, 1, H, W]
    depth_gt = F.interpolate(depth_gt, scale_factor=0.5, mode='nearest',
                             recompute_scale_factor=False)  # [bs, 1, H, W]


    img_disp_r = sample['img_disp_r'].cuda()
    img_disp_r = F.interpolate(img_disp_r, scale_factor=0.5, mode='nearest',
                                recompute_scale_factor=False)
    disp_gt = apply_disparity_cu(img_disp_r, img_disp_r.type(torch.int))  # [bs, 1, H, W]
    del img_disp_r

    mask = (disp_gt < cfg.ARGS.MAX_DISP) * (disp_gt > 0)  # Note in training we do not exclude bg


    #losses = dict()

    # outputs = model(imgL, imgR, disp_L)


    disp_pred_L, disp_pred_R = model(img_L, img_R)
    #outputs = [torch.unsqueeze(output, 1) for output in outputs]

    train_EPE_left = EPE_metric(disp_pred_L[0], disp_gt[0], mask[0])

    scalar_outputs_psmnet = {'epe loss': train_EPE_left.item()}

    err_metrics = compute_err_metric(disp_gt, depth_gt, disp_pred_L, img_focal_length, img_baseline, mask)
    scalar_outputs_psmnet.update(err_metrics)

    pred_disp_err_np = disp_error_img(disp_pred_L[[0]], disp_gt[[0]], mask[[0]])
    pred_disp_err_tensor = torch.from_numpy(np.ascontiguousarray(pred_disp_err_np[None].transpose([0, 3, 1, 2])))

    img_outputs_psmnet = {
        'img_L': img_L,
        'img_R': img_R,
        'disp_gt': disp_gt[[0]].repeat([1, 3, 1, 1]),
        'disp_pred': disp_pred_L[[0]].repeat([1, 3, 1, 1]),
        'disp_err': pred_disp_err_tensor
    }

        # loss = 0.
        # if getattr(cfg, 'DispVolume', True) and cfg.loss_disp:
        #     pass
            # # depth_preds = [torch.squeeze(o, 1) for o in outputs['depth_preds']]

            # disp_loss = 0.
            # # weight = [0.5, 0.7, 1.0]
            # # for i, o in enumerate(depth_preds):
            # #     disp_loss += weight[3 - len(depth_prTrueeds) + i]  * F.smooth_l1_loss(o[mask], disp_true[mask], size_average=True)
            # losses.update(disp_loss=disp_loss)
            # loss += disp_loss
    #losses.update(loss=loss)

    """
    if args.multiprocessing_distributed:
        with torch.no_grad():
            loss_names = []
            all_losses = []
            for k in sorted(losses.keys()):
                loss_names.append(k)
                all_losses.append(losses[k])
            all_losses = torch.stack(all_losses, dim=0)
            dist.all_reduce(all_losses)
            all_losses /= args.ngpus_per_node
            reduced_losses = {k: v.item() for k, v in zip(loss_names, all_losses)}
    else:
        reduced_losses = {k: v.item() for k, v in losses.items()}
    """

    return scalar_outputs_psmnet, img_outputs_psmnet

class BatchCollator(object):
    def __init__(self, cfg):
        super(BatchCollator, self).__init__()
        self.cfg = cfg

    def __call__(self, batch):
        transpose_batch = list(zip(*batch))
        ret = dict()


        ret['imgL'] = torch.cat(transpose_batch[0], dim=0)
        
        ret['imgR'] = torch.cat(transpose_batch[1], dim=0)
        ret['disp_L'] = torch.stack(transpose_batch[2], dim=0)
        # print(ret['disp_L'].size())
        ret['norm_L'] = torch.stack(transpose_batch[3], dim=0)
        # print(ret['norm_L'].size())
        return ret

def adjust_learning_rate(optimizer, epoch, step=None, args=None):
    # if epoch > 1 or step is None or step > args.max_warmup_step:
    #     if epoch <= args.lr_scale:
    #         lr = 0.001 / args.ngpus_per_node
    #     else:
    #         lr = 0.0001 / args.ngpus_per_node
    # else:
    #     lr = 0.001 / args.ngpus_per_node
    #     warmup_pro = float(step) / args.max_warmup_step
    #     lr = lr * (warmup_pro + 1./3. * (1. - warmup_pro))
    lr = 4e-4
    if epoch>2:
        lr = 1e-4
    if epoch>4:
        lr = 4e-5
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':
    main()

