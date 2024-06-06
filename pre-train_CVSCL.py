"""
Train CMC with AlexNet
"""
from __future__ import print_function
CUDA_VISIBLE_DEVICES=6,7
import os
import sys
import time
import torch
import torch.backends.cudnn as cudnn
import argparse
import socket

import tensorboard_logger as tb_logger
from torch.utils import data

from torchvision import transforms, datasets
from dataset import RGB2Lab, RGB2YCbCr
from dataset_multi import RandomSampler, ConcatDataset
from util import adjust_learning_rate, AverageMeter

from models.alexnet import MyAlexNetCMC
from models.resnet import MyResNetsCMC
from NCE.NCEAverage import NCEAverage
from NCE.NCECriterion import NCECriterion
from NCE.NCECriterion import NCESoftmaxLoss

from dataset import ImageFolderInstance

try:
    from apex import amp, optimizers
except ImportError:
    pass
"""
TODO: python 3.6 ModuleNotFoundError
"""

def parse_option():

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=10, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=0, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.003, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='160,200', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # resume path
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    # model definition
    parser.add_argument('--model', type=str, default='resnet18v1', choices=['alexnet',
                                                                         'resnet50v1', 'resnet101v1', 'resnet18v1',
                                                                         'resnet50v2', 'resnet101v2', 'resnet18v2',
                                                                         'resnet50v3', 'resnet101v3', 'resnet18v3'])

    parser.add_argument('--softmax', default='softmax', help='using softmax contrastive loss rather than NCE')  # use softmaxCE
    # parser.add_argument('--softmax', action='store_true', help='using softmax contrastive loss rather than NCE')   # use CE
    parser.add_argument('--nce_k', type=int, default=2048)
    parser.add_argument('--nce_t', type=float, default=0.05)
    parser.add_argument('--nce_m', type=float, default=0.5)
    parser.add_argument('--feat_dim', type=int, default=128, help='dim of feat for inner product')

    # dataset
    parser.add_argument('--dataset', type=str, default='casme3', choices=['imagenet100', 'imagenet', 'casme3'])

    # specify folder
    parser.add_argument('--data_folder1', type=str,
                        # default='/home/zhouhl/Documents/data/casme3/casme3_diff_imgs',
                        # default='/home/zhouhl/Documents/Code/tryMer/MacroExpression/RGB/casme3_diff_rgb',
                        # default='/home/zhouhl/Documents/data/casme3/cmc_test/casme3_diff_memae/casme3_diff_rgb_memae',
                        default='/home/zhouhl/Documents/data/casme3/cmc_test/casme3_diff_big/casme3_diff_big_rgb',
                        help='path to data')
    parser.add_argument('--data_folder2', type=str,
                        # default='/home/zhouhl/Documents/data/casme3/casme3_diff_depth',
                        # default='/home/zhouhl/Documents/Code/tryMer/MacroExpression/Depth/casme3_diff_depth',
                        # default='/home/zhouhl/Documents/data/casme3/cmc_test/casme3_diff_memae/casme3_diff_depth_memae',
                        default='/home/zhouhl/Documents/data/casme3/cmc_test/casme3_diff_big/casme3_diff_big_depth',
                        # default='/home/zhouhl/Documents/data/casme3/cmc_test/casme3_diff_big/casme3_diff_big_gray',
                        help='path to data')

    parser.add_argument('--model_path', type=str,
                        # default='/home/zhouhl/Documents/Code/CMC/model_path/casme3_mae',
                        default='/home/zhouhl/Documents/Code/CMC/model_path/casme3_big',
                        help='path to save model')

    parser.add_argument('--tb_path', type=str,
                        # default='/home/zhouhl/Documents/Code/CMC/tb_path/casme3_mae',
                        default='/home/zhouhl/Documents/Code/CMC/tb_path/casme3_big',
                        help='path to tensorboard')

    # add new views
    parser.add_argument('--view', type=str, default='RGB_D', choices=['Lab', 'YCbCr', 'RGB_D', 'RGB_gray'])

    # mixed precision setting
    parser.add_argument('--amp', action='store_true', help='using mixed precision')
    parser.add_argument('--opt_level', type=str, default='O2', choices=['O1', 'O2'])

    # data crop threshold
    parser.add_argument('--crop_low', type=float, default=0.2, help='low area in crop')

    parser.add_argument('--display_id', type=int, default=0, help='display id')
    opt = parser.parse_args()

    if (opt.data_folder1 is None) or (opt.data_folder2 is None) or (opt.model_path is None) or (opt.tb_path is None):
        raise ValueError('one or more of the folders is None: data_folder | model_path | tb_path')

    if opt.dataset == 'casme3':
        if 'alexnet' not in opt.model:
            opt.crop_low = 0.08

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.method = 'softmax' if opt.softmax else 'nce'
    opt.model_name = 'memory_{}_{}_{}_lr_{}_decay_{}_bsz_{}'.format(opt.method, opt.nce_k, opt.model, opt.learning_rate,
                                                                    opt.weight_decay, opt.batch_size)

    if opt.amp:
        opt.model_name = '{}_amp_{}'.format(opt.model_name, opt.opt_level)

    opt.model_name = '{}_view_{}'.format(opt.model_name, opt.view)

    opt.model_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.model_folder):
        os.makedirs(opt.model_folder)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    if not os.path.isdir(opt.data_folder1):
        raise ValueError('data path1 not exist: {}'.format(opt.data_folder1))
    if not os.path.isdir(opt.data_folder2):
        raise ValueError('data path2 not exist: {}'.format(opt.data_folder2))

    return opt


def get_train_loader(args):
    """get the train loader"""
    data_folder1 = os.path.join(args.data_folder1, 'train')
    data_folder2 = os.path.join(args.data_folder2, 'train')

    train_transform1 = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.RandomGrayscale(p=0.2),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.4454, 0.4474, 0.4504], [0.4474, 0.4505, 0.4502]),
    ])
    train_transform2 = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.RandomGrayscale(p=0.2),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.0423, 0.0423, 0.0423], [0.1761, 0.176, 0.1761]),
        # transforms.Normalize([0.4454, 0.4474, 0.4504], [0.4474, 0.4505, 0.4502]),
    ])

    dataset1 = ImageFolderInstance(data_folder1, transform=train_transform1)
    dataset2 = ImageFolderInstance(data_folder2, transform=train_transform2)

    s = torch.randperm(len(dataset1)).tolist()
    # print(s)
    # sampler = RandomSampler(dataset1, s)
    # sampler1 = RandomSampler(dataset2, s)

    """
    # 两个dataset拼起来，放到一个dataloader中
    dataset1 = datasets.ImageFolder(data_folder1, transform=train_transform1)
    dataset2 = datasets.ImageFolder(data_folder2, transform=train_transform2)

    train_loader = torch.utils.data.DataLoader(
        ConcatDataset(dataset1, dataset2),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True)
    """
    data_loader = torch.utils.data.DataLoader(
        ConcatDataset(dataset1, dataset2),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True)
    # data_loader1 = data.DataLoader(dataset1, args.batch_size,
    #                                num_workers=args.num_workers, sampler=RandomSampler(dataset1, s),
    #                                shuffle=False, collate_fn=None, pin_memory=True)
    #
    # data_loader2 = data.DataLoader(dataset2, args.batch_size,
    #                                num_workers=args.num_workers, sampler=RandomSampler(dataset2, s),
    #                                shuffle=False, collate_fn=None, pin_memory=True)
    #
    # # # num of samples
    n_data = len(dataset1)
    print('number of samples: {}'.format(n_data))

    # return data_loader1, data_loader2, n_data
    return data_loader, n_data


def set_model(args, n_data):
    # set the model
    if args.model == 'alexnet':
        model = MyAlexNetCMC(args.feat_dim)
    elif args.model.startswith('resnet'):
        model = MyResNetsCMC(args.model)
    else:
        raise ValueError('model not supported yet {}'.format(args.model))

    contrast = NCEAverage(args.feat_dim, n_data, args.nce_k, args.nce_t, args.nce_m, args.softmax)
    criterion_rgb = NCESoftmaxLoss() if args.softmax else NCECriterion(n_data)
    criterion_depth = NCESoftmaxLoss() if args.softmax else NCECriterion(n_data)

    if torch.cuda.is_available():
        model = model.cuda()
        contrast = contrast.cuda()
        criterion_rgb = criterion_rgb.cuda()
        criterion_depth = criterion_depth.cuda()
        cudnn.benchmark = True

    return model, contrast, criterion_rgb, criterion_depth


def set_optimizer(args, model):
    # return optimizer
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    # optimizer = torch.optim.Adam(model.parameters(),
    #                              lr=args.learning_rate,
    #                              weight_decay=args.weight_decay,
    #                              betas=(args.beta1, args.beta2))
    return optimizer


def train(epoch, train_loader, model, contrast, criterion_rgb, criterion_depth, optimizer, opt):
    """
    one epoch training
    """
    model.train()
    contrast.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    rgb_loss_meter = AverageMeter()
    depth_loss_meter = AverageMeter()
    rgb_prob_meter = AverageMeter()
    depth_prob_meter = AverageMeter()

    end = time.time()
    for idx, dataloader in enumerate(train_loader):
        data_time.update(time.time() - end)

        bsz = dataloader[0][0].size(0)
        inputs = dataloader[0][0]
        _ = dataloader[0][1]
        index = dataloader[0][2]

        # bsz1 = dataloader[1][0].size(0)
        inputs1 = dataloader[1][0]
        inputs1 = inputs1[:, 0:1, :, :]  # 取单通道
        _1 = dataloader[1][1]
        index1 = dataloader[1][2]

        new_inputs = torch.cat((inputs, inputs1), 1)

        if torch.cuda.is_available():
            index = index.cuda()
            # inputs = inputs.cuda()
            # index1 = index1.cuda()
            # inputs1 = inputs1.cuda()
            new_inputs = new_inputs.cuda()

        # ===================forward=====================
        feat_rgb, feat_depth = model(new_inputs)

        # print("feature_rgb & depth")
        # print(feat_rgb.size(), feat_rgb.shape, feat_rgb, feat_depth.size(), feat_depth.shape,  feat_depth)
        out_rgb, out_depth = contrast(feat_rgb, feat_depth, index)
        # print("contrast rgb & depth")

        rgb_loss = criterion_rgb(out_rgb)
        depth_loss = criterion_depth(out_depth)
        rgb_prob = out_rgb[:, 0].mean()
        depth_prob = out_depth[:, 0].mean()

        loss = rgb_loss + depth_loss
        # print("loss")

        # ===================backward=====================
        optimizer.zero_grad()
        if opt.amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()
        # print("backward")

        # ===================meters=====================
        losses.update(loss.item(), bsz)
        rgb_loss_meter.update(rgb_loss.item(), bsz)
        rgb_prob_meter.update(rgb_prob.item(), bsz)
        depth_loss_meter.update(depth_loss.item(), bsz)
        depth_prob_meter.update(depth_prob.item(), bsz)

        torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'rgb_p {rgbprobs.val:.3f} ({rgbprobs.avg:.3f})\t'
                  'depth_p {depthprobs.val:.3f} ({depthprobs.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, rgbprobs=rgb_prob_meter,
                   depthprobs=depth_prob_meter))
            print(out_rgb.shape, out_depth.shape)
            sys.stdout.flush()

    return rgb_loss_meter.avg, rgb_prob_meter.avg, depth_loss_meter.avg, depth_prob_meter.avg


def main():

    # parse the args
    args = parse_option()

    # set the loader
    train_loader, n_data = get_train_loader(args)

    # set the model
    model, contrast, criterion_depth, criterion_rgb = set_model(args, n_data)

    # set the optimizer
    optimizer = set_optimizer(args, model)

    # set mixed precision
    if args.amp:
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level)

    # optionally resume from a checkpoint
    args.start_epoch = 1
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            args.start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            contrast.load_state_dict(checkpoint['contrast'])
            if args.amp and checkpoint['opt'].amp:
                print('==> resuming amp state_dict')
                amp.load_state_dict(checkpoint['amp'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            del checkpoint
            torch.cuda.empty_cache()
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # tensorboard
    logger = tb_logger.Logger(logdir=args.tb_folder, flush_secs=2)

    # routine
    for epoch in range(args.start_epoch, args.epochs + 1):

        adjust_learning_rate(epoch, args, optimizer)
        print("==> training...")

        time1 = time.time()
        rgb_loss, rgb_prob, depth_loss, depth_prob = train(epoch, train_loader, model,
                                                           contrast, criterion_rgb, criterion_depth, optimizer, args)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # tensorboard logger
        logger.log_value('rgb_loss', rgb_loss, epoch)
        logger.log_value('rgb_prob', rgb_prob, epoch)
        logger.log_value('depth_loss', depth_loss, epoch)
        logger.log_value('depth_prob', depth_prob, epoch)

        # save model
        if epoch % args.save_freq == 0:
            print('==> Saving...')
            state = {
                'opt': args,
                'model': model.state_dict(),
                'contrast': contrast.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }
            if args.amp:
                state['amp'] = amp.state_dict()
            save_file = os.path.join(args.model_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)
            # help release GPU memory
            del state

        torch.cuda.empty_cache()


if __name__ == '__main__':
    CUDA_VISIBLE_DEVICES = 6, 7
    main()
