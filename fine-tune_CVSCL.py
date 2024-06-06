import os, datetime, random
import scipy.io as scio
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torchvision
import sys
import time
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn as nn
import argparse
from torch.utils.data import distributed

from dataset_multi import ConcatDataset

from torchvision import transforms

from models.alexnet import MyAlexNetCMC
from models.resnet import MyResNetsCMC
from models.LinearModel import LinearClassifierAlexNet, LinearClassifierResNet
from models.LinearModel_Att import  LinearClassifierResNet as LinearClassifierResNet_att

from Datasets import MEGC2019_SI as MEGC2019
import Metrics as metrics
import LossFunctions


def arg_process():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataversion1', default='3_rgb_half-oversampling', help='the version of input data')
    parser.add_argument('--dataversion2', default='3_depth_half-oversampling', help='the version of input data')
    parser.add_argument('--n_label', type=int, default=4, help='the number of classes')
    parser.add_argument('--epochs', type=int, default=30, help='the number of training epochs')
    parser.add_argument('--batchsize', type=int, default=128, help='the batch size for training')
    parser.add_argument('--gpuid', default='cuda:2', help='the gpu index for training')

    parser.add_argument('--learning_rate', type=float, default=0.0001, help='the learning rate for training')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')

    # model definition
    parser.add_argument('--model', type=str, default='resnet18v1', choices=['alexnet',
                                                                            'resnet50v1', 'resnet101v1', 'resnet18v1',
                                                                            'resnet50v2', 'resnet101v2', 'resnet18v2',
                                                                            'resnet50v3', 'resnet101v3',
                                                                            'resnet18v3'])  # 要改
    parser.add_argument('--attention', type=str, default=None, choices=['att'])

    parser.add_argument('--model_path', type=str,
                        # default='/home/zhouhl/Documents/Code/CMC/model_path/casme3_mae/memory_softmax_2048_alexnet_lr_0.003_decay_0.0001_bsz_128_view_RGB_D/ckpt_epoch_110.pth',
                        # default='/home/zhouhl/Documents/Code/CMC/model_path/casme3_mae/memory_softmax_2048_resnet50v1_lr_0.003_decay_0.0001_bsz_128_view_RGB_D/ckpt_epoch_110.pth',
                        # default='/home/zhouhl/Documents/Code/CMC/model_path/casme3_big/memory_nce_2048_resnet50v1_lr_0.003_decay_0.0001_bsz_128_view_RGB_D/ckpt_epoch_120.pth',
                        # default='/home/zhouhl/Documents/Code/CMC/model_path/casme3_big/memory_nce_2048_resnet18v1_lr_0.003_decay_0.0001_bsz_128_view_RGB_D/ckpt_epoch_190.pth',  # rgb_d res18
                        # default='/home/zhouhl/Documents/Code/CMC/model_path/casme3_big/memory_softmax_2048_resnet18v1_lr_0.003_decay_0.0001_bsz_128_view_RGB_D_weakAug/ckpt_epoch_240.pth',  # rgb_d res18 softmaxCE
                        default='/home/zhouhl/Documents/Code/CMC/model_path/casme3_big/memory_softmax_4096_resnet18v1_lr_0.003_decay_0.0001_bsz_128_view_RGB_D_centercrop/ckpt_epoch_120.pth',

                        # default='/home/zhouhl/Documents/Code/CMC/model_path/casme3_linear_big_all/calibrated_memory_nce_2048_resnet50v1_lr_0.003_decay_0.0001_bsz_128_view_RGB_D_bsz_256_lr_0.0001_decay_0.0001_view_RGBD/ckpt_epoch_50.pth',
                        # default='/home/zhouhl/Documents/Code/CMC/model_path/casme3_big/memory_nce_2048_alexnet_lr_0.003_decay_0.0001_bsz_128_view_RGB_D/ckpt_epoch_100.pth',  # rgb_d alex
                        # default='/home/zhouhl/Documents/Code/CMC/model_path/casme3_big/memory_nce_2048_resnet50v1_lr_0.003_decay_0.0001_bsz_128_view_RGB_gray/ckpt_epoch_120.pth',  # rgb_gray
                        help='the model to test')
    parser.add_argument('--layer', type=int, default=5, help='which layer to evaluate')  # 要改
    parser.add_argument('--model_name', type=str, default='RGBDcls_res18_softmaxCE_aug_half-oversampling', help='the model name')
    parser.add_argument('--lossfunction', default='crossentropy', help='the loss functions')
    args = parser.parse_args()
    return args


def set_model(args):
    if args.model.startswith('alexnet'):
        model = MyAlexNetCMC()
        classifier = LinearClassifierAlexNet(layer=args.layer, n_label=args.n_label, pool_type='max')
    elif args.model.startswith('resnet'):
        model = MyResNetsCMC(args.model)
        if args.model.endswith('v1'):
            if args.attention == 'att':
                classifier = LinearClassifierResNet_att(args.layer, args.n_label, 'avg', 1)
            else:
                classifier = LinearClassifierResNet(args.layer, args.n_label, 'avg', 1)
        elif args.model.endswith('v2'):
            classifier = LinearClassifierResNet(args.layer, args.n_label, 'avg', 2)
        elif args.model.endswith('v3'):
            classifier = LinearClassifierResNet(args.layer, args.n_label, 'avg', 4)
        else:
            raise NotImplementedError('model not supported {}'.format(args.model))
    else:
        raise NotImplementedError('model not supported {}'.format(args.model))

    # load pre-trained model
    print('==> loading pre-trained model')
    ckpt = torch.load(args.model_path)
    model.load_state_dict(ckpt['model'])
    print("==> loaded checkpoint '{}' (epoch {})".format(args.model_path, ckpt['epoch']))
    print('==> done')

    model = model.cuda()
    classifier = classifier.cuda()

    model.eval()

    loss_function = args.lossfunction
    if loss_function == 'crossentropy':
        criterion = nn.CrossEntropyLoss().cuda(args.gpuid)
    elif loss_function == 'focal':
        criterion = LossFunctions.FocalLoss(class_num=4).cuda(args.gpuid)
    elif loss_function == 'balanced':
        criterion = LossFunctions.BalancedLoss(class_num=4).cuda(args.gpuid)
    elif loss_function == 'cosine':
        criterion = LossFunctions.CosineLoss().cuda(args.gpuid)

    return model, classifier, criterion


def set_optimizer(args, classifier):
    optimizer = optim.SGD(classifier.parameters(),
                          lr=args.learning_rate,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)
    # optimizer = torch.optim.Adam(classifier.parameters(),
    #                              lr=args.learning_rate,
    #                              weight_decay=args.weight_decay,
    #                              betas=(args.beta1, args.beta2))
    return optimizer


def train_model(model, classifier, dataloaders, criterion, optimizer, device='gpu', num_epochs=25):
    since = time.time()
    # best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    for epoch in range(num_epochs):
        print('\tEpoch {}/{}'.format(epoch, num_epochs - 1))
        print('\t' + '-' * 10)
        # Each epoch has a training
        model.eval()
        classifier.train()  # Set model to training mode

        running_loss = 0.0
        running_corrects = 0
        # Iterate over data
        for j, samples in enumerate(dataloaders):
            inputs, class_labels = samples[0]["data"], samples[0]["class_label"]

            inputs1 = samples[1]["data"]
            inputs1 = inputs1[:, 0:1, :, :]  # 取单通道

            new_inputs = torch.cat((inputs, inputs1), 1)
            inputs = torch.FloatTensor(new_inputs).to(device)
            class_labels = class_labels.to(device)

            with torch.no_grad():
                feat_rgb, feat_depth = model(inputs, 5)  # 第6层
                feat = torch.cat((feat_rgb.detach(), feat_depth.detach()), dim=1)
            # forward to get model outputs and calculate loss
            output_class = classifier(feat)
            loss = criterion(output_class, class_labels)
            # backward
            loss.backward()
            optimizer.step()
            # update learning_rate
            scheduler.step()
            # statistics
            _, predicted = torch.max(output_class.data, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(predicted == class_labels)

        epoch_loss = running_loss / len(dataloaders.dataset)
        epoch_acc = running_corrects.double() / len(dataloaders.dataset)
        print('\t{} Loss: {:.4f} Acc: {:.4f}'.format('Train', epoch_loss, epoch_acc))

    time_elapsed = time.time() - since
    print('\tTraining complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return model, classifier


def test_model(model, classifier, dataloaders, device):
    model.eval()
    classifier.eval()
    num_samples = len(dataloaders.dataset.datasets[0])
    predVec = torch.zeros(num_samples)
    labelVec = torch.zeros(num_samples)
    start_idx = 0
    end_idx = 0
    for j, samples in enumerate(dataloaders):
        inputs, class_labels = samples[0]['data'], samples[0]['class_label']

        inputs1 = samples[1]["data"]
        inputs1 = inputs1[:, 0:1, :, :]  # 取单通道

        new_inputs = torch.cat((inputs, inputs1), 1)
        inputs = torch.FloatTensor(new_inputs).to(device)

        # update the index of ending point
        end_idx = start_idx + inputs.shape[0]

        feat_rgb, feat_depth = model(inputs, 5)  # 第6层
        # print(feat_rgb.shape, feat_depth.shape)
        feat = torch.cat((feat_rgb.detach(), feat_depth.detach()), dim=1)
        # print(feat.shape)
        output_class = classifier(feat)
        # print(output_class.shape)

        _, predicted = torch.max(output_class.data, 1)
        predVec[start_idx:end_idx] = predicted
        labelVec[start_idx:end_idx] = class_labels
        # update the starting point
        start_idx += inputs.shape[0]
    return predVec, labelVec


def main():
    """
    Goal: process images by file lists, evaluating the datasize with different model size
    Version: 5.0
    """
    print('PyTorch Version: ', torch.__version__)
    print('Torchvision Version: ', torchvision.__version__)
    now = datetime.datetime.now()
    random.seed(1)
    torch.manual_seed(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args = arg_process()
    runFileName = sys.argv[0].split('.')[0]
    # need to modify according to the enviroment
    version_rgb = args.dataversion1  # view1,rgb
    version_depth = args.dataversion2  # view2,depth
    num_epochs = args.epochs
    lr = args.learning_rate
    batch_size = args.batchsize
    model_name = args.model_name
    model_nn = args.model

    logPath = os.path.join('result', model_name + '_log.txt')
    data_transforms1 = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.4454, 0.4474, 0.4504], [0.4474, 0.4505, 0.4502]),
    ])
    data_transforms2 = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.0423, 0.0423, 0.0423], [0.1761, 0.176, 0.1761]),  # depth
        # transforms.Normalize([0.4454, 0.4474, 0.4504], [0.4474, 0.4505, 0.4502]),
    ])
    # move to GPU
    print(torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # obtian the subject information in LOSO
    verFolder_rgb = 'v_{}'.format(version_rgb)
    verFolder_depth = 'v_{}'.format(version_depth)
    filePath = os.path.join('data', verFolder_rgb, 'subName.txt')
    subjects = []
    with open(filePath, 'r') as f:
        for textline in f:
            texts = textline.strip('\n')
            subjects.append(texts)
    # predicted and label vectors
    preds_db = {}
    preds_db['casme3'] = torch.tensor([])
    preds_db['smic'] = torch.tensor([])
    preds_db['samm'] = torch.tensor([])
    preds_db['all'] = torch.tensor([])
    labels_db = {}
    labels_db['casme3'] = torch.tensor([])
    labels_db['smic'] = torch.tensor([])
    labels_db['samm'] = torch.tensor([])
    labels_db['all'] = torch.tensor([])
    # open the log file and begin to record
    log_f = open(logPath, 'a')
    log_f.write('{}\n'.format(now))
    log_f.write('-' * 80 + '\n')
    log_f.write('-' * 80 + '\n')
    log_f.write('Results:\n')
    allRST = []

    time_s = time.time()
    for subject in subjects:
        print('Subject Name: {}'.format(subject))
        print('---------------------------')
        # random.seed(1)
        # setup a dataloader for training
        imgDir_rgb = os.path.join('data', verFolder_rgb, '{}_train.txt'.format(subject))
        imgDir_depth = os.path.join('data', verFolder_depth, '{}_train.txt'.format(subject))
        print(imgDir_rgb, imgDir_depth)

        image_db_train_rgb = MEGC2019(imgList=imgDir_rgb, transform=data_transforms1)
        image_db_train_depth = MEGC2019(imgList=imgDir_depth, transform=data_transforms2)
        image_db_train = ConcatDataset(image_db_train_rgb, image_db_train_depth)  # 训练集
        dataloader_train = torch.utils.data.DataLoader(image_db_train, batch_size=batch_size, shuffle=True,
                                                       num_workers=1)

        # Initialize the model
        print('\tCreating deep model....')
        # set the model
        model, classifier, criterion = set_model(args)

        # set optimizer
        optimizer = set_optimizer(args, classifier)

        # Train and evaluate
        model_ft, classifier_ft = train_model(model, classifier, dataloader_train, criterion, optimizer, device,
                                              num_epochs=num_epochs)
        # torch.save(model_ft, os.path.join('data', 'model_s{}.pth').format(subject))

        # Test model
        imgDir_rgb = os.path.join('data', verFolder_rgb, '{}_test.txt'.format(subject))
        imgDir_depth = os.path.join('data', verFolder_depth, '{}_test.txt'.format(subject))
        image_db_test_rgb = MEGC2019(imgList=imgDir_rgb, transform=data_transforms1)
        image_db_test_depth = MEGC2019(imgList=imgDir_depth, transform=data_transforms2)
        image_db_test = ConcatDataset(image_db_test_rgb, image_db_test_depth)  # 测试集
        dataloaders_test = torch.utils.data.DataLoader(image_db_test, batch_size=batch_size, shuffle=False,
                                                       num_workers=1)

        preds, labels = test_model(model_ft, classifier_ft, dataloaders_test, device)
        preds_np = np.array(preds)
        labels_np = np.array(labels)
        allRST.append([preds_np, labels_np])
        acc = torch.sum(preds == labels).double() / len(preds)
        print('\tSubject {} has the accuracy:{:.4f}\n'.format(subject, acc))
        print('---------------------------\n')
        log_f.write('\tSubject {} has the accuracy:{:.4f}\n'.format(subject, acc))

        # saving the subject results
        preds_db['all'] = torch.cat((preds_db['all'], preds), 0)
        labels_db['all'] = torch.cat((labels_db['all'], labels), 0)
        if subject.find('sp') != -1:
            preds_db['casme3'] = torch.cat((preds_db['casme3'], preds), 0)
            labels_db['casme3'] = torch.cat((labels_db['casme3'], labels), 0)
        else:
            if subject.find('s') != -1:
                preds_db['smic'] = torch.cat((preds_db['smic'], preds), 0)
                labels_db['smic'] = torch.cat((labels_db['smic'], labels), 0)
            else:
                preds_db['samm'] = torch.cat((preds_db['samm'], preds), 0)
                labels_db['samm'] = torch.cat((labels_db['samm'], labels), 0)
    time_e = time.time()
    hours, rem = divmod(time_e - time_s, 3600)
    miniutes, seconds = divmod(rem, 60)

    # evaluate all data
    allRST = np.array(allRST)
    rstPath = os.path.join('result', model_name + '_layer' + str(args.layer) + '_lr' + str(lr) + '_rst.mat')
    scio.savemat(rstPath, {'Xia': allRST})
    eval_acc = metrics.accuracy()
    eval_f1 = metrics.f1score()
    acc_w, acc_uw = eval_acc.eval(preds_db['all'], labels_db['all'])
    f1_w, f1_uw = eval_f1.eval(preds_db['all'], labels_db['all'])
    print('\nThe dataset has the UAR and UF1:{:.4f} and {:.4f}'.format(acc_uw, f1_uw))
    log_f.write('\nOverall:\n\tthe UAR and UF1 of all data are {:.4f} and {:.4f}\n'.format(acc_uw, f1_uw))
    # casme3
    if preds_db['casme3'].nelement() != 0:
        acc_w, acc_uw = eval_acc.eval(preds_db['casme3'], labels_db['casme3'])
        f1_w, f1_uw = eval_f1.eval(preds_db['casme3'], labels_db['casme3'])
        print('\nThe casme3 dataset has the ACC:{:.4f} and UAR and UF1:{:.4f} and {:.4f}'.format(acc_w, acc_uw, f1_uw))
        log_f.write('\tthe UAR and UF1 of casme3 are {:.4f} and {:.4f}\n'.format(acc_uw, f1_uw))
    # smic
    if preds_db['smic'].nelement() != 0:
        acc_w, acc_uw = eval_acc.eval(preds_db['smic'], labels_db['smic'])
        f1_w, f1_uw = eval_f1.eval(preds_db['smic'], labels_db['smic'])
        print('\nThe smic dataset has the UAR and UF1:{:.4f} and {:.4f}'.format(acc_uw, f1_uw))
        log_f.write('\tthe UAR and UF1 of smic are {:.4f} and {:.4f}\n'.format(acc_uw, f1_uw))
    # samm
    if preds_db['samm'].nelement() != 0:
        acc_w, acc_uw = eval_acc.eval(preds_db['samm'], labels_db['samm'])
        f1_w, f1_uw = eval_f1.eval(preds_db['samm'], labels_db['samm'])
        print('\nThe samm dataset has the UAR and UF1:{:.4f} and {:.4f}'.format(acc_uw, f1_uw))
        log_f.write('\tthe UAR and UF1 of samm are {:.4f} and {:.4f}\n'.format(acc_uw, f1_uw))
    # writing parameters into log file
    print('\tNetname:{}, Dataversion:{}, Modelnn:{}\n\tLearning rate:{}, Epochs:{}, Batchsize:{}.'
          .format(model_name, version_rgb + '+' + version_depth, model_nn, lr, num_epochs, batch_size))
    print('\tElapsed time: {:0>2}:{:0>2}:{:05.2f}'.format(int(hours), int(miniutes), seconds))
    log_f.write('\nOverall:\n\tthe weighted and unweighted accuracy of all data are {:.4f} and {:.4f}\n'.format(acc_w, acc_uw))
    log_f.write('\nSetting:\n\tNetname:{}, Dataversion:{}, Modelnn:{}\n\tLearning rate:{}, Epochs:{}, Batchsize:{}.\n'.format(
                                                                                                                model_name,
                                                                                                                version_rgb + '+' + version_depth,
                                                                                                                model_nn, lr,
                                                                                                                num_epochs,
                                                                                                                batch_size))
    # # save subject's results
    # torch.save({
    #     'predicts':preds_db,
    #     'labels':labels_db,
    #     'weight_acc':acc_w,
    #     'unweight_acc':acc_uw
    # },resultPath)
    log_f.write('-' * 80 + '\n')
    log_f.write('-' * 80 + '\n')
    log_f.write('\n')
    log_f.close()


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    main()
