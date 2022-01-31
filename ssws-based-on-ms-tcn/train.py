import adabound
import argparse
import os
import pandas as pd
import random
import sys
import time
import torch
import torch.optim as optim
import yaml

from addict import Dict
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from libs import models
from libs.loss_fn import ActionSegmentationLoss
from libs.checkpoint import save_checkpoint, resume
from libs.class_weight import get_class_weight
from libs.dataset import ActionSegmentationDataset, collate_fn
from libs.metric import ScoreMeter, AverageMeter
from libs.transformer import TempDownSamp, ToTensor
from utils.class_id_map import get_id2class_map, get_n_classes


def get_arguments():
    '''
    parse all the arguments from command line inteface
    return a list of parsed arguments
    '''
    # 运行的命令 python3 train.py --config ./result/gtea/ms-tcn/split3/config.yaml 
    # 可以更换的参数 python3 train.py --config ./result/50salads(50salads/gtea/breakfast)/ms-tcn/split1(split1/split2/split3/split4/split5/)/config.yaml 
    # 三个数据集 (50salads/gtea/breakfast) 建议只跑gtea 和 50salads 这两个数据集 因为breakfast数据集时间太长
    # 可更换的split 每个数据集都有几个不同的split
    # --resume 是意外停止时 恢复训练

    parser = argparse.ArgumentParser(
        description='train a network for action recognition')
    parser.add_argument('--config', type=str, help='path of a config file',default="./result/gtea/ms-tcn/split2/config.yaml")
    parser.add_argument('--resume', action='store_true',
                        help='Add --resume option if you start training from checkpoint.')
    parser.add_argument('--seed', type=int, default=1538574472)
    return parser.parse_args()


def train(train_loader, model, criterion, optimizer, epoch, config, device):
    losses = AverageMeter('Loss', ':.4e')

    # switch training mode
    model.train() 

    for i, sample in enumerate(train_loader):

        x = sample['feature']
        t = sample['label']

        x = x.to(device)   # shape  1 2048 643  n 2048 T
        t = t.to(device)  # 指定GPU  1 643 n g

        batch_size = x.shape[0]

        # compute output and loss
        output = model(x)

        if isinstance(output, list):  # loss函数
            loss = 0.0
            for out in output:
                loss += criterion(out, t, x)
        else:
            loss = criterion(output, t, x)

        # record loss
        losses.update(loss.item(), batch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return losses.avg


def validate(val_loader, model, criterion, config, device):
    losses = AverageMeter('Loss', ':.4e')
    scores = ScoreMeter(
        id2class_map=get_id2class_map(
            config.dataset, dataset_dir=config.dataset_dir),
        thresholds=config.thresholds
    )

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for sample in val_loader:
            x = sample['feature']
            t = sample['label']
            x = x.to(device)
            t = t.to(device)

            batch_size = x.shape[0]

            # compute output and loss
            output = model(x)

            loss = criterion(output, t, x)

            # measure accuracy and record loss
            losses.update(loss.item(), batch_size)

            # measure pixel accuracy, mean accuracy, Frequency Weighted IoU, mean IoU, class IoU
            pred = output.data.max(1)[1].squeeze(0).cpu().numpy()
            gt = t.data.cpu().squeeze(0).numpy()
            scores.update(pred, gt)

    acc, edit_score, f1s = scores.get_scores()

    return losses.avg, acc, edit_score, f1s


def main():
    # argparser
    args = get_arguments()

    # configuration
    CONFIG = Dict(yaml.safe_load(open(args.config)))                # 获取config文件中的各项参数

    seed = args.seed                                                # 设置随机种子 保证每次训练的初始化时一样的
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True                       #将这个flag置为True的话，每次返回的卷积算法将是确定的，Torch 的随机种子为固定值的话，可以保证每次运行网络相同输入的输出是固定的
    torch.cuda.set_device(CONFIG.device)                            # 设置模型在那张显卡上跑

    # cpu or cuda
    device = 'cuda' if torch.cuda.is_available() else 'cpu'         # 设置gpu
    if device == 'cuda':
        torch.backends.cudnn.benchmark = True
    else:
        print('You have to use GPUs because training CNN is computationally expensive.')
        sys.exit(1)

    # Dataloader
    # Temporal downsampling is applied to only videos in 50Salads
    print("Dataset: {}\tSplit: {}".format(CONFIG.dataset, CONFIG.split))  # 只有在50Salads时 为了保证FPS的一致性 采取抽帧的操作
    print(
        "Batch Size: {}\tNum in channels: {}\tNum Workers: {}"
        .format(CONFIG.batch_size, CONFIG.in_channel, CONFIG.num_workers)
    )

    downsamp_rate = 2 if CONFIG.dataset == '50salads' else 1    

    train_data = ActionSegmentationDataset(                          
        CONFIG.dataset,
        transform=Compose([
            ToTensor(),
            TempDownSamp(downsamp_rate),
        ]),
        mode='trainval' if not CONFIG.param_search else 'training',
        split=CONFIG.split,
        dataset_dir=CONFIG.dataset_dir,
        csv_dir=CONFIG.csv_dir
    )                                                                  

    train_loader = DataLoader(                                            # 数据加载 
        train_data,
        batch_size=CONFIG.batch_size,
        shuffle=True,
        num_workers=CONFIG.num_workers,
        drop_last=True if CONFIG.batch_size > 1 else False,
        collate_fn=collate_fn
    )

    # if you do validation to determine hyperparams
    if CONFIG.param_search:                      # 设置为True 表示在训练中 每训练一轮测试一次
        val_data = ActionSegmentationDataset(
            CONFIG.dataset,
            transform=Compose([
                ToTensor(),
                TempDownSamp(downsamp_rate),
            ]),
            mode='validation',
            split=CONFIG.split,
            dataset_dir=CONFIG.dataset_dir,
            csv_dir=CONFIG.csv_dir
        )

        val_loader = DataLoader(
            val_data,
            batch_size=1,
            shuffle=False,
            num_workers=CONFIG.num_workers
        )

    # load model
    print('\n------------------------Loading Model------------------------\n')

    n_classes = get_n_classes(CONFIG.dataset, dataset_dir=CONFIG.dataset_dir)    # 得到类别数目

    print('Multi Stage TCN will be used as a model.')
    print('stages: {}\tn_features: {}\tn_layers of dilated TCN: {}\tkernel_size of ED-TCN: {}'
          .format(CONFIG.stages, CONFIG.n_features, CONFIG.dilated_n_layers, CONFIG.kernel_size))
    model = models.MultiStageTCN(                      #模型的初始化
        in_channel=CONFIG.in_channel,
        n_classes=n_classes,
        stages=CONFIG.stages,
        n_features=CONFIG.n_features,
        dilated_n_layers=CONFIG.dilated_n_layers,
        kernel_size=CONFIG.kernel_size
    )

    # send the model to cuda/cpu
    model.to(CONFIG.device)

    if CONFIG.optimizer == 'Adam':  # 选择优化器
        print(CONFIG.optimizer + ' will be used as an optimizer.')
        optimizer = optim.Adam(model.parameters(), lr=CONFIG.learning_rate)
    elif CONFIG.optimizer == 'SGD':
        print(CONFIG.optimizer + ' will be used as an optimizer.')
        optimizer = optim.SGD(
            model.parameters(),
            lr=CONFIG.learning_rate,
            momentum=CONFIG.momentum,
            dampening=CONFIG.dampening,
            weight_decay=CONFIG.weight_decay,
            nesterov=CONFIG.nesterov
        )
    elif CONFIG.optimizer == 'AdaBound':
        print(CONFIG.optimizer + ' will be used as an optimizer.')
        optimizer = adabound.AdaBound(
            model.parameters(),
            lr=CONFIG.learning_rate,
            final_lr=CONFIG.final_lr,
            weight_decay=CONFIG.weight_decay
        )
    else:
        print('There is no optimizer which suits to your option.')
        sys.exit(1)

    # learning rate scheduler
    if CONFIG.scheduler == 'onplateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=CONFIG.lr_patience
        )
    else:
        scheduler = None

    # resume if you want
    columns = ['epoch', 'lr', 'train_loss']

    # if you do validation to determine hyperparams
    if CONFIG.param_groups:
        columns += ['val_loss', 'acc', 'edit']
        columns += ["f1s@{}".format(CONFIG.thresholds[i])
                    for i in range(len(CONFIG.thresholds))]

    begin_epoch = 0
    best_loss = 100
    log = pd.DataFrame(columns=columns)
    if args.resume:
        if os.path.exists(os.path.join(CONFIG.result_path, 'checkpoint.pth')):
            print('loading the checkpoint...')
            checkpoint = resume(
                CONFIG.result_path, model, optimizer, scheduler)
            begin_epoch, model, optimizer, best_loss, scheduler = checkpoint
            print('training will start from {} epoch'.format(begin_epoch))
        else:
            print("there is no checkpoint at the result folder")
        if os.path.exists(os.path.join(CONFIG.result_path, 'log.csv')):
            print('loading the log file...')
            log = pd.read_csv(os.path.join(CONFIG.result_path, 'log.csv'))
        else:
            print("there is no log file at the result folder.")
            print('Making a log file...')

    # criterion for loss
    if CONFIG.class_weight:
        class_weight = get_class_weight(
            CONFIG.dataset, split=CONFIG.split, csv_dir=CONFIG.csv_dir)
        class_weight = class_weight.to(device)
    else:
        class_weight = None

    criterion = ActionSegmentationLoss(
        ce=CONFIG.ce, tmse=CONFIG.tmse, weight=class_weight,
        ignore_index=255, tmse_weight=CONFIG.tmse_weight
    )

    # save best F1  and  edit  model  记录所有的轮数中分数最高的
    Best_F1 = 0
    Best_Edit = 0
    All_Time = 0
    Best_Acc = 0

    # train and validate model
    print('\n---------------------------Start training---------------------------\n')
    for epoch in range(begin_epoch, CONFIG.max_epoch):
        # training
        start = time.time()
        train_loss = train(
            train_loader, model, criterion, optimizer, epoch, CONFIG, device)
        train_time = (time.time() - start)

        # if you do validation to determine hyperparams
        if CONFIG.param_search:
            start = time.time()
            val_loss, acc, edit_score, f1s = validate(
                val_loader, model, criterion, CONFIG, device)
            val_time = (time.time() - start)

            # save a model if top1 acc is higher than ever
            if best_loss > val_loss:
                best_loss = val_loss
                torch.save(
                    model.state_dict(),
                    os.path.join(CONFIG.result_path, 'best_loss_model.prm')
                )

        # save checkpoint every epoch
        save_checkpoint(
            CONFIG.result_path, epoch, model, optimizer, best_loss, scheduler)

        # write logs to dataframe and csv file
        tmp = [epoch, optimizer.param_groups[0]['lr'], train_loss]

        # if you do validation to determine hyperparams
        if CONFIG.param_search:
            tmp += [val_loss, acc, edit_score]
            tmp += [f1s[i] for i in range(len(CONFIG.thresholds))]

        tmp_df = pd.Series(tmp, index=log.columns)

        log = log.append(tmp_df, ignore_index=True)
        log.to_csv(os.path.join(CONFIG.result_path, 'log.csv'), index=False)

        # save best F1  and  edit  model 
        Best_All = pd.DataFrame(columns=columns)
        if Best_F1 < tmp[6]:
            Best_F1 = tmp[6]
            Best_F1_All = tmp
            Best_All.to_csv(os.path.join(CONFIG.result_path, 'Best_F1_All.csv'), index=False)
            tmp_df.to_csv(os.path.join(CONFIG.result_path, 'Best_F1_All.csv'), index=False)
            torch.save(
                    model.state_dict(),
                    os.path.join(CONFIG.result_path, 'best_val_F1_model.prm')
                )
        if Best_Edit < tmp[5]:
            Best_Edit = tmp[5]
            Best_Edit_All = tmp
            Best_All.to_csv(os.path.join(CONFIG.result_path, 'Best_Edit_All.csv'), index=False)
            tmp_df.to_csv(os.path.join(CONFIG.result_path, 'Best_Edit_All.csv'), index=False)
            torch.save(
                    model.state_dict(),
                    os.path.join(CONFIG.result_path, 'best_val_Edit_model.prm')
                )
        if Best_Acc < tmp[4]:
            Best_Acc = tmp[4]
            Best_Acc_All = tmp
            tmp_df.to_csv(os.path.join(CONFIG.result_path, 'Best_Acc_All.csv'), index=False)
            torch.save(
                    model.state_dict(),
                    os.path.join(CONFIG.result_path, 'best_val_Acc_model.prm')
                )

        if CONFIG.param_search:
            # if you do validation to determine hyperparams
            print(
                'epoch: {}  lr: {:.4f}  train_time: {:.1f}s  val_time: {:.1f}s  train loss: {:.4f}  val loss: {:.4f}  val_acc: {:.4f}  val_edit: {:.4f} F1s: {}'
                .format(epoch, optimizer.param_groups[0]['lr'], train_time, val_time,
                        train_loss, val_loss, acc, edit_score, f1s)
            )
        else:
            print(
                'epoch: {}\tlr: {:.4f}\ttrain_time: {:.1f}min\ttrain loss: {:.4f}'
                .format(epoch, optimizer.param_groups[0]['lr'], train_time, train_loss)
            )
        All_Time = All_Time + train_time + val_time

    # save models
    torch.save(
        model.state_dict(), os.path.join(CONFIG.result_path, 'final_model.prm'))

    print("")
    print("")
    print("**************************************************************  Best Acc ***************************************************************")
    print("")
    print(
        'epoch: {}\tlr: {:.4f}\tval_acc: {:.4f}\tval_edit: {:.4f}\tF1s: {}'
        .format(Best_Acc_All[0], Best_Acc_All[1], Best_Acc_All[4], Best_Acc_All[5], Best_Acc_All[-3:])
    )
    print("")
    print("**************************************************************  Best Edit **************************************************************")
    print("")
    print(
        'epoch: {}\tlr: {:.4f}\tval_acc: {:.4f}\tval_edit: {:.4f}\tF1s: {}'
        .format(Best_Edit_All[0], Best_Edit_All[1], Best_Edit_All[4], Best_Edit_All[5], Best_Edit_All[-3:])
    )
    print("")
    print("**************************************************************  Best F1 ***************************************************************")
    print("")
    print(
        'epoch: {}\tlr: {:.4f}\tval_acc: {:.4f}\tval_edit: {:.4f}\tF1s: {}'
        .format(Best_F1_All[0], Best_F1_All[1], Best_F1_All[4], Best_F1_All[5], Best_F1_All[-3:])
    )
    print("")
    print("**************************************************************   config  ****************************************************************")
    print("")
    print("tmse_weight",CONFIG.tmse_weight, "  optimizer: ", CONFIG.optimizer, " scheduler: ", CONFIG.scheduler,"n_classes: ",n_classes)
    print("kernel_size",CONFIG.kernel_size, "  n_features: ", CONFIG.n_features, " in_channel: ", CONFIG.in_channel)
    print("Dataset: {}\tSplit: {}".format(CONFIG.dataset, CONFIG.split))
    print("Batch Size: {}\tNum in channels: {}\tNum Workers: {}".format(CONFIG.batch_size, CONFIG.in_channel, CONFIG.num_workers))
    print("Dataset: {}\tSplit: {}".format(CONFIG.dataset, CONFIG.split))
    print("train_data: ",len(train_data))
    print("")
    print("***************************************************************************************************************************************")
    print("")
    print("All_time: {:.4f}min".format(All_Time/60))
    print(CONFIG.result_path)


if __name__ == '__main__':
    main()
