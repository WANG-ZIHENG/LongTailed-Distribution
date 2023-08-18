import argparse
import time
import warnings
import pickle
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import models
from tensorboardX import SummaryWriter
from utils import *
from imagenet_lt_data import *
from losses import LDAMLoss, FocalLoss, SupConLoss
from collections import OrderedDict

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Cifar Training')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnext50_32x4d',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnext50_32x4d)')
parser.add_argument('--rand_number', default=0, type=int, help='fix random number for data sampling')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=20, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                    metavar='W', help='weight decay',
                    dest='weight_decay')
parser.add_argument('--epoch_thresh', default=160, type=int, metavar='N',
                    help='the epoch threshold for generating rare samples')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--root_log', type=str, default='log')
parser.add_argument('--root_model', type=str, default='checkpoint')
parser.add_argument('--data_dir', type=str, default='data/mini-ImageNet')
parser.add_argument('--cls_num', type=int, default=100)
parser.add_argument('--head_tail_ratio', type=float, default=0.2)
parser.add_argument('--cbl_beta', default=0.999, type=float,
                    help='Beta for class balanced loss.')
parser.add_argument('--norm_out', type=bool, default=True)
parser.add_argument('--add_rsg', type=bool, default=True)
parser.add_argument('--add_cross_entropy_loss', type=bool, default=False)
parser.add_argument('--add_ldam_loss', type=bool, default=False)
parser.add_argument('--ldam_weight_type', type=str, choices=['cbl', 'icf'], default='icf')
parser.add_argument('--ldam_loss_weight', type=float, default=0.1)
parser.add_argument('--add_focal_loss', type=bool, default=False)
parser.add_argument('--add_supcon_loss', type=bool, default=True)
parser.add_argument('--supcon_loss_weight', type=float, default=0.9)
parser.add_argument('--add_arc_margin_loss', type=bool, default=False)
parser.add_argument('--add_add_margin_loss', type=bool, default=False)
parser.add_argument('--add_sphere_loss', type=bool, default=False)

best_acc1 = 0


def get_loss(args, output, target, features,
             cross_entropy_loss=None, ldam_loss=None, focal_loss=None, supcon_loss=None,
             arc_margin_output=None, add_margin_output=None, sphere_output=None):
    loss = 0
    if cross_entropy_loss is not None:
        loss += cross_entropy_loss(output, target)
        if arc_margin_output is not None:
            loss += cross_entropy_loss(arc_margin_output, target)
        if add_margin_output is not None:
            loss += cross_entropy_loss(add_margin_output, target)
        if sphere_output is not None:
            loss += cross_entropy_loss(sphere_output, target)
    if ldam_loss is not None:
        loss += args.ldam_loss_weight * ldam_loss(output, target)
        if arc_margin_output is not None:
            loss += args.ldam_loss_weight * ldam_loss(arc_margin_output, target)
        if add_margin_output is not None:
            loss += args.ldam_loss_weight * ldam_loss(add_margin_output, target)
        if sphere_output is not None:
            loss += args.ldam_loss_weight * ldam_loss(sphere_output, target)
    if focal_loss is not None:
        loss += focal_loss(output, target)
        if arc_margin_output is not None:
            loss += focal_loss(arc_margin_output, target)
        if add_margin_output is not None:
            loss += focal_loss(add_margin_output, target)
        if sphere_output is not None:
            loss += focal_loss(sphere_output, target)
    if supcon_loss is not None:
        loss += args.supcon_loss_weight * supcon_loss(features, target)

    return loss


def main():
    args = parser.parse_args()

    dataset = args.data_dir[args.data_dir.rindex('/') + 1:]
    store_name = '%s_%s_%s' % ('ImageNet_LT', dataset, args.arch)
    if args.norm_out:
        store_name += '_normout'
    if args.add_rsg:
        store_name += '_rsg'
    if args.add_cross_entropy_loss:
        store_name += '_ce'
    if args.add_ldam_loss:
        store_name += '_ldam'
    if args.add_focal_loss:
        store_name += '_focal'
    if args.add_supcon_loss:
        store_name += '_supcon'
    if args.add_arc_margin_loss:
        store_name += '_arcmar'
    if args.add_add_margin_loss:
        store_name += '_addmar'
    if args.add_sphere_loss:
        store_name += '_sphere'
    args.store_name = store_name

    prepare_folders(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    ngpus_per_node = torch.cuda.device_count()
    print(ngpus_per_node)
    main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1

    print("=> creating model '{}'".format(args.arch))
    # Data loading code
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    transform_val = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    train_dataset = ImageNet_LT(args.data_dir, args.cls_num, transform_train, 'train')
    val_dataset = ImageNet_LT(args.data_dir, args.cls_num, transform_val, 'val')

    cls_num_list = train_dataset.get_cls_num_list()
    print('cls num list:')
    print(cls_num_list)

    args.cls_num_list = cls_num_list.copy()

    head_lists = []
    Inf = 0
    for i in range(int(args.cls_num * args.head_tail_ratio)):
        head_lists.append(cls_num_list.index(max(cls_num_list)))
        cls_num_list[cls_num_list.index(max(cls_num_list))] = Inf

    model = models.__dict__[args.arch](
        num_classes=args.cls_num,
        phase_train=True,
        norm_out=args.norm_out,
        add_rsg=args.add_rsg,
        head_lists=head_lists,
        add_arc_margin_loss=args.add_arc_margin_loss,
        add_add_margin_loss=args.add_add_margin_loss,
        add_sphere_loss=args.add_sphere_loss,
        epoch_thresh=args.epoch_thresh
    )
    model = torch.nn.DataParallel(model).cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9, last_epoch=-1)

    cudnn.benchmark = True
    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=100, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # init log for training
    with open(os.path.join(args.root_log, args.store_name, 'args.pkl'), 'wb') as fout:
        pickle.dump(args, fout)

    log_training = open(os.path.join(args.root_log, args.store_name, 'log_train.csv'), 'w')
    log_testing = open(os.path.join(args.root_log, args.store_name, 'log_test.csv'), 'w')
    tf_writer = SummaryWriter(log_dir=os.path.join(args.root_log, args.store_name))
    for epoch in range(args.start_epoch, args.epochs):
        # adjust_learning_rate(optimizer, epoch, args)

        if epoch == 160:
            train_sampler = ImbalancedDatasetSampler(train_dataset, label_count=args.cls_num_list)
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
                num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

        if args.ldam_weight_type == 'cbl':
            effective_num = 1.0 - np.power(args.cbl_beta, args.cls_num_list)
            per_cls_weights = (1.0 - args.cbl_beta) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(args.cls_num_list)
            per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()
        else:
            # 计算倒数总和
            inverse_sum = sum([1 / x for x in args.cls_num_list])
            # 计算权重
            per_cls_weights = torch.tensor([1 / (x * inverse_sum) for x in args.cls_num_list]).cuda()

        cross_entropy_loss = None
        if args.add_cross_entropy_loss:
            cross_entropy_loss = nn.CrossEntropyLoss()

        ldam_loss = None
        if args.add_ldam_loss:
            ldam_loss = LDAMLoss(cls_num_list=args.cls_num_list, max_m=0.3, s=30, weight=per_cls_weights).cuda()

        focal_loss = None
        if args.add_focal_loss:
            focal_loss = FocalLoss(args.cls_num, alpha=None, gamma=0).cuda()

        supcon_loss = None
        if args.add_supcon_loss:
            supcon_loss = SupConLoss().cuda()

        # train for one epoch
        train(train_loader, model, optimizer, epoch, args, log_training, tf_writer,
              cross_entropy_loss=cross_entropy_loss, ldam_loss=ldam_loss,
              focal_loss=focal_loss, supcon_loss=supcon_loss)

        # evaluate on validation set
        acc1 = validate(val_loader, model, epoch, args, log_testing, tf_writer,
                        cross_entropy_loss=cross_entropy_loss, ldam_loss=ldam_loss,
                        focal_loss=focal_loss, supcon_loss=supcon_loss)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        tf_writer.add_scalar('acc/test_top1_best', best_acc1, epoch)
        output_best = 'Best Prec@1: %.3f\n' % (best_acc1)
        print(output_best)
        log_testing.write(output_best + '\n')
        log_testing.flush()

        # remove RSG module, since RSG is not used during testing.
        new_state_dict = OrderedDict()
        for k in model.state_dict().keys():
            name = k[7:]  # remove `module.`
            if 'RSG' in k:
                continue
            new_state_dict[name] = model.state_dict()[k]

        save_checkpoint(args, {
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': new_state_dict,
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
        }, is_best)
        scheduler.step()


def train(train_loader, model, optimizer, epoch, args, log, tf_writer,
          cross_entropy_loss=None, ldam_loss=None, focal_loss=None, supcon_loss=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(iter(train_loader)):
        # measure data loading time
        data_time.update(time.time() - end)
        target = target.cuda(non_blocking=True)

        # compute output
        fea, output, target, cesc_loss, total_mv_loss, arc_margin_output, add_margin_output, sphere_output = model(
            input, epoch, target)

        loss = 0
        if args.add_rsg:
            loss += 0.1 * cesc_loss.mean() + 0.01 * total_mv_loss.mean()
        loss += get_loss(args, output, target, fea,
                         cross_entropy_loss=cross_entropy_loss, ldam_loss=ldam_loss,
                         focal_loss=focal_loss, supcon_loss=supcon_loss,
                         arc_margin_output=arc_margin_output, add_margin_output=add_margin_output,
                         sphere_output=sphere_output)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5,
                lr=optimizer.param_groups[-1]['lr'] * 0.1))  # TODO
            print(output)
            log.write(output + '\n')
            log.flush()

    tf_writer.add_scalar('loss/train', losses.avg, epoch)
    tf_writer.add_scalar('acc/train_top1', top1.avg, epoch)
    tf_writer.add_scalar('acc/train_top5', top5.avg, epoch)
    tf_writer.add_scalar('lr', optimizer.param_groups[-1]['lr'], epoch)


def validate(val_loader, model, epoch, args, log=None, tf_writer=None, flag='val',
             cross_entropy_loss=None, ldam_loss=None, focal_loss=None, supcon_loss=None):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    # switch to evaluate mode
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            # if args.gpu is not None:
            input = input.cuda(0, non_blocking=True)
            target = target.cuda(0, non_blocking=True)

            # compute output
            fea, output = model(input, phase_train=False)
            loss = get_loss(args, output, target, fea,
                            cross_entropy_loss=cross_entropy_loss, ldam_loss=ldam_loss,
                            focal_loss=focal_loss, supcon_loss=supcon_loss)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            _, pred = torch.max(output, 1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

            if i % args.print_freq == 0:
                output = ('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))
                print(output)
        cf = confusion_matrix(all_targets, all_preds).astype(float)
        cls_cnt = cf.sum(axis=1)
        cls_hit = np.diag(cf)
        cls_acc = cls_hit / cls_cnt
        output = ('{flag} Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
                  .format(flag=flag, top1=top1, top5=top5, loss=losses))
        out_cls_acc = '%s Class Accuracy: %s' % (
            flag, (np.array2string(cls_acc, separator=',', formatter={'float_kind': lambda x: "%.3f" % x})))
        print(output)
        print(out_cls_acc)
        if log is not None:
            log.write(output + '\n')
            log.write(out_cls_acc + '\n')
            log.flush()

        tf_writer.add_scalar('loss/test_' + flag, losses.avg, epoch)
        tf_writer.add_scalar('acc/test_' + flag + '_top1', top1.avg, epoch)
        tf_writer.add_scalar('acc/test_' + flag + '_top5', top5.avg, epoch)
        tf_writer.add_scalars('acc/test_' + flag + '_cls_acc', {str(i): x for i, x in enumerate(cls_acc)}, epoch)

    return top1.avg


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    epoch = epoch + 1
    if epoch > 180:
        lr = args.lr * 0.001
    elif epoch > 160:
        lr = args.lr * 0.01
    elif epoch > 120:
        lr = args.lr * 0.1
    else:
        if epoch <= 5:
            lr = args.lr * epoch / 5
        else:
            lr = args.lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
