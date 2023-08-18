import argparse
import pickle
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import models
from imagenet_lt_data import *
from utils import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Places Testing')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--data_dir', type=str, default='data/mini-ImageNet')
parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint')
parser.add_argument('--log_dir', type=str, default='./log')
parser.add_argument('--model_dir', type=str,
                    default='ImageNet_LT_mine_resnext50_32x4d_rsg_ce')


def load_checkpoint(config, filepath):
    checkpoint = torch.load(filepath)
    model = models.__dict__[config.arch](
        num_classes=config.cls_num,
        phase_train=False,
        norm_out=config.norm_out
    )

    model.load_state_dict(checkpoint['state_dict'], strict=False)
    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval()
    return model


def shot_acc(train_class_count, test_class_count, class_correct, many_shot_thr=100, low_shot_thr=20):
    many_shot = []
    median_shot = []
    low_shot = []
    for i in range(len(train_class_count)):
        if train_class_count[i] >= many_shot_thr:
            many_shot.append((class_correct[i].detach().cpu().numpy() / test_class_count[i]))
        elif train_class_count[i] <= low_shot_thr:
            low_shot.append((class_correct[i].detach().cpu().numpy() / test_class_count[i]))
        else:
            median_shot.append((class_correct[i].detach().cpu().numpy() / test_class_count[i]))

    return np.mean(many_shot), np.mean(median_shot), np.mean(low_shot)


def mic_acc_cal(preds, labels):
    acc_mic_top1 = (preds == labels).sum().item()
    return acc_mic_top1


if __name__ == "__main__":
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu)

    dataset = args.data_dir[args.data_dir.rindex('/') + 1:]

    with open('%s/%s/args.pkl' % (args.log_dir, args.model_dir), 'rb') as fin:
        config = pickle.load(fin)

    best_checkpoint = '%s/%s/ckpt.best.pth.tar' % (args.checkpoint_dir, args.model_dir)
    model = load_checkpoint(config, best_checkpoint)
    model = model.cuda(args.gpu)

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    train_dataset = ImageNet_LT(args.data_dir, config.cls_num, transform_train, 'train')
    test_dataset = ImageNet_LT(args.data_dir, config.cls_num, transform_test, 'test')

    cls_num_list_train = train_dataset.get_cls_num_list()
    cls_num_list_test = test_dataset.get_cls_num_list()

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    overall_acc = 0.0
    many_shot_overall = 0.0
    median_shot_overall = 0.0
    low_shot_overall = 0.0
    total_num = 0
    correct_class = [0] * config.cls_num

    for i, (input, label) in enumerate(test_loader):
        _, output = model(input.cuda(), phase_train=False)
        predict_ = torch.topk(output, 1, dim=1, largest=True, sorted=True, out=None)[1]
        predict = predict_.cpu().detach().squeeze()
        acc = mic_acc_cal(predict, label.cpu())

        for l in range(0, config.cls_num):
            correct_class[l] += (predict[label == l] == label[label == l]).sum()

        overall_acc += acc
        total_num += len(label.cpu())

    if dataset == 'mini-ImageNet':
        many_shot_thr = 100
        low_shot_thr = 20
    else:
        cls_num_list = train_dataset.get_cls_num_list()
        break_point = int(len(cls_num_list) / 4)
        many_shot_thr = cls_num_list[break_point]
        low_shot_thr = cls_num_list[-break_point]

    overall_acc = overall_acc * 1.0 / total_num
    many_shot_overall, median_shot_overall, low_shot_overall = shot_acc(
        cls_num_list_train, cls_num_list_test, correct_class,
        many_shot_thr=many_shot_thr, low_shot_thr=low_shot_thr)

    test_result = 'The overall accuracy: %.2f. The many shot accuracy: %.2f. The median shot accuracy: %.2f. The low shot accuracy: %.2f.' % (
        overall_acc * 100, many_shot_overall * 100, median_shot_overall * 100, low_shot_overall * 100)
    print(test_result)
    with open('%s/%s/test.log' % (args.log_dir, args.model_dir), 'w', encoding='utf-8') as fout:
        fout.write(test_result)
