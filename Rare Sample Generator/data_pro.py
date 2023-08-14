import os
import codecs
import random
import pickle
import imageio
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def split_mine(data_dir):
    # n01532829
    pics = os.listdir('%s/images' % data_dir)
    labels = [x[:9] for x in pics]
    label_index = set(labels)
    label_index = [(x, i) for i, x in enumerate(label_index)]
    label_index = dict(label_index)  # 数字标签
    print(label_index)
    labels = [label_index[x] for x in labels]  # 转换成数字标签
    pics = ['%s/images/' % data_dir + x for x in pics]  # 完整图片路径

    df = pd.DataFrame()
    df['img'] = pics
    df['label'] = labels
    print(df)

    trainall = []
    testall = []
    valall = []
    for x in range(100):
        d = df[df['label'] == x]
        # 在每个类别数据中划分出验证集测试集
        train, _ = train_test_split(d, test_size=0.2, random_state=0)
        test, val = train_test_split(_, test_size=0.5, random_state=0)
        num = random.randint(5, 480)  # train里抽样
        train = train.sample(num)
        trainall.append(train)
        testall.append(test)
        valall.append(val)

    trainall = pd.concat(trainall, axis=0)
    testall = pd.concat(testall, axis=0)
    valall = pd.concat(valall, axis=0)
    print(trainall)

    trainall.to_csv('%s/train.csv' % data_dir, index=None)
    testall.to_csv('%s/test.csv' % data_dir, index=None)
    valall.to_csv('%s/val.csv' % data_dir, index=None)


# Imbalanced Data
def get_img_num_per_cls(cls_num, img_max, imb_factor=None):
    """
    Get a list of image numbers for each class, given cifar version
    Num of imgs follows emponential distribution
    img max: 5000 / 500 * e^(-lambda * 0);
    img min: 5000 / 500 * e^(-lambda * int(cifar_version - 1))
    exp(-lambda * (int(cifar_version) - 1)) = img_max / img_min
    args:
      cifar_version: str, '10', '100', '20'
      imb_factor: float, imbalance factor: img_min/img_max,
        None if geting default cifar data number
    output:
      img_num_per_cls: a list of number of images per class
    """
    if imb_factor is None:
        return [img_max] * cls_num
    img_num_per_cls = []
    for cls_idx in range(cls_num):
        num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
        img_num_per_cls.append(int(num))
    return img_num_per_cls


def read_cifar10(data_dir):
    train = {idx: [] for idx in range(10)}
    for batch_idx in range(1, 6):
        with open('%s/data_batch_%s' % (data_dir, batch_idx), 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
            b_data = dict[b'data']
            b_labels = dict[b'labels']
            b_filenames = dict[b'filenames']
            for b_idx in range(len(b_labels)):
                b_filename = b_filenames[b_idx]
                b_label = b_labels[b_idx]
                b_image = b_data[b_idx]
                b_image = np.reshape(b_image, (3, 32, 32))
                b_image = b_image.transpose(1, 2, 0)
                # imageio.imsave('E:/temp/test.jpg', b_image)
                train[b_label].append([b_filename, b_image])

    test = {idx: [] for idx in range(10)}
    with open('%s/test_batch' % data_dir, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
        b_data = dict[b'data']
        b_labels = dict[b'labels']
        b_filenames = dict[b'filenames']
        for b_idx in range(len(b_labels)):
            b_filename = b_filenames[b_idx]
            b_label = b_labels[b_idx]
            b_image = b_data[b_idx]
            b_image = np.reshape(b_image, (3, 32, 32))
            b_image = b_image.transpose(1, 2, 0)
            # imageio.imsave('E:/temp/test.jpg', b_image)
            test[b_label].append([b_filename, b_image])

    return train, test


def read_cifar100(data_dir):
    train = {idx: [] for idx in range(100)}
    with open('%s/train' % data_dir, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
        b_data = dict[b'data']
        b_labels = dict[b'fine_labels']
        b_filenames = dict[b'filenames']
        for b_idx in range(len(b_labels)):
            b_filename = b_filenames[b_idx]
            b_label = b_labels[b_idx]
            b_image = b_data[b_idx]
            b_image = np.reshape(b_image, (3, 32, 32))
            b_image = b_image.transpose(1, 2, 0)
            # imageio.imsave('E:/temp/test.jpg', b_image)
            train[b_label].append([b_filename, b_image])

    test = {idx: [] for idx in range(100)}
    with open('%s/test' % data_dir, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
        b_data = dict[b'data']
        b_labels = dict[b'fine_labels']
        b_filenames = dict[b'filenames']
        for b_idx in range(len(b_labels)):
            b_filename = b_filenames[b_idx]
            b_label = b_labels[b_idx]
            b_image = b_data[b_idx]
            b_image = np.reshape(b_image, (3, 32, 32))
            b_image = b_image.transpose(1, 2, 0)
            # imageio.imsave('E:/temp/test.jpg', b_image)
            test[b_label].append([b_filename, b_image])

    return train, test


def split_cifar10(data_dir):
    if not os.path.exists('%s/images' % data_dir):
        os.makedirs('%s/images' % data_dir)

    train, test = read_cifar10('%s/cifar-10-python' % data_dir)
    train_num_per_cls = get_img_num_per_cls(10, 5000, imb_factor=0.1)
    val_test_num_per_cls = get_img_num_per_cls(10, 500, imb_factor=0.1)

    train_data = []
    val_data = []
    test_data = []
    for label in train:
        print('processing label %s' % label)
        train_image_list = train[label]
        test_image_list = test[label]
        train_num = train_num_per_cls[label]
        val_test_num = val_test_num_per_cls[label]

        train_image_list = train_image_list[:train_num]
        val_image_list = test_image_list[:val_test_num]
        test_image_list = test_image_list[500:500 + val_test_num]

        for filename, image in train_image_list:
            filename = 'images/%s' % filename.decode('UTF-8')
            imageio.imsave('%s/%s' % (data_dir, filename), image)
            train_data.append([filename, label])
        for filename, image in val_image_list:
            filename = 'images/%s' % filename.decode('UTF-8')
            imageio.imsave('%s/%s' % (data_dir, filename), image)
            val_data.append([filename, label])
        for filename, image in test_image_list:
            filename = 'images/%s' % filename.decode('UTF-8')
            imageio.imsave('%s/%s' % (data_dir, filename), image)
            test_data.append([filename, label])

    with codecs.open('%s/train.csv' % data_dir, 'w', 'utf-8') as fout:
        fout.write('img,label\n')
        for record in train_data:
            fout.write('%s,%s\n' % (record[0], record[1]))
    with codecs.open('%s/val.csv' % data_dir, 'w', 'utf-8') as fout:
        fout.write('img,label\n')
        for record in val_data:
            fout.write('%s,%s\n' % (record[0], record[1]))
    with codecs.open('%s/test.csv' % data_dir, 'w', 'utf-8') as fout:
        fout.write('img,label\n')
        for record in test_data:
            fout.write('%s,%s\n' % (record[0], record[1]))

    print('complete')


def split_cifar100(data_dir):
    if not os.path.exists('%s/images' % data_dir):
        os.makedirs('%s/images' % data_dir)

    train, test = read_cifar100('%s/cifar-100-python' % data_dir)
    train_num_per_cls = get_img_num_per_cls(100, 500, imb_factor=0.1)
    val_test_num_per_cls = get_img_num_per_cls(100, 50, imb_factor=0.1)

    train_data = []
    val_data = []
    test_data = []
    for label in train:
        print('processing label %s' % label)
        train_image_list = train[label]
        test_image_list = test[label]
        train_num = train_num_per_cls[label]
        val_test_num = val_test_num_per_cls[label]

        train_image_list = train_image_list[:train_num]
        val_image_list = test_image_list[:val_test_num]
        test_image_list = test_image_list[50:50 + val_test_num]

        for filename, image in train_image_list:
            filename = 'images/%s' % filename.decode('UTF-8')
            imageio.imsave('%s/%s' % (data_dir, filename), image)
            train_data.append([filename, label])
        for filename, image in val_image_list:
            filename = 'images/%s' % filename.decode('UTF-8')
            imageio.imsave('%s/%s' % (data_dir, filename), image)
            val_data.append([filename, label])
        for filename, image in test_image_list:
            filename = 'images/%s' % filename.decode('UTF-8')
            imageio.imsave('%s/%s' % (data_dir, filename), image)
            test_data.append([filename, label])

    with codecs.open('%s/train.csv' % data_dir, 'w', 'utf-8') as fout:
        fout.write('img,label\n')
        for record in train_data:
            fout.write('%s,%s\n' % (record[0], record[1]))
    with codecs.open('%s/val.csv' % data_dir, 'w', 'utf-8') as fout:
        fout.write('img,label\n')
        for record in val_data:
            fout.write('%s,%s\n' % (record[0], record[1]))
    with codecs.open('%s/test.csv' % data_dir, 'w', 'utf-8') as fout:
        fout.write('img,label\n')
        for record in test_data:
            fout.write('%s,%s\n' % (record[0], record[1]))

    print('complete')


if __name__ == '__main__':
    data_dir = './data/'

    # split_mine('%s/mine' % data_dir)
    # split_cifar10('%s/cifar10' % data_dir)
    # split_cifar100('%s/cifar100' % data_dir)
