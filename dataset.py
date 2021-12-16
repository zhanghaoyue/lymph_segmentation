import os
import argparse
import random
import numpy as np
import shutil
from shutil import copyfile
from misc import printProgressBar


def rm_mkdir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
        print('Remove path - %s' % dir_path)
    os.makedirs(dir_path)
    print('Create path - %s' % dir_path)


def main(config):
    rm_mkdir(config.train_path)
    rm_mkdir(config.train_GT_path)
    rm_mkdir(config.valid_path)
    rm_mkdir(config.valid_GT_path)
    rm_mkdir(config.test_path)
    rm_mkdir(config.test_GT_path)

    train_data = open(config.train_list, 'r')
    valid_data = open(config.valid_list, 'r')
    lines_train = train_data.readlines()
    lines_valid = valid_data.readlines()

    num_train = len(lines_train)
    num_valid = len(lines_valid)

    print('\nNum of train set : ', num_train)
    print('\nNum of valid set : ', num_valid)

    for filename in lines_train:
        img_train_src = os.path.join(config.origin_data_path, config.train_fold, filename)
        img_train_dst = os.path.join(config.train_path, filename)
        copyfile(img_train_src, img_train_dst)

        gt_train_src = os.path.join(config.origin_GT_path, config.train_fold, filename)
        gt_train_dst = os.path.join(config.train_GT_path, filename)
        if os.path.exists(gt_train_src):
            copyfile(gt_train_src, gt_train_dst)
        else:
            img = np.load(img_train_src)
            gt = np.zeros(img.shape)
            np.save(gt_train_dst, gt)

    for filename in lines_valid:
        img_valid_src = os.path.join(config.origin_data_path, config.valid_fold, filename)
        img_valid_dst = os.path.join(config.valid_path, filename)
        copyfile(img_valid_src, img_valid_dst)

        gt_valid_src = os.path.join(config.origin_GT_path, config.valid_fold, filename)
        gt_valid_dst = os.path.join(config.valid_GT_path, filename)
        if os.path.exists(gt_valid_src):
            copyfile(gt_valid_src, gt_valid_dst)
        else:
            img = np.load(img_valid_src)
            gt = np.zeros(img.shape)
            np.save(gt_valid_dst, gt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # input
    parser.add_argument('--train_list', type=float, default='/data/train_01.txt')
    parser.add_argument('--valid_list', type=float, default='/data/valid_01.txt')
    parser.add_argument('--train_fold', type=float, default='train_01')
    parser.add_argument('--valid_fold', type=float, default='valid_01')

    # data path
    parser.add_argument('--origin_data_path', type=str, default='data/npy/img/')
    parser.add_argument('--origin_GT_path', type=str, default='data/npy/msk/')
    # prepared data path
    parser.add_argument('--train_path', type=str, default='./dataset/train/')
    parser.add_argument('--train_GT_path', type=str, default='./dataset/train_GT/')
    parser.add_argument('--valid_path', type=str, default='./dataset/valid/')
    parser.add_argument('--valid_GT_path', type=str, default='./dataset/valid_GT/')

    config = parser.parse_args()
    print(config)
    main(config)
