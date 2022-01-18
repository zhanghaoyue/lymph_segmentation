import os
import argparse
import random
import numpy as np
import shutil
from shutil import copyfile
from misc import printProgressBar
from random import sample


def rm_mkdir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
        print('Remove path - %s' % dir_path)
    os.makedirs(dir_path)
    print('Create path - %s' % dir_path)
    if os.path.exists(str(dir_path) + '_balanced'):
        shutil.rmtree(str(dir_path) + '_balanced')
        print('Remove path - %s'% str(dir_path) + '_balanced')
    os.makedirs(str(dir_path) + '_balanced')
    print('Create path - %s' % str(dir_path) + '_balanced')
    if os.path.exists(str(dir_path) + '_pos'):
        shutil.rmtree(str(dir_path) + '_pos')
        print('Remove path - %s'% str(dir_path) + '_pos')
    os.makedirs(str(dir_path) + '_pos')
    print('Create path - %s' % str(dir_path) + '_pos')


def main(config):
    rm_mkdir(config.train_path)
    rm_mkdir(config.train_GT_path)
    rm_mkdir(config.valid_path)
    rm_mkdir(config.valid_GT_path)

    with open(config.train_list) as f:
        lines_train = f.read().splitlines()
    with open(config.valid_list) as f:
        lines_valid = f.read().splitlines()

    num_train = len(lines_train)
    num_valid = len(lines_valid)

    print('\nNum of train set : ', num_train)
    print('\nNum of test set : ', num_valid)

    for filename in lines_train:
        # original image path
        img_train_src = os.path.join(config.origin_data_path, filename)

        # original GT path
        gt_train_src = os.path.join(config.origin_GT_path, filename)
        # GT copyto path and rename positive and negative cases
        gt_train_dst_pos = os.path.join(config.train_GT_path, 'pos_' + os.path.basename(filename))
        gt_train_dst_neg = os.path.join(config.train_GT_path, 'neg_' + os.path.basename(filename))
        gt_train_dst_pos_balanced = os.path.join(config.train_GT_path + '_balanced',
                                                 'pos_' + os.path.basename(filename))
        if os.path.exists(gt_train_src):
            copyfile(gt_train_src, gt_train_dst_pos)
            copyfile(img_train_src, os.path.join(config.train_path, 'pos_' + os.path.basename(filename)))
            copyfile(gt_train_src, gt_train_dst_pos_balanced)
            copyfile(img_train_src, os.path.join(config.train_path+'_pos', 'pos_' + os.path.basename(filename)))
            copyfile(gt_train_src, os.path.join(config.train_GT_path + '_pos', 'pos_' + os.path.basename(filename)))
        else:
            img = np.load(img_train_src)
            gt = np.zeros(img.shape)
            np.save(gt_train_dst_neg, gt)
            copyfile(img_train_src, os.path.join(config.train_path, 'neg_' + os.path.basename(filename)))

    for filename in lines_valid:
        img_valid_src = os.path.join(config.origin_data_path, filename)

        gt_valid_src = os.path.join(config.origin_GT_path, filename)
        gt_valid_dst_pos = os.path.join(config.valid_GT_path, 'pos_' + os.path.basename(filename))
        gt_valid_dst_neg = os.path.join(config.valid_GT_path, 'neg_' + os.path.basename(filename))
        gt_valid_dst_pos_balanced = os.path.join(config.valid_GT_path + '_balanced',
                                                 'pos_' + os.path.basename(filename))
        if os.path.exists(gt_valid_src):
            copyfile(gt_valid_src, gt_valid_dst_pos)
            copyfile(img_valid_src, os.path.join(config.valid_path, 'pos_' + os.path.basename(filename)))
            copyfile(gt_valid_src, gt_valid_dst_pos_balanced)
            copyfile(img_valid_src, os.path.join(config.valid_path + '_pos', 'pos_' + os.path.basename(filename)))
            copyfile(gt_valid_src, os.path.join(config.valid_GT_path+'_pos', 'pos_' + os.path.basename(filename)))
        else:
            img = np.load(img_valid_src)
            gt = np.zeros(img.shape)
            np.save(gt_valid_dst_neg, gt)
            copyfile(img_valid_src, os.path.join(config.valid_path, 'neg_' + os.path.basename(filename)))

    # make balanced folder for training data
    _, _, train_GT_pos_balanced = next(os.walk(config.train_GT_path + '_balanced'))
    pos_size = len(train_GT_pos_balanced)
    print("train pos size: ")
    print(pos_size)
    pos_files = [filename for filename in os.listdir(config.train_GT_path) if filename.startswith('pos_')]
    neg_files = [filename for filename in os.listdir(config.train_GT_path) if filename.startswith('neg_')]
    for file in pos_files:
        copyfile(os.path.join(config.train_path, file), os.path.join(config.train_path+'_balanced', file))
    sampled_neg_file = sample(neg_files, pos_size)
    for file in sampled_neg_file:
        copyfile(os.path.join(config.train_GT_path, file), os.path.join(config.train_GT_path+'_balanced', file))
        copyfile(os.path.join(config.train_path, file), os.path.join(config.train_path+'_balanced', file))

    # make balanced folder for testing data
    _, _, valid_GT_pos_balanced = next(os.walk(config.valid_GT_path + '_balanced'))
    pos_size = len(valid_GT_pos_balanced)
    print("valid pos size:")
    print(pos_size)
    pos_files = [filename for filename in os.listdir(config.valid_GT_path) if filename.startswith('pos_')]
    neg_files = [filename for filename in os.listdir(config.valid_GT_path) if filename.startswith('neg_')]
    for file in pos_files:
        copyfile(os.path.join(config.valid_path, file), os.path.join(config.valid_path + '_balanced', file))
    sampled_neg_file = sample(neg_files, pos_size)
    for file in sampled_neg_file:
        copyfile(os.path.join(config.valid_GT_path, file), os.path.join(config.valid_GT_path + '_balanced', file))
        copyfile(os.path.join(config.valid_path, file), os.path.join(config.valid_path + '_balanced', file))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # input
    parser.add_argument('--train_list', type=str, default='/home/sanford2021/Desktop/lymph/data/txt/train_0.txt')
    parser.add_argument('--valid_list', type=str, default='/home/sanford2021/Desktop/lymph/data/txt/valid_0.txt')
    parser.add_argument('--train_fold', type=str, default='train_0')
    parser.add_argument('--valid_fold', type=str, default='valid_0')

    # data path
    parser.add_argument('--origin_data_path', type=str, default='/home/sanford2021/Desktop/lymph/data/npy/img/')
    parser.add_argument('--origin_GT_path', type=str, default='/home/sanford2021/Desktop/lymph/data/npy/detect/')
    # prepared data path
    parser.add_argument('--train_path', type=str, default='./dataset/train')
    parser.add_argument('--train_GT_path', type=str, default='./dataset/train_GT')
    parser.add_argument('--valid_path', type=str, default='./dataset/valid')
    parser.add_argument('--valid_GT_path', type=str, default='./dataset/valid_GT')

    config = parser.parse_args()
    print(config)
    main(config)
