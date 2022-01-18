import argparse
import os
from solver import Solver
from data_loader import get_loader
from torch.backends import cudnn
import random
import warnings
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

warnings.filterwarnings("ignore")


def main(config):
    cudnn.benchmark = True
    if config.model_type not in ['U_Net', 'R2U_Net', 'AttU_Net', 'R2AttU_Net', 'RCNN']:
        print('ERROR!! model_type should be selected in U_Net/R2U_Net/AttU_Net/R2AttU_Net/RCNN')
        print('Your input for model_type was %s' % config.model_type)
        return

    # Create directories if not exist
    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)
    config.result_path = os.path.join(config.result_path, config.model_type)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)

    # lr = random.random() * 0.0005 + 0.0000005
    # augmentation_prob = random.random() * 0.7
    # epoch = random.choice([100, 150])
    # decay_ratio = random.random() * 0.8
    # decay_epoch = int(epoch * decay_ratio)
    #
    # config.augmentation_prob = augmentation_prob
    # config.num_epochs = epoch
    # config.lr = lr
    # config.num_epochs_decay = decay_epoch

    print(config)
    validation_split = 0.2
    path, dirs, files = next(os.walk(config.train_path))
    dataset_size = len(files)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = get_loader(image_path=config.train_path,
                              image_size=config.image_size,
                              batch_size=config.batch_size,
                              num_workers=config.num_workers,
                              sampler=train_sampler,
                              mode='train')
    valid_loader = get_loader(image_path=config.train_path,
                              image_size=config.image_size,
                              batch_size=config.batch_size,
                              num_workers=config.num_workers,
                              sampler=valid_sampler,
                              mode='valid')
    test_loader = get_loader(image_path=config.valid_path,
                             image_size=config.image_size,
                             batch_size=config.batch_size,
                             num_workers=config.num_workers,
                             sampler=None,
                             mode='test')

    solver = Solver(config, train_loader, valid_loader, test_loader)

    # Train and sample the images
    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model hyper-parameters
    parser.add_argument('--image_size', type=int, default=112)
    parser.add_argument('--t', type=int, default=3, help='t for Recurrent step of R2U_Net or R2AttU_Net')

    # training hyper-parameters
    parser.add_argument('--img_ch', type=int, default=3)
    parser.add_argument('--output_ch', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--num_epochs_decay', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--beta1', type=float, default=0.9)  # momentum1 in Adam
    parser.add_argument('--beta2', type=float, default=0.999)  # momentum2 in Adam
    parser.add_argument('--augmentation_prob', type=float, default=0.0)

    parser.add_argument('--log_step', type=int, default=2)
    parser.add_argument('--val_step', type=int, default=2)

    # misc
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--model_type', type=str, default='RCNN', help='U_Net/R2U_Net/AttU_Net/R2AttU_Net')
    parser.add_argument('--model_path', type=str, default='./models')
    parser.add_argument('--train_path', type=str, default='./dataset/train_pos/')
    parser.add_argument('--valid_path', type=str, default='./dataset/valid_pos/')
    parser.add_argument('--result_path', type=str, default='./result/')

    parser.add_argument('--cuda_idx', type=int, default=1)

    config = parser.parse_args()
    main(config)
