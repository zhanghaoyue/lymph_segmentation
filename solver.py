import os
from collections import OrderedDict
import collections
import numpy as np
import time
import datetime
import torch
import torchvision
from torch import optim
from torch.autograd import Variable
from tqdm import tqdm
import torch.nn.functional as F
from evaluation import *
from network import U_Net, R2U_Net, AttU_Net, R2AttU_Net
from detection_nets import get_detection_model
from losses import *
from torchvision.ops import box_iou
import misc
import csv


def focal_loss(output, target):
    if target.size() != output.size():
        target = target.unsqueeze(1)
    target = target.float()
    alpha = 0.25
    gamma = 2.0
    BCE_loss = torch.nn.functional.binary_cross_entropy_with_logits(output, target, reduction='none')
    pt = -torch.exp(-BCE_loss)
    F_loss = alpha * (1 - pt) ** gamma * BCE_loss
    return F_loss.mean()


def gd_loss(output, target):
    if target.size() != output.size():
        target = target.unsqueeze(1)
    target = target.float()
    target_sum = target.sum(-1)
    class_weights = Variable(1. / (target_sum * target_sum).clamp(min=1e-5), requires_grad=False)
    intersect = (output * target).sum(-1) * class_weights
    intersect = intersect.sum()

    denominator = ((output + target).sum(-1) * class_weights).sum()

    return - 2. * intersect / denominator.clamp(min=1e-5)


def combo_loss(output, target):
    alpha = 5
    beta = 3
    return alpha * focal_loss(output, target) + beta * gd_loss(output, target)


class Solver(object):
    def __init__(self, config, train_loader, valid_loader, test_loader):

        # Data loader
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        # Models
        self.net = None
        self.optimizer = None
        self.scheduler = None
        self.img_ch = config.img_ch
        self.output_ch = config.output_ch
        self.criterion = combo_loss
        self.augmentation_prob = config.augmentation_prob

        # Hyper-parameters
        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2

        # Training settings
        self.num_epochs = config.num_epochs
        self.num_epochs_decay = config.num_epochs_decay
        self.batch_size = config.batch_size

        # Step size
        self.log_step = config.log_step
        self.val_step = config.val_step

        # Path
        self.model_path = config.model_path
        self.result_path = config.result_path
        self.mode = config.mode

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = config.model_type

        # LR
        self.cycle_r = False
        self.gap_epoch = 10
        self.patience = 3
        self.cos_lr = False
        self.Tmax = 20
        self.lr_gap = 100

        # detection range
        self.cls_th = 0.05
        self.nms_th = 0.01
        self.s_th = 0.05
        self.max_dets = 100
        self.iou_th = 0.3

        self.t = config.t
        self.build_model()

    def build_model(self):
        """Build generator and discriminator."""
        if self.model_type == 'U_Net':
            # self.unet = U_Net(img_ch=3, output_ch=1)
            self.net = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                                      in_channels=3, out_channels=1, init_features=32, pretrained=True)
        elif self.model_type == 'R2U_Net':
            self.net = R2U_Net(img_ch=3, output_ch=1, t=self.t)
        elif self.model_type == 'AttU_Net':
            self.net = AttU_Net(img_ch=3, output_ch=1)
        elif self.model_type == 'R2AttU_Net':
            self.net = R2AttU_Net(img_ch=3, output_ch=1, t=self.t)
        elif self.model_type == 'RCNN':
            self.net = get_detection_model(num_classes=2)

        self.optimizer = optim.Adam(list(self.net.parameters()),
                                    self.lr, [self.beta1, self.beta2])
        if self.cos_lr:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.Tmax,
                                                                        eta_min=self.lr / self.lr_gap)
        else:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'max', patience=self.patience)
        self.net.to(self.device)

    # self.print_network(self.unet, self.model_type)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def to_data(self, x):
        """Convert variable to tensor."""
        if torch.cuda.is_available():
            x = x.cpu()
        return x.data

    def update_lr(self, g_lr, d_lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr

    def reset_grad(self):
        """Zero the gradient buffers."""
        self.net.zero_grad()

    def compute_accuracy(self, SR, GT):
        SR_flat = SR.view(-1)
        GT_flat = GT.view(-1)

        acc = GT_flat.data.cpu() == (SR_flat.data.cpu() > 0.5)

    def tensor2img(self, x):
        img = (x[:, 0, :, :] > x[:, 1, :, :]).float()
        img = img * 255
        return img

    def train(self):
        """Train encoder, generator and discriminator."""

        # ====================================== Training ===========================================#
        # ===========================================================================================#

        unet_path = os.path.join(self.model_path, '%s-%d-%.4f-%d-%.4f.pkl' % (
            self.model_type, self.num_epochs, self.lr, self.num_epochs_decay, self.augmentation_prob))
        print("start training...")
        # U-Net Train
        if os.path.isfile(unet_path):
            # Load the pretrained Encoder
            self.net.load_state_dict(torch.load(unet_path))
            print('%s is Successfully Loaded from %s' % (self.model_type, unet_path))
        else:
            # Train for Encoder
            lr = self.lr
            best_net = None
            epoch_save = 0
            best_metric = 0
            lr_change = 0
            loss_hist = collections.deque(maxlen=500)

            for epoch in range(self.num_epochs):
                tmp_epoch = epoch + 1
                tmp_lr = self.optimizer.__getstate__()['param_groups'][0]['lr']
                if self.cycle_r > 0:
                    if epoch % (2 * self.Tmax) == 0:
                        best_net = None
                        best_metric_list = np.zeros((2 - 1))
                        best_metric = 0
                        min_loss = 10

                else:
                    if tmp_epoch > epoch_save + self.gap_epoch:
                        break
                    if lr_change == 2:
                        break
                self.net.train(True)

                for i, (images, GT) in tqdm(enumerate(self.train_loader)):
                    # GT : Ground Truth

                    im = list(image.to(self.device) for image in images)
                    label = [{k: v.to(self.device) for k, v in t.items()} for t in GT]

                    if epoch == 0 and i == 0:
                        print('input size:', im[0].shape)
                    # forward
                    loss_dict = self.net(im, label)
                    loss = sum(ls for ls in loss_dict.values())

                    if bool(loss == 0):
                        continue

                    self.optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.1)
                    self.optimizer.step()
                    loss_hist.append(float(loss))
                    if i % 50 == 0:
                        print(
                            'Ep: {} | Iter: {} | Running loss: {:1.4f}'.format(
                                tmp_epoch, i, np.mean(loss_hist)))

                # Decay learning rate
                if (epoch + 1) > (self.num_epochs - self.num_epochs_decay):
                    lr -= (self.lr / float(self.num_epochs_decay))
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = lr
                    print('Decay learning rate to lr: {}.'.format(lr))
                torch.cuda.empty_cache()
                # ===================================== Validation ====================================#
                self.net.train(False)
                self.net.eval()
                val_data_num = self.valid_loader.__len__
                data_length = val_data_num
                all_detections = [None for j in range(data_length)]
                all_annotations = [None for j in range(data_length)]

                with torch.no_grad():
                    for i, (images, GT) in enumerate(self.valid_loader):
                        im = list(image.cuda() for image in images)
                        if epoch == 0 and i == 0:
                            print('val input size:', im[0].shape)

                        outputs = self.net(im)

                        scores = outputs[0]['scores'].detach().cpu().numpy()
                        labels = outputs[0]['labels'].detach().cpu().numpy()
                        boxes = outputs[0]['boxes'].detach().cpu().numpy()

                        indices = np.where(scores > self.s_th)[0]

                        if indices.shape[0] > 0:
                            scores = scores[indices]
                            boxes = boxes[indices]
                            labels = labels[indices]
                            # find the order with which to sort the scores
                            scores_sort = np.argsort(-scores)[:self.max_dets]
                            # select detections
                            image_boxes = boxes[scores_sort]
                            image_scores = scores[scores_sort]
                            image_labels = labels[scores_sort]
                            image_detections = np.concatenate(
                                [image_boxes, np.expand_dims(image_scores, axis=1),
                                 np.expand_dims(image_labels, axis=1)],
                                axis=1)
                            all_detections[i] = image_detections[:, :-1]
                        else:
                            all_detections[i] = np.zeros((0, 5))
                            # if all_detections[i].shape[0] != 0:
                            #     print(all_detections[i])
                            ###########################################################
                            ##################### Get annotations #####################
                            annotations = GT[0]["boxes"].detach().cpu().numpy()
                            all_annotations[i] = annotations
                        ###########################################################
                    false_positives = np.zeros((0,))
                    true_positives = np.zeros((0,))
                    scores = np.zeros((0,))
                    num_annotations = 0.0

                    for i in range(data_length):
                        detections = all_detections[i]
                        annotations = all_annotations[i]
                        num_annotations += annotations.shape[0]
                        detected_annotations = []
                        for d in detections:
                            scores = np.append(scores, d[4])
                            if annotations.shape[0] == 0:
                                false_positives = np.append(false_positives, 1)
                                true_positives = np.append(true_positives, 0)
                                continue
                            d_tensor = torch.tensor(d[:4][np.newaxis])
                            a_tensor = torch.tensor(annotations)
                            overlaps = box_iou(d_tensor, a_tensor).numpy()
                            assigned_annotation = np.argmax(overlaps, axis=1)
                            max_overlap = overlaps[0, assigned_annotation]
                            if max_overlap >= self.iou_th and assigned_annotation not in detected_annotations:
                                false_positives = np.append(false_positives, 0)
                                true_positives = np.append(true_positives, 1)
                                detected_annotations.append(assigned_annotation)
                            else:
                                false_positives = np.append(false_positives, 1)
                                true_positives = np.append(true_positives, 0)
                    if len(false_positives) == 0 and len(true_positives) == 0:
                        print('No detection')
                    else:
                        # sort by score
                        indices = np.argsort(-scores)
                        scores = scores[indices]
                        false_positives = false_positives[indices]
                        true_positives = true_positives[indices]
                        # compute false positives and true positives
                        false_positives = np.cumsum(false_positives)
                        true_positives = np.cumsum(true_positives)
                        # compute recall and precision
                        recall = true_positives / num_annotations
                        precision = true_positives / np.maximum(true_positives + false_positives,
                                                                np.finfo(np.float64).eps)
                        # compute average precision
                        average_precision = misc.compute_ap(recall, precision)
                        print('mAP: {}'.format(average_precision))
                        print("Precision: ", precision[-1])
                        print("Recall: ", recall[-1])
                        if average_precision > best_metric:
                            best_metric = average_precision
                            epoch_save = tmp_epoch
                            save_dict = {}
                            save_dict['net'] = self.net
                            torch.save(save_dict, self.model_path + 'K%s_%s_AP_%.4f_Pr_%.4f_Re_%.4f.pkl' %
                                       (k, str(epoch_save).rjust(3, '0'), best_metric, precision[-1], recall[-1]))

                            del save_dict

                            print('====================== model save ========================')
                    if self.cos_lr:
                        self.scheduler.step()
                    else:
                        self.scheduler.step(best_metric)
                        # 经过学习率策略后的 lr
                        # 如果有变化，记录变化情况
                    before_lr = self.optimizer.__getstate__()['param_groups'][0]['lr']
                    if before_lr != tmp_lr:
                        epoch_save = tmp_epoch
                        lr_change += 1
                        print('================== lr change to %.6f ==================' % before_lr)
                    # 清除缓存，减少训练中内存占用
                    torch.cuda.empty_cache()
