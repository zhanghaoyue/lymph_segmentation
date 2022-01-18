import os
import random
from random import shuffle
import numpy as np
import torch
from torch.utils import data
import detection_transform as T
from skimage.transform import resize
from torchvision.ops import masks_to_boxes
import misc
from PIL import Image


class ImageFolder(data.Dataset):
    def __init__(self, root, image_size=112, mode='train'):
        """Initializes image paths and preprocessing module."""
        self.root = root

        # GT : Ground Truth
        self.GT_paths = root[:-1] + '_GT/'
        self.image_paths = list(map(lambda x: os.path.join(root, x), os.listdir(root)))
        self.image_size = image_size
        self.mode = mode
        if self.mode == 'train':
            self.transforms = get_transform(train=True)
        else:
            self.transforms = get_transform(train=False)
        print("image count in {} path :{}".format(self.mode, len(self.image_paths)))

    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""
        image_path = self.image_paths[index]
        GT_path = self.GT_paths + os.path.basename(image_path)
        im_array = np.load(image_path)
        GT_array = np.load(GT_path)
        image = Image.fromarray(im_array.astype(np.int16))
        GT = Image.fromarray(GT_array.astype(np.int16))
        # image = image.resize((self.image_size, self.image_size))
        # GT = GT.resize((self.image_size, self.image_size))
        GT = np.array(GT)
        image = np.array(image)

        # instances are encoded as different colors
        obj_ids = np.unique(GT)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        GTs = GT == obj_ids[:, None, None]

        num_objs = len(obj_ids)
        boxes = []
        area = []
        labels = []
        for i in range(num_objs):
            pos = np.where(GTs[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            if xmax > xmin and ymax > ymin:
                boxes.append([xmin, ymin, xmax, ymax])
                area.append((ymax - ymin) * (xmax - xmin))
                labels.append(torch.ones(1, dtype=torch.int64))
            else:
                boxes.append([0, 1, 2, 3])
                labels.append(torch.zeros(1, dtype=torch.int64))
                area.append(0.)

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # convert everything into a torch.Tensor
        # there is only one class
        masks = torch.as_tensor(GTs, dtype=torch.uint8)

        image_id = torch.tensor([index])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        labels = torch.cat(labels)
        area = torch.FloatTensor(area)
        target = {"boxes": boxes, "labels": labels, "masks": masks, "image_id": image_id, "area": area,
                  "iscrowd": iscrowd}

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self):
        """Returns the total number of font files."""
        return len(self.image_paths)


def get_loader(image_path, image_size, batch_size, sampler, num_workers=2, mode='train'):
    """Builds and returns Dataloader."""

    dataset = ImageFolder(root=image_path, image_size=image_size, mode=mode)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  sampler=sampler,
                                  shuffle=False,
                                  num_workers=num_workers,
                                  drop_last=True,
                                  collate_fn=misc.collate_fn)
    return data_loader


def get_transform(train):
    transforms = [T.ToTensor()]
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)
