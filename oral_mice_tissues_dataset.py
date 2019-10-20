import torch
import torchvision.transforms.functional as TF

import random
import os.path

from torch.utils.data import Dataset
from albumentations import (
    PadIfNeeded,
    HorizontalFlip,
    VerticalFlip,
    CenterCrop,
    Crop,
    Compose,
    Transpose,
    RandomRotate90,
    ElasticTransform,
    GridDistortion,
    OpticalDistortion,
    RandomSizedCrop,
    OneOf,
    CLAHE,
    RandomBrightnessContrast,
    RandomGamma
)
from image_utils import *

import matplotlib.pyplot as plt
import numpy as np


class OralMiceTissuesDataset(Dataset):

    def __init__(self, img_dir="/home/dalifreire/Documents/Doutorado/github/histological_oral_mice_tissues/roi",
                 img_input_size=(256,448), img_output_size=(256,448), method="1-original", dysplasia_level="all",
                 augmentation=True, dataset_type="train"):
        self.img_dir = img_dir
        self.samples = load_dataset(img_dir, dataset_type, method, dysplasia_level)
        self.img_input_size = img_input_size
        self.img_output_size = img_output_size
        self.augmentation = augmentation
        self.dataset_type = dataset_type

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path_img, path_mask, fname = self.samples[idx]
        image = load_pil_image(path_img)
        mask = load_pil_image(path_mask) if os.path.exists(path_mask) else None

        x, y, fname, original_size = self.transform(image, mask, fname)
        return [x, y, fname, original_size]

    def transform(self, image, mask, fname):
        x, y = data_augmentation(image, mask, self.img_input_size, self.img_output_size, self.augmentation)
        return x, y, fname, image.size


def is_valid_file(filename, extensions=('.jpg', '.bmp', '.tif', 'png')):
    return filename.lower().endswith(extensions)


def load_dataset(img_dir, dataset_type, method, dysplasia_level):
    images = []
    dir_str = "{}/image/{}/{}".format(img_dir, dataset_type, method)
    dysplasia_levels = os.listdir(dir_str)
    for dysplasia_type in sorted(dysplasia_levels):
        for root, _, fnames in sorted(os.walk(dir_str + "/" + dysplasia_type)):
            if dysplasia_level.lower() == 'all' or root.endswith(dysplasia_level) or dysplasia_type in dysplasia_level:
                for fname in sorted(fnames):
                    path_img = os.path.join(root, fname)
                    path_mask = os.path.join(img_dir + "/mask/" + dysplasia_type, fname)
                    if is_valid_file(path_img) and is_valid_file(path_mask):
                        item = (path_img, path_mask, fname)
                        images.append(item)
    return images


def data_augmentation(input_image, output_mask, img_input_size=(256,448), img_output_size=(256,448), aug=True):
    image = TF.resize(input_image, size=img_input_size)
    mask = TF.resize(output_mask, size=img_input_size) if output_mask is not None else None
    if aug:

        # Random horizontal flipping
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Random vertical flipping
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        # Random rotation
        if random.random() > 0.5 and img_input_size[0] == img_input_size[1]:
            augmented = RandomRotate90(p=1)(image=np.array(image), mask=np.array(mask))
            image = Image.fromarray(augmented['image'])
            mask = Image.fromarray(augmented['mask'])

        # Random transpose
        if random.random() > 0.5 and img_input_size[0] == img_input_size[1]:
            augmented = Transpose(p=1)(image=np.array(image), mask=np.array(mask))
            image = Image.fromarray(augmented['image'])
            mask = Image.fromarray(augmented['mask'])

        # Random elastic transformation
        if random.random() > 0.5:
            alpha = random.randint(100, 200)
            augmented = ElasticTransform(p=1, alpha=alpha, sigma=alpha * 0.05, alpha_affine=alpha * 0.03)(
                image=np.array(image), mask=np.array(mask))
            image = Image.fromarray(augmented['image'])
            mask = Image.fromarray(augmented['mask'])

        # Random GridDistortion
        if random.random() > 0.5:
            augmented = GridDistortion(p=1)(image=np.array(image), mask=np.array(mask))
            image = Image.fromarray(augmented['image'])
            mask = Image.fromarray(augmented['mask'])

        # Random OpticalDistortion
        if random.random() > 0.5:
            augmented = OpticalDistortion(p=1, distort_limit=1, shift_limit=0.5)(image=np.array(image),
                                                                                 mask=np.array(mask))
            image = Image.fromarray(augmented['image'])
            mask = Image.fromarray(augmented['mask'])

    # Transform to grayscale (1 channel)
    # image = TF.to_grayscale(image, num_output_channels=1)
    mask = TF.to_grayscale(mask, num_output_channels=1) if mask is not None else None

    # Crop the mask to the desired output size
    # mask = transforms.CenterCrop(img_output_size)(mask) if mask is not None else None

    # Transform to pytorch tensor and binarize the mask
    image = TF.to_tensor(image).float()
    # mask = binarize(TF.to_tensor(mask)).long() if mask is not None else torch.zeros(img_output_size, img_output_size)
    # mask = binarize(TF.to_tensor(mask)).float() if mask is not None else torch.zeros(img_output_size)
    mask = TF.to_tensor(np_to_pil(hysteresis_threshold(np_img=pil_to_np(mask)))).squeeze(0).float() if mask is not None else torch.zeros(img_output_size)

    return image, mask


def create_dataloader(method="1-original", batch_size=1, dataset_dir="/home/dalifreire/Documents/Doutorado/github/histological_oral_mice_tissues/roi"):

    level = "all"
    image_datasets = {x: OralMiceTissuesDataset(img_dir=dataset_dir, method=method, dysplasia_level=level,
                                                augmentation=True if x == 'train' else False,
                                                dataset_type='train' if x == 'train' else 'test') for x in ['train', 'test']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=False, num_workers=4) for x in ['train', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}

    print("Train images: {} (augmentation: {})".format(dataset_sizes['train'], image_datasets['train'].augmentation))
    print("Test images: {} (augmentation: {})".format(dataset_sizes['test'], image_datasets['test'].augmentation))
    return dataloaders


def show_image(img):
    if isinstance(img, np.ndarray) or len(img.shape) == 2:
        plt.imshow(img, cmap='gray')
    else:
        rgb = img.permute(1, 2, 0)
        plt.imshow(rgb)


def dataset_show(dataloader, batch_size=6):
    for batch_idx, (images, masks, fname, output_size) in enumerate(dataloader):

        # print('Batch {}: {}/{} images {} masks {}'.format(
        #    (batch_idx+1),
        #    (batch_idx+1) * len(images), dataset_sizes['train'],
        #    images.shape,
        #    masks.shape))

        # show 1 line of 'batch_size' images
        fig = plt.figure(figsize=(20, 20))
        for idx in np.arange(batch_size):
            ax = fig.add_subplot(1, batch_size, idx + 1, xticks=[], yticks=[])
            show_image(images[idx])
            ax.set_title(fname[idx])

        # show 1 line of 'batch_size' masks
        fig = plt.figure(figsize=(20, 20))
        for idx in np.arange(batch_size):
            ax = fig.add_subplot(1, batch_size, idx + 1, xticks=[], yticks=[])
            show_image(masks[idx])