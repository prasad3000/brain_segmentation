import json
import os

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from BrainTumorDataset import BrainSegmentationDataset as Dataset
from logger import Logger
from loss import DiceLoss
from transform import transforms
from model import UNet
from utils import log_images, dsc


def main():
    
    weights = './weights'
    logs = './logs'
    makedirs(weights, logs)
    #snapshot(logs)
    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda:0")
    
    images = './kaggle_3m'
    image_size = 256
    scale = 0.05
    angle = 15
    batch_size = 16
    workers = 4

    loader_train, loader_valid = data_loaders(images, image_size, scale, angle, batch_size, workers)
    loaders = {"train": loader_train, "valid": loader_valid}

    unet = UNet(in_channels=Dataset.in_channels, out_channels=Dataset.out_channels)
    unet.to(device)

    dsc_loss = DiceLoss()
    best_validation_dsc = 0.0
    
    lr = 0.0001

    optimizer = optim.Adam(unet.parameters(), lr)

    logger = Logger(logs)
    loss_train = []
    loss_valid = []

    step = 0
    epochs = 100
    vis_images = 200
    vis_freq = 10

    for epoch in tqdm(range(epochs), total = epochs):
        for phase in ["train", "valid"]:
            if phase == "train":
                unet.train()
            else:
                unet.eval()

            validation_pred = []
            validation_true = []

            for i, data in enumerate(loaders[phase]):
                if phase == "train":
                    step += 1

                x, y_true = data
                x, y_true = x.to(device), y_true.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    y_pred = unet(x)

                    loss = dsc_loss(y_pred, y_true)

                    if phase == "valid":
                        loss_valid.append(loss.item())
                        y_pred_np = y_pred.detach().cpu().numpy()
                        validation_pred.extend(
                            [y_pred_np[s] for s in range(y_pred_np.shape[0])]
                        )
                        y_true_np = y_true.detach().cpu().numpy()
                        validation_true.extend(
                            [y_true_np[s] for s in range(y_true_np.shape[0])]
                        )
                        if (epoch % vis_freq == 0) or (epoch == epochs - 1):
                            if i * batch_size < vis_images:
                                tag = "image/{}".format(i)
                                num_images = vis_images - i * batch_size
                                logger.image_list_summary(
                                    tag,
                                    log_images(x, y_true, y_pred)[:num_images],
                                    step,
                                )

                    if phase == "train":
                        loss_train.append(loss.item())
                        loss.backward()
                        optimizer.step()

                if phase == "train" and (step + 1) % 10 == 0:
                    log_loss_summary(logger, loss_train, step)
                    loss_train = []

            if phase == "valid":
                log_loss_summary(logger, loss_valid, step, prefix="val_")
                mean_dsc = np.mean(
                    dsc_per_volume(
                        validation_pred,
                        validation_true,
                        loader_valid.dataset.patient_slice_index,
                    )
                )
                logger.scalar_summary("val_dsc", mean_dsc, step)
                if mean_dsc > best_validation_dsc:
                    best_validation_dsc = mean_dsc
                    torch.save(unet.state_dict(), os.path.join(weights, "unet.pt"))
                loss_valid = []

    print("Best validation mean DSC: {:4f}".format(best_validation_dsc))


def data_loaders(images, image_size, scale, angle, batch_size, workers):
    dataset_train, dataset_valid = datasets(images, image_size, scale, angle)

    def worker_init(worker_id):
        np.random.seed(42 + worker_id)

    loader_train = DataLoader(
        dataset_train,
        batch_size = batch_size,
        shuffle = True,
        drop_last = True,
        num_workers = workers,
        worker_init_fn = worker_init,
    )
    loader_valid = DataLoader(
        dataset_valid,
        batch_size = batch_size,
        drop_last = False,
        num_workers = workers,
        worker_init_fn = worker_init,
    )

    return loader_train, loader_valid


def datasets(images, image_size, scale, angle):
    train = Dataset(
        images_dir=images,
        subset="train",
        image_size = image_size,
        transform=transforms(scale, angle, flip_prob=0.5),
    )
    valid = Dataset(
        images_dir=images,
        subset="validation",
        image_size = image_size,
        random_sampling=False,
    )
    return train, valid


def dsc_per_volume(validation_pred, validation_true, patient_slice_index):
    dsc_list = []
    num_slices = np.bincount([p[0] for p in patient_slice_index])
    index = 0
    for p in range(len(num_slices)):
        y_pred = np.array(validation_pred[index : index + num_slices[p]])
        y_true = np.array(validation_true[index : index + num_slices[p]])
        dsc_list.append(dsc(y_pred, y_true))
        index += num_slices[p]
    return dsc_list


def log_loss_summary(logger, loss, step, prefix=""):
    logger.scalar_summary(prefix + "loss", np.mean(loss), step)


def makedirs(weights, logs):
    os.makedirs(weights, exist_ok=True)
    os.makedirs(logs, exist_ok=True)


def snapshot(logs):
    logs_file = os.path.join(logs, "logs.json")
    with open(logs_file, "w") as fp:
        json.dump(vars(logs), fp)


if __name__ == "__main__":
    main()