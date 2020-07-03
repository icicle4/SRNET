import torch
import torch.nn as nn
from torchvision.transforms import transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from dataset import SuperResolutionDataset


class LightSuperResolutioner(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.conv0_01 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.conv0_02 = nn.Conv2d(64, 96, 3, stride=1, padding=1)
        self.conv0_03 = nn.Conv2d(96, 96, 3, stride=1, padding=1)
        self.conv0_04 = nn.Conv2d(96, 96, 3, stride=1, padding=1)

        self.conv0_1 = nn.Conv2d(96, 64, 1, stride=1)

        self.conv0_21 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv0_22 = nn.Conv2d(64, 3, 3, stride=1, padding=1)

        self.dataset_roots = ['/Users/linda/Downloads/Asian_dataset',
                         '/Users/linda/Downloads/cfp_datset',
                         '/Users/linda/Downloads/lfw-deepfunneled']

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.augmentation = transforms.Compose([
            transforms.ToTensor(),
            self.normalize
        ])

    def forward(self, x):
        x = self.conv0_01(x)
        x = self.conv0_02(x)
        x = self.conv0_03(x)
        x = self.conv0_04(x)

        x = self.conv0_1(x)

        x = self.conv0_21(x)
        x = self.conv0_22(x)
        return x

    def training_step(self, batch):
        low_img, hr_img = batch
        loss = F.mse_loss(self(low_img), hr_img, size_average=True)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch):
        low_img, hr_img = batch
        loss = F.mse_loss(self(low_img), hr_img, size_average=True)
        tensorboard_logs = {'val_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

    def train_dataloader(self):
        train_dataset = SuperResolutionDataset(self.dataset_roots, (244, 244), 'train', self.augmentation)
        train_loader = DataLoader(train_dataset, batch_size=32, num_workers=8, pin_memory=True, drop_last=True)
        return train_loader

    def val_dataloader(self):
        val_dataset = SuperResolutionDataset(self.dataset_roots, (244, 244), 'val', self.augmentation)
        val_loader = DataLoader(val_dataset, batch_size=32, num_workers=8, pin_memory=True, drop_last=True)
        return val_loader


def main():
    model = LightSuperResolutioner()
    trainer = pl.Trainer(gpus=1, precision=16, auto_lr_find=True, check_val_every_n_epoch=10)
    trainer.fit(model)


if __name__ == '__main__':
    main()
