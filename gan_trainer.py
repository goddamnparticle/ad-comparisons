import os
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from torch.utils.data import DataLoader
from architectures.dcgan import Generator, Discriminator, weight_init
from torchvision.datasets import ImageFolder

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 64 if AVAIL_GPUS else 64
NUM_WORKERS = int(os.cpu_count() / 2)
IMG_SIZE = 96 


class GAN(LightningModule):
    def __init__(
        self,
        channels,
        img_size,
        latent_dim: int = 100,
        lr: float = 0.0002,
        b1: float = 0.5,
        b2: float = 0.999,
        batch_size: int = BATCH_SIZE,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        # networks
        self.generator = Generator(img_size=self.hparams.img_size).apply(weight_init)
        self.discriminator = Discriminator(img_size=self.hparams.img_size).apply(weight_init)
        self.validation_z = torch.randn(9, self.hparams.latent_dim)
        self.example_input_array = torch.zeros(2, self.hparams.latent_dim)

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx):
        imgs, _ = batch

        optimizer_g, optimizer_d = self.optimizers()
        optimizer_g.zero_grad()

        # sample noise
        z = torch.randn(imgs.shape[0], self.hparams.latent_dim)
        z = z.type_as(imgs)
        # train generator
        # generate images
        self.generated_imgs = self(z)

        # ground truth result (ie: all fake)
        # put on GPU because we created this tensor inside training_loop
        valid = torch.ones(imgs.size(0), 1)
        valid = valid.type_as(imgs)
        # adversarial loss is binary cross-entropy
        g_loss = self.adversarial_loss(self.discriminator(self(z)), valid)
        self.manual_backward(g_loss)
        optimizer_g.step()

        # train discriminator
        # Measure discriminator's ability to classify real from generated samples
        # how well can it label as real?
        optimizer_d.zero_grad()
        valid = torch.ones(imgs.size(0), 1)
        valid = valid.type_as(imgs)
        real_loss = self.adversarial_loss(self.discriminator(imgs), valid)
        # how well can it label as fake?
        fake = torch.zeros(imgs.size(0), 1)
        fake = fake.type_as(imgs)
        fake_loss = self.adversarial_loss(self.discriminator(self(z).detach()), fake)
        # discriminator loss is the average of these
        d_loss = (real_loss + fake_loss) / 2
        self.manual_backward(d_loss)
        optimizer_d.step()

        self.log_dict({"d_loss": d_loss, "g_loss": g_loss}, prog_bar=True)

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []

    def on_epoch_end(self):
        z = self.validation_z.type_as(self.generator.model[0].weight)
        # log sampled images
        sample_imgs = self(z)
        grid = torchvision.utils.make_grid(sample_imgs, padding=2, nrow=3, normalize=True)
        self.logger.experiment.add_image("generated_images", grid, self.current_epoch)


class CustomDS(LightningDataModule):
    def __init__(
        self,
        data_dir: str = PATH_DATASETS,
        batch_size: int = BATCH_SIZE,
        num_workers: int = NUM_WORKERS,
        img_size: int = IMG_SIZE,
    ):
        super().__init__()
        self.data_dir_train = "data/customds/train_frames"
        self.data_dir_test = "data/customds/test_frames"
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = transforms.Compose(
            [
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        # self.dims is returned when you call dm.size()
        # Setting default dims here because we know them.
        # Could optionally be assigned dynamically in dm.setup()
        # self.dims = (3, 64, 64)

    def prepare_data(self):
        # download
        ImageFolder(self.data_dir_train, transform=self.transform)
        ImageFolder(self.data_dir_test, transform=self.transform)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.dataset_train = ImageFolder(self.data_dir_train, transform=self.transform)

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.dataset_test = ImageFolder(self.data_dir_test)

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )


dm = CustomDS()
model = GAN(channels=3, img_size=IMG_SIZE)
trainer = Trainer(gpus=AVAIL_GPUS, max_epochs=20, progress_bar_refresh_rate=1)
trainer.fit(model, dm)
