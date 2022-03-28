import os
import torch
import torchvision
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from architectures.dcgan import Generator, Discriminator, weight_init

class GANTrainerModule(LightningModule):
    def __init__(
        self,
        channels=3,
        img_size=64,
        latent_dim=100,
        lr=0.0002,
        b1=0.5,
        b2=0.999,
        batch_size=64,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        # networks
        self.generator = Generator(img_size=self.hparams.img_size).apply(weight_init)
        self.discriminator = Discriminator(img_size=self.hparams.img_size).apply(weight_init)
        self.validation_z = torch.randn(9, self.hparams.latent_dim)
        self.example_input_array = torch.zeros(1, self.hparams.latent_dim)

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch):
        imgs, _ = batch
        optimizer_g, optimizer_d = self.optimizers()

        # Generator training phase.
        optimizer_g.zero_grad()
        z = torch.randn(imgs.shape[0], self.hparams.latent_dim)
        z = z.type_as(imgs)
        self.generated_imgs = self(z)
        valid = torch.ones(imgs.size(0), 1)
        valid = valid.type_as(imgs)
        g_loss = self.adversarial_loss(self.discriminator(self(z)), valid)
        self.manual_backward(g_loss)
        optimizer_g.step()

        # Discriminator training phase.
        optimizer_d.zero_grad()
        real_loss = self.adversarial_loss(self.discriminator(imgs), valid)
        fake = torch.zeros(imgs.size(0), 1)
        fake = fake.type_as(imgs)
        fake_loss = self.adversarial_loss(self.discriminator(self(z).detach()), fake)
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
