from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from pytorch_lightning import LightningDataModule


class CustomDS(LightningDataModule):
    def __init__(
        self,
        data_dir="data/customds/train_frames",
        batch_size=64,
        num_workers=1,
        img_size=64,
        shuffle=True,
    ):
        super().__init__()
        self.shuffle = shuffle
        self.data = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.Compose(
            [
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def prepare_data(self):
        # download
        ImageFolder(self.data, transform=self.transform)

    def setup(self, stage=None):
        self.dataset_train = ImageFolder(self.data, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle,
        )
