import os
import torch
from datasets.custom_dataset import CustomDS
from pytorch_lightning import Trainer
from modules.gan_module import GANTrainerModule

def main():
    AVAIL_GPUS = min(1, torch.cuda.device_count())
    BATCH_SIZE = 64 if AVAIL_GPUS else 64
    NUM_WORKERS = int(os.cpu_count() / 2)
    IMG_SIZE = 64 
    dm = CustomDS(
        data_dir="data/customds/train_frames",
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=True,
    )
    model = GANTrainerModule(channels=3, img_size=IMG_SIZE, batch_size=BATCH_SIZE)
    trainer = Trainer(gpus=AVAIL_GPUS, max_epochs=20, progress_bar_refresh_rate=2)
    trainer.fit(model, dm)


if __name__ == "__main__":
    main()
