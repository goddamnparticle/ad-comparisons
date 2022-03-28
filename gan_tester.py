import torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from modules.gan_module import GANTrainerModule


def anomaly_score(query_img, generated_img, g_features, d_features, c_lambda=0.1):
    residual_loss = torch.sum(torch.abs(query_img - generated_img))
    dscr_loss = torch.sum(torch.abs(d_features - g_features))
    return (1 - c_lambda) * residual_loss + c_lambda * dscr_loss


def main():

    img_size = 256 
    lr = 0.002
    b1 = 0.5
    b2 = 0.999
    latent_dim = 100
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    # Load test dataset.
    transform = transforms.Compose(
            [
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
    dataset = ImageFolder(root="data/customds/test_frames", transform=transform)
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

    # Load trained module module.
    module = GANTrainerModule().load_from_checkpoint(
        "lightning_logs/version_3/checkpoints/epoch=399-step=6799.ckpt"
    )
    module.freeze()
    generator = module.generator.to(device)
    discriminator = module.discriminator.to(device)

    img_scores = []
    query_imgs = []
    generated_imgs = []

    for i, data in enumerate(tqdm(dataloader)):
        query_img = data[0].to(device)
        
        z = torch.rand(1, latent_dim, requires_grad=True, device=device)
        z_optim = torch.optim.Adam([z], lr=lr, betas=(b1, b2))
        
        for j in range(200):
            generated_img = generator(z)
            g_features = discriminator.model[:11](generated_img)
            d_features = discriminator.model[:11](query_img)

            loss = anomaly_score(query_img, generated_img, g_features, d_features)
            loss.backward()
            z_optim.step()
            z.grad.zero_()

        if i == 5: break

        img_scores.append(loss.item())
        query_imgs.append(query_img)
        generated_imgs.append(generated_img)

    # Saving testing logs.
    np.save('testing_logs/img_scores.npy', np.array(img_scores))
    query_imgs = [query_img.to('cpu').detach().numpy() for query_img in query_imgs]
    generated_imgs = [generated_img.to('cpu').detach().numpy() for generated_img in generated_imgs]

    np.save('testing_logs/query_imgs.npy', np.array(query_imgs))
    np.save('testing_logs/generated_imgs.npy', np.array(generated_imgs))

if __name__ == "__main__":
    main()
