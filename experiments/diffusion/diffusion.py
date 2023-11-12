import fire, math
import random

import torch
from torch import nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
from torchvision.utils import make_grid

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import clearbox as cb
from clearbox.tools import here, d, fc, gradient_norm, tic, toc

from pathlib import Path

import tqdm, wandb

""" 
Implementations of the basic idea behind diffusion. 
"""

def naive(
        epochs=5,
        steps=60,
        k=32,
        lr=3e-4,
        bs=16,
        limit=float('inf'), # limits the number of batches per epoch,
        imsize=(32, 32)
     ):
    """
    A Naive approach to diffussion with very little math.

    :param epochs:
    :param steps:
    :param k:
    :param lr:
    :param bs:
    :param limit:
    :param imsize:
    :return:
    """

    h,w = imsize

    # Load MNIST and scale up to 32x32, with color channels
    transform = transforms.Compose(
        [torchvision.transforms.Grayscale(num_output_channels=3),
         transforms.Resize((h, w)),
         transforms.ToTensor()])

    data = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(data, batch_size=bs, shuffle=True, num_workers=2)

    unet = cb.diffusion.UNet(a=8, b=12, c=16, res=(h, w), numouts=3)

    opt = torch.optim.Adam(lr=lr, params=unet.parameters())

    Path(here(__file__,'samples_naive/')).mkdir(parents=True, exist_ok=True)

    for e in range(epochs):
        for i, (batch, _) in (bar := tqdm.tqdm(enumerate(dataloader))):
            if i > limit:
                break

            for s in range(steps):

                old_batch = batch.clone()

                if i == 0 and (s+1) % 10 == 0:
                    grid = make_grid(batch, nrow=4).permute(1, 2, 0)
                    plt.imshow(grid)
                    plt.savefig(here(__file__, f'samples_naive/noising-{s:05}.png'))

                # Sample a random binary tensor, the same size as the batch
                noise = torch.rand(size=(bs, 1, h, w)).round().expand(bs, 3, h, w)

                # Sample `k` pixel indices to apply the noise to
                indices = torch.randint(low=0, high=32, size=(k, 2))
                # -- To keep things simple, we'll corrupt the same pixels for each image in the batch. Whether they are made
                #    black or white still differs per image.

                # Change the values of the sampled indices to those of the random tensor
                batch[:, :, indices[:, 0], indices[:, 1]] = noise[:, :, indices[:, 0], indices[:, 1]]

                # Train the model to denoise
                output = unet(batch, step=s/steps).sigmoid()

                loss = ((output - old_batch) ** 2.0).mean()
                # loss = F.binary_cross_entropy(output, old_batch).mean()

                loss.backward()
                opt.step()
                opt.zero_grad()

                bar.set_postfix({'loss' : loss.item()})

        with torch.no_grad():
            # Sample from the model
            batch = torch.rand(size=(bs, 1, h, w)).round().expand(bs, 3, h, w)
            # -- This is the distribution to which our noising process above converges
            for s in (bar := tqdm.trange(steps)):
                batch = unet(batch, step=(steps-s-1)/steps).sigmoid()

                if (s+1) % 10 == 0:
                    grid = make_grid(batch.clip(0, 1), nrow=4).permute(1, 2, 0)
                    plt.imshow(grid)
                    plt.savefig(here(__file__,f'samples_naive/denoising-{e}-{s:05}.png'))

def data(name, data_dir, batch_size, nw=2):

    if name == 'mnist':
        h, w = 32, 32
        # Load MNIST and scale up to 32x32, with color channels
        transform = transforms.Compose(
            [torchvision.transforms.Grayscale(num_output_channels=3),
             transforms.Resize((h, w)),
             transforms.ToTensor()])

        dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=nw)

        return dataloader, (h, w)

    if name == 'mnist-128':
        h, w = 128, 128
        # Load MNIST and scale up to 32x32, with color channels
        transform = transforms.Compose(
            [torchvision.transforms.Grayscale(num_output_channels=3),
             transforms.Resize((h, w)),
             transforms.ToTensor()])

        dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=nw)

        return dataloader, (h, w)

    if name == 'ffhq-thumbs':

        h, w = 128, 128
        # Load MNIST and scale up to 32x32, with color channels
        transform = transforms.Compose(
            [transforms.ToTensor()])

        dataset = (torchvision.datasets.ImageFolder(root=data_dir, transform=transform))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=nw)

        return dataloader, (h, w)

    if name == 'ffhq':

        h, w = 1024, 1024
        # Load MNIST and scale up to 32x32, with color channels
        transform = transforms.Compose(
            [transforms.ToTensor()])

        dataset = (torchvision.datasets.ImageFolder(root=data_dir, transform=transform))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=nw)

        return dataloader, (h, w)

    fc(name)

def naive2(
        epochs = 5,
        steps = 60,
        lr = 3e-4,
        bs = 16,              # Training batch size
        sample_bs = 16,       # Sampling batch size
        limit = float('inf'), # limits the number of batches per epoch,
        algorithm2 = False,
        fix_noise = False,  # When "renoising", always use the same noise
        data_name = 'mnist',
        data_dir = None,
        unet_channels = (16, 32, 64), # Basic structure of the UNet in channels per block
        debug = False,         # Debugging mode bypasses wandb
        name = '',
        res_cat = False,
        warmup = 1000,
        gc = 1.0,
        dp = False,
        plot_renoised = False,
        plot_every = 10,       # Plots every n steps of the sampling process
        num_workers = 2,       # Number fo workers for the data loader
        blocks_per_level = 3   # Number of Res blocks per level of the UNet
    ):
    """

    A second naive approach. This method trains the model to fully denoise and then samples
    by a denoising/renoising process

    :param epochs:
    :param steps:
    :param k:
    :param lr:
    :param bs:
    :param limit:
    :param imsize:
    :return:
    """

    wd = wandb.init(
        name = f'naive2-{name}-{data_name}',
        project = 'diffusion',
        tags = [],
        config =locals(),
        mode = 'disabled' if debug else 'online'
    )

    scaler = torch.cuda.amp.GradScaler()

    tic()
    dataloader, (h, w) = data(data_name, data_dir, batch_size=bs, nw=num_workers)
    print(f'data loaded ({toc():.4} s)')

    unet = cb.diffusion.UNet(res=(h, w), channels=unet_channels, num_blocks=blocks_per_level, mid_layers=3, res_cat=res_cat)

    if torch.cuda.is_available():
        unet = unet.cuda()

    if dp:
        unet = torch.nn.DataParallel(unet)
    print('Model created.')

    opt = torch.optim.Adam(lr=lr, params=unet.parameters())
    if warmup > 0:
        sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(i / (warmup / bs), 1.0))

    # Create an output directory to put the samples
    Path('./samples_naive2/').mkdir(parents=True, exist_ok=True)

    # Create a list containing all pixel indices for quick sampling
    indices = [(x, y) for x in range(h) for y in range(w)]
    random.shuffle(indices)
    total = len(indices)
    print('Pixel indices created.')

    for e in range(epochs):

        for i, (batch, _) in (bar := tqdm.tqdm(enumerate(dataloader))):
            if i > limit:
                break

            if torch.cuda.is_available():
                batch = batch.cuda()

            initial_batch = batch.clone()

            t = random.randrange(1, len(indices))

            random.shuffle(indices)
            batch = add_noise(batch, t, indices)

            # Train the model to denoise
            with torch.cuda.amp.autocast():
                output = unet(batch, time=t/total).sigmoid()

                loss = ((output - initial_batch) ** 2.0).mean()
                # -- We predict the _fully_ denoised batch. This will be blurry for high t, but we fix this in the sampling.

            scaler.scale(loss).backward()

            bar.set_postfix({'loss' : loss.item()})
            wandb.log({
                'loss': loss,
                'gradient_norm': gradient_norm(unet),
            })

            if gc > 0.0:
                nn.utils.clip_grad_norm_(unet.parameters(), gc)

            scaler.step(opt)
            scaler.update()

            opt.zero_grad()

            if warmup > 0:
                sch.step()

        with ((torch.no_grad())):
            # Sample from the model

            batch = torch.rand(size=(sample_bs, 1, h, w), device=d()).round().expand(sample_bs, 3, h, w)
            # -- This is the distribution to which our noising process above converges

            noise = torch.rand(size=(sample_bs, 1, h, w), device=d()).round().expand(sample_bs, 3, h, w) \
                    if fix_noise else None;

            for s in (bar := tqdm.trange(steps)):

                t = (steps-s)/steps  # where we are in the denoising process from 0 to 1
                tm1 = (steps-s-1)/steps  # next noise level, t-1

                denoised = unet(batch, time=t / total).sigmoid()  # denoise

                if not fix_noise:
                    random.shuffle(indices)

                # -- note that the indices are _not_ shuffled, so that the noise is kept the same between steps

                if not algorithm2:
                    batch = add_noise(denoised, int(tm1 * total), indices, noise=noise) # renoise
                else:
                    batch = batch - add_noise(denoised, int(t * total), indices, noise=noise) \
                                  + add_noise(denoised, int(tm1 * total), indices, noise=noise)

                if (s+1) % plot_every == 0:
                    grid = make_grid(denoised.cpu().clip(0, 1), nrow=4).permute(1, 2, 0)
                    plt.imshow(grid)
                    plt.savefig(f'./samples_naive2/denoised-{e}-{s:05}.png')

                    if plot_renoised:
                        grid = make_grid(batch.cpu().clip(0, 1), nrow=4).permute(1, 2, 0)
                        plt.imshow(grid)
                        plt.savefig(f'./samples_naive2/renoised-{e}-{s:05}.png')

def add_noise(batch, t, indices, noise=None):
    """
    Take a batch of clean images, and add noise to level t
    :param batch:
    :param indices:
    :param t:
    :return:
    """
    batch = batch.clone()

    if t == 0:
        return batch

    b, c, h, w = batch.size()

    # sample a random t between 1 and 32^2 pixels, and then sample t indices in the image
    indt = torch.tensor(indices[:t])

    # Sample a random binary tensor, the same size as the batch
    if noise is None:
        noise = torch.rand(size=(b, 1, h, w), device=d()).round().expand(b, 3, h, w)
    # -- To keep things simple, we'll corrupt the same pixels for each image in the batch. Whether they are made
    #    black or white still differs per image.

    # Change the values of the sampled indices to those of the random tensor
    batch[:, :, indt[:, 0], indt[:, 1]] = noise[:, :, indt[:, 0], indt[:, 1]]

    return batch

def gaussian_constant(
        epochs=5,
        steps=60,
        k=32,
        lr=3e-4,
        bs=16,
        limit=float('inf'), # limits the number of batches per epoch,
        imsize=(32, 32),
        beta=0.9
    ):

    """
    Gaussian diffusion with constant parameters. That every noising step has

    $$z_{t+1} ~ N(\beta z_{t}, \sigma I)$$

    with fixed constants $\beta$ and $\sigma$. To ensure that the process converges to a standard normal distribution
    we set $0 < \beta <1$ and $\sigma = 1 - \beta^2$.

    :param epochs:
    :param steps:
    :param k:
    :param lr:
    :param bs:
    :param limit:
    :param imsize:
    :return:
    """

    h,w = imsize

    sigma = 1 - beta ** 2

    # Load MNIST and scale up to 32x32, with color channels
    transform = transforms.Compose(
        [torchvision.transforms.Grayscale(num_output_channels=3),
         transforms.Resize((h, w)),
         transforms.ToTensor()])

    data = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(data, batch_size=bs, shuffle=True, num_workers=2)

    unet = cb.diffusion.UNet(res=(h, w), numouts=3)

    opt = torch.optim.Adam(lr=lr, params=unet.parameters())

    Path(here(__file__,'samples_gc/')).mkdir(parents=True, exist_ok=True)

    for e in range(epochs):
        for i, (batch, _) in (bar := tqdm.tqdm(enumerate(dataloader))):
            if i > limit:
                break

            for s in range(steps):

                old_batch = batch.clone()

                if i == 0 and (s+1) % 10 == 0:
                    grid = make_grid(batch, nrow=4).permute(1, 2, 0)
                    plt.imshow(grid)
                    plt.savefig(here(__file__, f'samples_gc/noising-{s:05}.png'))

                with torch.no_grad():
                    # Sample some stand-normally distriobuted noise
                    noise = torch.randn(size=(bs, 3, h, w), device=d())
                    # add noise to the current batch
                    batch = batch * beta + noise * math.sqrt(sigma)

                # Train the model to denoise
                output = unet(batch, step=s/steps).sigmoid()

                loss = ((output - old_batch) ** 2.0).mean()

                loss.backward()
                opt.step()
                opt.zero_grad()

                bar.set_postfix({'loss' : loss.item()})

        with torch.no_grad():
            # Sample from the model
            batch = torch.rand(size=(bs, 1, h, w)).round().expand(bs, 3, h, w)
            # -- This is the distribution to which our noising process above converges
            for s in (bar := tqdm.trange(steps)):

                batch = unet(batch, step=(steps-s)/steps).sigmoid()

                if (s+1) % 10 == 0:
                    grid = make_grid(batch.clip(0, 1), nrow=4).permute(1, 2, 0)
                    plt.imshow(grid)
                    plt.savefig(here(__file__,f'samples_gc/denoising-{e}-{s:05}.png'))


if __name__ == '__main__':
    fire.Fire()