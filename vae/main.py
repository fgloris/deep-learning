from time import time
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.transforms import ToPILImage

from load_imgnet import get_dataloader
from model import VAE

# Hyperparameters
n_epochs = 10
kl_weight = 0.0005
lr = 0.0002
grad_clip = 1.0


def loss_fn(y, y_hat, mean, logvar):
    recons_loss = torch.nn.functional.mse_loss(y_hat, y)
    kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mean**2 - logvar.exp(), 1))
    loss = recons_loss + kl_loss * kl_weight
    return loss


def setup():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup():
    dist.destroy_process_group()


def train(local_rank, dataloader, model):
    model = DDP(model, device_ids=[local_rank])
    optimizer = torch.optim.Adam(model.parameters(), lr)

    begin_time = time()
    for i in range(n_epochs):
        dataloader.sampler.set_epoch(i)
        loss_sum = 0
        for x in dataloader:
            x = x.cuda(local_rank)
            y_hat, mean, logvar = model(x)
            loss = loss_fn(x, y_hat, mean, logvar)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            loss_sum += loss.item()

        loss_sum /= len(dataloader)
        training_time = time() - begin_time
        minute = int(training_time // 60)
        second = int(training_time % 60)

        if local_rank == 0:
            print(f'epoch {i}: loss {loss_sum} {minute}:{second}')
            torch.save(model.module.state_dict(), 'model.pth')


def reconstruct(local_rank, dataloader, model):
    if local_rank != 0:
        return
    model.eval()
    batch = next(iter(dataloader))
    x = batch[0:1, ...].cuda(local_rank)
    output = model(x)[0]
    output = output[0].detach().cpu()
    input = batch[0].detach().cpu()
    combined = torch.cat((output, input), 1)
    img = ToPILImage()(combined)
    img.save('vae/result/reconstruct.jpg')


def generate(local_rank, model):
    if local_rank != 0:
        return
    model.eval()
    output = model.sample(f'cuda:{local_rank}')
    output = output[0].detach().cpu()
    img = ToPILImage()(output)
    img.save('vae/result/generate.jpg')


def main():
    local_rank = setup()
    dataloader = get_dataloader()

    model = VAE().cuda(local_rank)

    train(local_rank, dataloader, model)
    reconstruct(local_rank, dataloader, model)
    generate(local_rank, model)
    cleanup()


if __name__ == '__main__':
    main()