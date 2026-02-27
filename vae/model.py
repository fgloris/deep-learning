import torch
import torch.nn as nn


class VAE(nn.Module):
    """VAE for 128x128 face generation.

    The hidden dimensions can be tuned.
    """

    def __init__(self, hiddens=[64, 128, 256, 512, 1024], latent_dim=512) -> None:
        super().__init__()

        # encoder
        prev_channels = 3
        modules = []
        img_length = 128
        for cur_channels in hiddens:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(prev_channels,
                              cur_channels,
                              kernel_size=3,
                              stride=2,
                              padding=1), 
                    nn.BatchNorm2d(cur_channels),
                    nn.ReLU()))
            prev_channels = cur_channels
            img_length //= 2
        self.encoder = nn.Sequential(*modules)
        self.mean_linear = nn.Linear(prev_channels * img_length * img_length,
                                     latent_dim)
        self.var_linear = nn.Linear(prev_channels * img_length * img_length,
                                    latent_dim)
        self.latent_dim = latent_dim
        # decoder
        modules = []
        self.decoder_projection = nn.Linear(
            latent_dim, prev_channels * img_length * img_length)
        self.decoder_input_chw = (prev_channels, img_length, img_length)
        for i in range(len(hiddens) - 1, 0, -1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hiddens[i],
                                       hiddens[i - 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hiddens[i - 1]), nn.ReLU()))
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(hiddens[0],
                                   hiddens[0],
                                   kernel_size=3,
                                   stride=2,
                                   padding=1,
                                   output_padding=1),
                nn.BatchNorm2d(hiddens[0]), nn.ReLU(),
                nn.Conv2d(hiddens[0], 3, kernel_size=3, stride=1, padding=1),
                nn.Sigmoid()))
        self.decoder = nn.Sequential(*modules)

    def forward(self, x):
        encoded = self.encoder(x)
        encoded = torch.flatten(encoded, 1)
        mean = self.mean_linear(encoded)  # 把卷积的encoder输出用一个全连接映射到潜空间的 μ
        logvar = self.var_linear(encoded) # 把卷积的encoder输出映射到潜空间的 logσ²
                                          # 图像空间是128x128, 压缩到512维的潜空间，即压缩了32倍。
        epsilon = torch.randn_like(logvar)
        std = torch.exp(logvar / 2)
        z = epsilon * std + mean          # 重参数化技巧: z = μ + σ·ε, 其中: ε ~ N(0, I)
        x = self.decoder_projection(z)    # 把潜空间的点z映射到decoder像素维度
        x = torch.reshape(x, (-1, *self.decoder_input_chw))
        decoded = self.decoder(x)

        return decoded, mean, logvar, epsilon

    def sample(self, device='cuda'):
        z = torch.randn(1, self.latent_dim).to(device)
        x = self.decoder_projection(z)
        x = torch.reshape(x, (-1, *self.decoder_input_chw))
        decoded = self.decoder(x)
        return decoded