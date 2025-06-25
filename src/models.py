import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_size, img_channels, img_size, features_g=64):
        super(Generator, self).__init__()
        self.init_size = img_size // 16
        self.fc = nn.Sequential(
            nn.Linear(latent_size, features_g * 8 * self.init_size * self.init_size),
            nn.BatchNorm1d(features_g * 8 * self.init_size * self.init_size),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv_blocks = nn.Sequential(
            nn.ConvTranspose2d(features_g * 8, features_g * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_g * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(features_g * 4, features_g * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_g * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(features_g * 2, features_g, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_g),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(features_g, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.fc(z)
        out = out.view(out.size(0), -1, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class Discriminator(nn.Module):
    def __init__(self, img_channels, img_size, features_d=64):
        super(Discriminator, self).__init__()
        final_feature_map_size = img_size // 32
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(img_channels, features_d, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features_d, features_d * 2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features_d * 2, features_d * 4, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features_d * 4, features_d * 8, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features_d * 8, features_d * 16, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(features_d * 16 * final_feature_map_size * final_feature_map_size, 1)
        )

    def forward(self, img):
        out = self.conv_blocks(img)
        score = self.fc(out)
        return score