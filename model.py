import torch
from torch import nn
import torch.nn.functional as F

class unGuidedVAE(nn.Module):
    def __init__(self, n_vae_dis=10, DIM=64):
        super().__init__()
        self.DIM = DIM
        self.layer1 = nn.Conv2d(1, DIM, 6, stride=2)
        self.layer2 = nn.Sequential(
            nn.Conv2d(DIM, 2*DIM, 5),
            nn.ReLU(True),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(2*DIM, 4*DIM, 5),
            nn.ReLU(True),
        )
        self.layer41 = nn.Linear(4*4*4*DIM, n_vae_dis)
        self.layer42 = nn.Linear(4*4*4*DIM, n_vae_dis)

        self.preprocess = nn.Sequential(
            nn.Linear(n_vae_dis, 4*4*4*DIM),
            nn.ReLU(True),
        )
        self.block1 = nn.Sequential(
            nn.ConvTranspose2d(4*DIM, 2*DIM, 5),
            nn.ReLU(True),
        )
        self.block2 = nn.Sequential(
            nn.ConvTranspose2d(2*DIM, DIM, 5),
            nn.ReLU(True),
        )
        self.deconv_out = nn.ConvTranspose2d(DIM, 1, 6, stride=2)

        self.fc1 = nn.Linear(n_vae_dis - 2, 784)
        self.fc2 = nn.Linear(1, 1)
        self.fc3 = nn.Linear(1, 2)

    def encode(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x).view(-1, self.DIM*4*4*4)
        return self.layer41(x), self.layer42(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        output = self.preprocess(z)
        output = output.view(-1, 4 * self.DIM, 4, 4)
        output = self.block1(output)
        output = self.block2(output)
        output = self.deconv_out(output)
        output = torch.sigmoid(output)
        return output.view(-1, 784)

    def light_decode(self, z):
        z_scale = z[:, 0:1]
        z_rotate = z[:, 1:2]
        z_content = z[:, 2:]
        device = z.device

        img_raw = torch.sigmoid(self.fc1(z_content).view(-1, 1, 28, 28))
        z2 = self.fc2(z_rotate).view(-1, 1)
        z3 = torch.cos(z2)
        z3 = torch.cat((z3, z3), 1)
        z3 = torch.matmul(z3.unsqueeze(2), torch.tensor([[1, 1, 0]]).float().to(device))
        z3 = z3 * torch.tensor([[1, 0, 0], [0, 1, 0]]).float().to(device)

        z4 = torch.sin(z2)
        z4 = torch.cat((z4, z4), 1)
        z4 = torch.matmul(z4.unsqueeze(2), torch.tensor([[1, 1, 0]]).float().to(device))
        z4 = z4 * torch.tensor([[0, 1, 0], [-1, 0, 0]]).float().to(device)

        grid = F.affine_grid(z3 + z4, img_raw.size(), align_corners=False)
        img_tran = F.grid_sample(img_raw, grid, align_corners=False)

        z5 = self.fc3(z_scale).view(-1, 2)
        z5 = torch.matmul(z5.unsqueeze(2), torch.tensor([[1, 1, 0]]).float().to(device))
        z5 = z5 * torch.tensor([[1, 0, 0], [0, 1, 0]]).float().to(device)

        grid = F.affine_grid(z5, img_tran.size(), align_corners=False)
        img_tran = F.grid_sample(img_tran, grid, align_corners=False)
        return img_raw, img_tran

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, self.light_decode(z), self.fc1.weight


class View(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)

class suGuidedVAE(nn.Module):
    def __init__(self, n_vae_dis=16):
        super().__init__()

        self.n_vae_dis = n_vae_dis

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 256, 4, 1),
            nn.ReLU(True),
            View((-1, 256*1*1)),
            nn.Linear(256, n_vae_dis*2),
        )

        self.decoder = nn.Sequential(
            nn.Linear(n_vae_dis, 256),
            View((-1, 256, 1, 1)),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 64, 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
        )

        self.cls_sq = nn.Sequential(
            nn.Linear(1, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(32, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)
        mu = x[:, :self.n_vae_dis]
        logvar = x[:, self.n_vae_dis:]
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        return self.decoder(z)

    def cls(self, z):
        z = torch.split(z, 1, 1)[0]
        return self.cls_sq(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, self.cls(z)

class Classifier(nn.Module):
    def __init__(self, n_vae_dis=16):
        super(Classifier, self).__init__()

        self.cls_sq = nn.Sequential(
            nn.Linear(n_vae_dis - 1, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(32, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.cls_sq(x)