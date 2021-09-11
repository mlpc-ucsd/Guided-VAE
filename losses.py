import torch
import torch.nn.functional as F

def loss_unsupervised(recon_x, x, mu, logvar, recon_light, par):
    BCE = F.binary_cross_entropy(recon_x.view(-1, 1, 28, 28), x, reduction='sum')

    BCE_light = F.binary_cross_entropy(recon_light, x, reduction='sum')

    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    pd = torch.mm(par.transpose(0, 1), par).abs()
    ORG = pd.sum() - pd.trace()

    return BCE + KLD + 0.5*BCE_light + ORG, BCE, BCE_light, ORG


def loss_supervised(recon_x, x, mu, logvar):
    MSE = F.mse_loss(recon_x, x, reduction="sum")
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD, MSE