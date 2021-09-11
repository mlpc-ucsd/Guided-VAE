from tqdm import tqdm
import torch
import torch.nn.functional as F
from losses import loss_supervised, loss_unsupervised

def train_unsupervised(epoch, model, optimizer, train_loader, device):
    model.train()
    train_loss = 0
    bce_loss = 0
    for batch_idx, (data, _) in enumerate(tqdm(train_loader)):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar, recon_light, par = model(data)
        loss_list = loss_unsupervised(recon_batch, data, mu, logvar, recon_light[1], par)
        loss = loss_list[0]
        loss.backward()

        train_loss += loss_list[0].item()
        bce_loss += loss_list[1].item()
        optimizer.step()
    print('epoch {} | training loss {:.4f}, bce loss {:.4f}'.format(epoch, 
        train_loss / len(train_loader.dataset), bce_loss / len(train_loader.dataset)))

def train_supervised(epoch, model, model_c, optimizer, optimizer_c, dataloader, w_cls, device):
    model.train()
    re_loss = 0
    cls_error = 0
    correct = 0

    cls1_error = 0
    cls2_error = 0

    correct1 = 0
    correct2 = 0
    for batch_idx, (data, label) in enumerate(tqdm(dataloader)):
        data = data.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar, re = model(data)
        loss_list = loss_supervised(recon_batch, data, mu, logvar)
        loss = loss_list[0]
        loss_cls = F.binary_cross_entropy(re, label, reduction='sum')
        cls_error += loss_cls
        loss += loss_cls * w_cls
        loss.backward()
        re_loss += loss_list[1].item()
        optimizer.step()

        optimizer_c.zero_grad()
        z = model.reparameterize(mu, logvar).detach()
        z = z[:, 1:]
        cls1 = model_c(z)
        loss = F.binary_cross_entropy(cls1, label, reduction='sum')
        cls1_error += loss.item()
        loss *= w_cls
        loss.backward()
        optimizer_c.step()

        optimizer.zero_grad()
        mu, logvar = model.encode(data)
        z = model.reparameterize(mu, logvar)
        z = z[:, 1:]
        cls2 = model_c(z)
        label1 = torch.empty_like(label).fill_(0.5)
        loss = F.binary_cross_entropy(cls2, label1, reduction='sum')
        cls2_error += loss.item()
        loss *= w_cls
        loss.backward()
        optimizer.step()

        pred = (re + 0.5).int()
        correct += pred.eq(label.int()).sum().item()

        pred = (cls1 + 0.5).int()
        correct1 += pred.eq(label.int()).sum().item()

        pred = (cls2 + 0.5).int()
        correct2 += pred.eq(label.int()).sum().item()

    cls_error = cls_error / len(dataloader.dataset)
    cls1_error = cls1_error / len(dataloader.dataset)
    cls2_error = cls2_error / len(dataloader.dataset)
    print('====> Epoch: {} reconstruction loss: {:.4f} Cls loss: {:.4f} acc: {:.2f} cls1 loss: {:.4f} cls2 loss: {:.4f} acc1: {:.2f} acc2: {:.2f}'.format(
          epoch, re_loss / len(dataloader.dataset), cls_error, 100. * correct / len(dataloader.dataset),
          cls1_error, cls2_error, 100. * correct1 / len(dataloader.dataset),
          100. * correct2 / len(dataloader.dataset)))