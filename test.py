import torch
from torchvision.utils import save_image
from losses import loss_supervised, loss_unsupervised

def test_unsupervised(epoch, model, test_loader, out_dir, device):
    model.eval()
    test_loss = 0
    BCE_loss = 0
    BCE_light = 0
    par_loss = 0
    for i, (data, _) in enumerate(test_loader):
        data = data.to(device)
        recon_batch, mu, logvar, (recon_light_1, recon_light_2), par = model(data)
        loss = loss_unsupervised(recon_batch, data, mu, logvar, recon_light_2, par)
        test_loss += loss[0].item()
        BCE_loss += loss[1].item()
        BCE_light += loss[2].item()
        par_loss += loss[3].item()
        if i == 0:
            z, _ = model.encode(data)
            nz = z.shape[1]
            sample1 = z[0:1].repeat(10, 1)
            show = []
            for i in range(nz):
                sample = sample1.clone()
                for j in range(10):
                    sample[j][i] = -2.0 + j * 0.4
                sample = model.decode(sample).cpu().view(-1, 1, 28, 28)
                show.append(sample)
            comparison = torch.cat(show)
            save_image(comparison, out_dir + '/recon_' + str(epoch) + '.png', nrow=10)

            
            sample2 = torch.randn(1, nz).repeat(10, 1).to(device)
            show = []
            for i in range(nz):
                sample = sample2.clone()
                for j in range(10):
                    sample[j][i] = -2.0 + j * 0.4
                sample = model.decode(sample).cpu().view(-1, 1, 28, 28)
                show.append(sample)
            comparison = torch.cat(show)
            save_image(comparison, out_dir + '/sample_' + str(epoch) + '.png', nrow=10)


    num = len(test_loader.dataset)
    test_loss /= num
    BCE_loss /= num
    BCE_light /= num
    par_loss /= num
    print('====> Epoch: {} Test set loss: {:.4f}, BCE loss: {:.4f} BCE light: {:.4f} Par Loss: {:.4f}'.format(epoch, test_loss, BCE_loss, BCE_light, par_loss))


def test_supervised(epoch, model, model_c, test_loader, out_dir, device):
    model.eval()
    loss_re = 0
    for batch_idx, (data, label) in enumerate(test_loader):
        data = data.cuda()
        recon_batch, mu, logvar, re = model(data)
        loss_list = loss_supervised(recon_batch, data, mu, logvar)
        loss_re += loss_list[1].item()
        show = []
        z = mu[:5]
        nz = z.shape[1]
        for i in range(5):
            sample1 = z[i, 1:].unsqueeze(0).repeat(7, 1)
            sample2 = torch.zeros(7, 1).to(device) - 3.0
            for j in range(7):
                sample2[j] += 1.0 * j
            sample = torch.cat([sample2, sample1], 1)
            sample = model.decode(sample).cpu().view(-1, 3, 64, 64)
            show.append(sample)
        results = torch.cat(show)
        save_image(results, out_dir + '/recon_' + str(epoch) + '.png', nrow=7)

        show = []
        for i in range(5):
            sample1 = torch.randn(1, nz - 1).to(device).repeat(7, 1)
            sample2 = torch.zeros(7, 1).to(device) - 3.0
            for j in range(7):
                sample2[j] += 1.0 * j
            sample = torch.cat((sample2, sample1), 1)
            sample = model.decode(sample).cpu().view(-1, 3, 64, 64)
            show.append(sample)

        results = torch.cat(show)
        save_image(results, out_dir + '/sample_' + str(epoch) + '.png', nrow=7)
    num = len(test_loader.dataset)
    loss_re /= num
    print('====> Epoch: {} Test set recon loss: {:.4f}'.format(epoch, loss_re))



