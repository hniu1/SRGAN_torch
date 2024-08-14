import torch
import torch.nn as nn
import torch.nn.functional as F


class WithLoss_init(nn.Module):
    def __init__(self, G_net, loss_fn1, loss_fn2):
        super(WithLoss_init, self).__init__()
        self.G_net = G_net
        self.loss_fn1 = loss_fn1
        self.loss_fn2 = loss_fn2

    def forward(self, lr, hr):
        out = self.G_net(lr)
        loss1 = self.loss_fn1(out, hr)
        # loss2 = self.loss_fn2(out, hr)
        return loss1
    

class WithLoss_G(nn.Module):
    def __init__(self, D_net, G_net, loss_fn1, loss_fn2, loss_fn3):
        super(WithLoss_G, self).__init__()
        self.D_net = D_net
        self.G_net = G_net
        self.loss_fn1 = loss_fn1
        self.loss_fn2 = loss_fn2
        self.loss_fn3 = loss_fn3

    def forward(self, lr, hr):
        fake_patches = self.G_net(lr)
        logits_fake = self.D_net(fake_patches)

        g_gan_loss = 1e-4 * self.loss_fn1(logits_fake, torch.ones_like(logits_fake)) # one means real image, default 1e-4
        g_gan_loss = torch.mean(g_gan_loss)
        
        mse_loss = 1e4 * self.loss_fn2(fake_patches, hr)
        g_loss = mse_loss + g_gan_loss
        return g_loss
    
class WithLoss_D(nn.Module):
    def __init__(self, D_net, G_net, loss_fn):
        super(WithLoss_D, self).__init__()
        self.D_net = D_net
        self.G_net = G_net
        self.loss_fn = loss_fn

    def forward(self, lr, hr):
        fake_patches = self.G_net(lr)
        logits_fake = self.D_net(fake_patches)
        logits_real = self.D_net(hr)
        d_loss_real = self.loss_fn(logits_real, torch.ones_like(logits_real))
        d_loss_real = torch.mean(d_loss_real)
        d_loss_fake = self.loss_fn(logits_fake, torch.zeros_like(logits_fake))
        d_loss_fake = torch.mean(d_loss_fake)
        d_loss = d_loss_real + d_loss_fake
        return d_loss