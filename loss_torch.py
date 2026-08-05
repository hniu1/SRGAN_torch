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
        # Establish accurate temperatures before introducing adversarial loss.
        # MSE controls large errors while a small L1 term improves robustness.
        return self.loss_fn1(out, hr) + 0.1 * self.loss_fn2(out, hr)
    

class WithLoss_G_old(nn.Module):
    def __init__(self, D_net, G_net, loss_fn1, loss_fn2, loss_fn3, w1_fn1=1e-4, w2_fn2=1e4):
        super(WithLoss_G, self).__init__()
        self.D_net = D_net
        self.G_net = G_net
        self.loss_fn1 = loss_fn1
        self.loss_fn2 = loss_fn2
        self.loss_fn3 = loss_fn3
        self.w1_fn1 = w1_fn1
        self.w2_fn2 = w2_fn2

    def forward(self, lr, hr):
        fake_patches = self.G_net(lr)
        logits_fake = self.D_net(fake_patches)

        g_gan_loss = self.w1_fn1 * self.loss_fn1(logits_fake, torch.ones_like(logits_fake)) # one means real image, default 1e-4
        # g_gan_loss = torch.mean(g_gan_loss)
        
        mse_loss = self.w2_fn2 * self.loss_fn2(fake_patches, hr) # 1e4
        g_loss = mse_loss + g_gan_loss
        return g_loss

class WithLoss_G(nn.Module):
    def __init__(self, D_net, loss_fn_gan, loss_fn_content, loss_fn_abs,
                 w_gan=1e-4, w_content=1e4):
        super().__init__()
        self.D_net = D_net
        self.loss_fn_gan = loss_fn_gan
        self.loss_fn_content = loss_fn_content
        self.loss_fn_abs = loss_fn_abs
        self.w_gan = w_gan
        self.w_content = w_content

    def forward(self, hr, fake):
        # GAN loss (D is frozen during G step)
        logits_fake = self.D_net(fake)
        g_gan_loss = self.loss_fn_gan(
            logits_fake, torch.ones_like(logits_fake)
        )

        # Content loss
        content_loss = self.loss_fn_content(fake, hr)

        g_loss = self.w_content * content_loss + self.w_gan * g_gan_loss
        return g_loss
    
class WithLoss_D_old(nn.Module):
    def __init__(self, D_net, G_net, loss_fn):
        super().__init__()
        self.D_net = D_net
        self.G_net = G_net
        self.loss_fn = loss_fn

    def forward(self, lr, hr):
        # ----- Real pass -----
        logits_real = self.D_net(hr)
        real_labels = torch.ones_like(logits_real)
        d_loss_real = self.loss_fn(logits_real, real_labels)

        # ----- Fake pass -----
        with torch.no_grad():
            fake_patches = self.G_net(lr)
        logits_fake = self.D_net(fake_patches.detach())
        fake_labels = torch.zeros_like(logits_fake)
        d_loss_fake = self.loss_fn(logits_fake, fake_labels)

        # ----- Combine -----
        d_loss = d_loss_real + d_loss_fake
        return d_loss


class WithLoss_D(nn.Module):
    def __init__(self, D_net, loss_fn):
        super().__init__()
        self.D_net = D_net
        self.loss_fn = loss_fn

    def forward(self, hr, fake):
        """
        hr:   real high-res image
        fake: fake image, MUST be detached
        """
        # ----- Real pass -----
        logits_real = self.D_net(hr)
        real_labels = torch.ones_like(logits_real)
        d_loss_real = self.loss_fn(logits_real, real_labels)

        # ----- Fake pass -----
        logits_fake = self.D_net(fake)
        fake_labels = torch.zeros_like(logits_fake)
        d_loss_fake = self.loss_fn(logits_fake, fake_labels)

        return d_loss_real + d_loss_fake
