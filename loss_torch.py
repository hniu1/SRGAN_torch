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


class WithLoss_G_balanced(nn.Module):
    """Numerically balanced GAN/content objective for climate fields.

    Kept separate from ``WithLoss_G`` so existing model versions remain fully
    reproducible.  Inputs are scaled temperature fields.
    """

    def __init__(self, D_net, loss_fn_gan, loss_fn_content, loss_fn_abs,
                 w_gan=1e-4, w_content=1.0, w_abs=0.1,
                 w_gradient=0.1):
        super().__init__()
        self.D_net = D_net
        self.loss_fn_gan = loss_fn_gan
        self.loss_fn_content = loss_fn_content
        self.loss_fn_abs = loss_fn_abs
        self.w_gan = float(w_gan)
        self.w_content = float(w_content)
        self.w_abs = float(w_abs)
        self.w_gradient = float(w_gradient)

    @staticmethod
    def gradient_loss(fake, hr):
        """L1 loss on first differences to discourage over-smoothing."""
        fake_dx = fake[..., :, 1:] - fake[..., :, :-1]
        hr_dx = hr[..., :, 1:] - hr[..., :, :-1]
        fake_dy = fake[..., 1:, :] - fake[..., :-1, :]
        hr_dy = hr[..., 1:, :] - hr[..., :-1, :]
        return 0.5 * (
            torch.mean(torch.abs(fake_dx - hr_dx))
            + torch.mean(torch.abs(fake_dy - hr_dy))
        )

    def components(self, hr, fake):
        logits_fake = self.D_net(fake)
        gan = self.loss_fn_gan(logits_fake, torch.ones_like(logits_fake))
        mse = self.loss_fn_content(fake, hr)
        absolute = self.loss_fn_abs(fake, hr)
        gradient = self.gradient_loss(fake, hr)
        content = mse + self.w_abs * absolute + self.w_gradient * gradient
        total = self.w_content * content + self.w_gan * gan
        return total, content, gan, mse, absolute, gradient

    def forward(self, hr, fake):
        return self.components(hr, fake)[0]
    
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
