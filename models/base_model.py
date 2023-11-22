

import torch
import torch.nn as nn
from torch.nn import init
from thop import profile
from thop import clever_format

class BaseModel(nn.Module):
    def __init__(self, model, opt):
        super().__init__()
        self.model = model

        self.opt = opt
        self.use_discriminator = False

        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()),
            lr=opt.lr,
            betas=(0.99, opt.beta2),
        )

        if opt.isTrain:
            self.old_lr = opt.lr

        if opt.init:
            self.init_weights()

    def init_weights(self, gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, "weight") and (
                classname.find("Conv") != -1 or classname.find("Linear") != -1
            ):
                if self.opt.init == "normal":
                    init.normal_(m.weight.data, 0.0, gain)
                elif self.opt.init == "xavier":
                    init.xavier_normal_(m.weight.data, gain=gain)
                elif self.opt.init == "xavier_uniform":
                    init.xavier_uniform_(m.weight.data, gain=1.0)
                elif self.opt.init == "kaiming":
                    init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
                elif self.opt.init == "orthogonal":
                    init.orthogonal_(m.weight.data)
                elif self.opt.init == "":  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError(
                        "initialization method [%s] is not implemented"
                        % self.opt.init
                    )
                if hasattr(m, "bias") and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

        # propagate to children
        for m in self.children():
            if hasattr(m, "init_weights"):
                m.init_weights(self.opt.init, gain)

    def __call__(
        self, dataloader, isval=False, return_batch=False
    ):

        if isval:
            batch = next(dataloader)
            t_losses, output_images = self.model(batch,isval)
            


            if self.opt.normalize_image:
                for k in output_images.keys():
                    if "Img" in k:
                        output_images[k] = 0.5 * output_images[k] + 0.5

            if return_batch:
                return t_losses, output_images, batch
            return t_losses, output_images
        self.optimizer.zero_grad()
        t_losses, output_images = self.model(next(dataloader),isval)
        t_losses["Total Loss"].mean().backward()
        self.optimizer.step()

        if self.opt.normalize_image:
            for k in output_images.keys():
                if "Img" in k:
                    output_images[k] = 0.5 * output_images[k] + 0.5

        return t_losses, output_images
