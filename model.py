import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions


class Generator(nn.Module):
    """Generator

    build Generator model

    Parametors
    ---------------------
    n_hidden: int
       dims of random vector z

    bottom_width: int
       Width when converting the output of the first layer
       to the 4-dimensional tensor

    in_ch: int
       Channel when converting the output of the first layer
       to the 4-dimensional tensor

    """

    def __init__(self, n_hidden=100, bottom_width=7, in_ch=128):
        super(Generator, self).__init__()
        self.n_hidden = n_hidden
        self.in_ch = in_ch
        self.bottom_width = bottom_width
        self.uniform = distributions.Uniform(-1, 1)

        # register parameters
        self.l0 = nn.Linear(in_features=self.n_hidden,
                            out_features=self.bottom_width*self.bottom_width*self.in_ch,
                            bias=False)
        self.dc1 = nn.ConvTranspose2d(in_channels=None,
                                      out_channels=self.in_ch//2,
                                      kernel_size=4,
                                      stride=2,
                                      padding=1,
                                      bias=False)  # (N, 3, 14, 14)
        self.dc2 = nn.ConvTranspose2d(in_channels=None,
                                      out_channels=1,
                                      kernel_size=4,
                                      stride=2,
                                      padding=1,
                                      bias=False)  # (N, 1, 24, 24)

        self.bn0 = nn.BatchNorm1d(
            num_features=self.bottom_width*self.bottom_width*self.in_ch)
        self.bn1 = nn.BatchNorm2d(num_features=self.in_ch)

    def make_hidden(self, batchsize):
        """
        Function that makes z random vector in accordance with the uniform(-1, 1)

        batchsize: int
           batchsize indicate len(z)
        """
        return self.uniform.sample(
            sample_shape=torch.Size([batchsize, self.n_hidden]))

    def forward(self, x):
        x = F.relu(self.bn0(self.l0(x)), inplace=True)
        x = x.view(-1, self.in_ch, self.bottom_width, self.bottom_width)
        x = F.relu(self.bn1(self.dc1(x)), inplace=True)
        x = F.tanh(self.dc2(x))

        return x


class Discriminator(nn.Module):
    """Discriminator

    build Discriminator model

    Parametors
    ---------------------
    in_ch: int
       Channel when converting the output of the first layer
       to the 4-dimensional tensor

    wscale: float
        std of normal initializer
    """

    def __init__(self, in_ch=1):
        super(Discriminator, self).__init__()

        # register parameters
        self.c0 = nn.Conv2d(
            in_channels=in_ch,
            out_channels=64,
            kernel_size=4,
            stride=2,
            padding=1
        )  # (N, 64, 14, 14)
        self.c1 = nn.Conv2d(
            in_channels=in_ch,
            out_channels=128,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False
        )  # (N, 128, 7, 7)
        self.l2 = nn.Linear(in_features=None, out_features=1)

        self.bn1 = nn.BatchNorm2d(num_features=128*7*7)

    def forward(self, x):
        x = F.leaky_relu(self.c0(x),
                         negative_slope=0.2, inplace=True)
        x = self.bn1(self.c1(x))
        x = F.leaky_relu(x.view(-1, self.num_flat_features(x)),
                         negative_slope=0.2, inplace=True)
        logits = self.l2(x)

        return logits

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
