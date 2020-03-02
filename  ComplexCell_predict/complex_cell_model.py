import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from torch.nn import functional as F
from torch.autograd import Variable


def gabor_fn(sigma, theta, Lambda, psi, gamma):
    sigma_x = sigma
    sigma_y = float(sigma) / gamma

    # Bounding box
    nstds = 3  # Number of standard deviation sigma
    xmax = max(abs(nstds * sigma_x * np.cos(theta)), abs(nstds * sigma_y * np.sin(theta)))
    xmax = np.ceil(max(1, xmax))
    ymax = max(abs(nstds * sigma_x * np.sin(theta)), abs(nstds * sigma_y * np.cos(theta)))
    ymax = np.ceil(max(1, ymax))
    xmin = -xmax
    ymin = -ymax
    (y, x) = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1))

    # Rotation
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    gb = np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * np.cos(
        2 * np.pi / Lambda * x_theta + psi)
    return gb


DEFAULT_CHANNEL = 1
DEFAULT_HEIGHT, DEFAULT_WIDTH = 144, 256


class ComplexCellModel(nn.Module):
    def __init__(self, nImages=1, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'), sigma=4,
                 theta=0.5, Lambda=10, gamma=1):
        super().__init__()
        self.device = device
        starting_images = torch.randn(nImages, DEFAULT_CHANNEL, DEFAULT_HEIGHT, DEFAULT_WIDTH).to(self.device)
        starting_images /= torch.norm(starting_images)
        self.input = Variable(starting_images, requires_grad=True).to(self.device)
        self.register_buffer('filter1', torch.tensor(gabor_fn(sigma, theta, Lambda, 0, gamma)[None, None, ...], dtype=torch.float, device=self.device))
        self.register_buffer('filter2',
                             torch.tensor(gabor_fn(sigma, theta, Lambda, np.pi / 2, gamma)[None, None, ...], dtype=torch.float, device=self.device))
        self._pad = 0

        for param in self.parameters():
            param.require_grad = False
            self.input.requires_grad_()

    def forward(self, x):
        self.input = x.clone().to(self.device)
        self.input.requires_grad_()
        y1 = F.conv2d(self.input, self.filter1, padding=self._pad, bias=None)
        y2 = F.conv2d(self.input, self.filter2, padding=self._pad, bias=None)
        return [y1.pow(2) + y2.pow(2)]
