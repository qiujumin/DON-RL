import numpy as np
import torch
import torch.nn as nn


m = 1.0
cm = 1e-2
um = 1e-6
mm = 1e-3
nm = 1e-9
W = 1

wavelength = 632.8 * nm
d = 8 * um
Nx = 1080
Ny = 1080
extent_x = Nx * d
extent_y = Ny * d
z = 10 * cm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MonochromaticField:
    def __init__(self,  wavelength, extent_x, extent_y, Nx, Ny, intensity=0.1 * W / (m**2)):
        self.extent_x = extent_x
        self.extent_y = extent_y

        self.dx = extent_x / Nx
        self.dy = extent_y / Ny

        self.x = self.dx * (torch.arange(Nx) - Nx // 2)
        self.y = self.dy * (torch.arange(Ny) - Ny // 2)
        self.xx, self.yy = torch.meshgrid(self.y, self.x)

        self.Nx = Nx
        self.Ny = Ny
        self.E = torch.ones((self.Ny, self.Nx)) * np.sqrt(intensity)
        self.λ = wavelength

    def set_source_amplitude(self, img):
        self.E = img

    def diffractive_layer(self, phase):
        self.E = self.E * torch.exp(1j * phase)

    def propagate(self, z):
        fft_c = torch.fft.fft2(self.E)
        c = torch.fft.fftshift(fft_c)

        fx = torch.fft.fftshift(torch.fft.fftfreq(self.Nx, d=self.dx))
        fy = torch.fft.fftshift(torch.fft.fftfreq(self.Ny, d=self.dy))
        fxx, fyy = torch.meshgrid(fy, fx)

        argument = (2 * torch.pi)**2 * ((1. / self.λ)**2 - fxx**2 - fyy**2)

        tmp = torch.sqrt(torch.abs(argument))
        kz = torch.where(argument >= 0, tmp, 1j * tmp).to(device)

        self.E = torch.fft.ifft2(
            torch.fft.ifftshift(c * torch.exp(1j * kz * z)))

    def get_intensity(self):
        return torch.real(self.E * torch.conj(self.E))

    def shortcut(self):
        self.E0 = self.E.clone()
        self.propagate(z)
        I = self.get_intensity()
        self.E = torch.clone(self.E0)

        return I


class DON(nn.Module):
    def __init__(self, layer):
        super(DON, self).__init__()
        self.layer = layer

    def forward(self, x):
        F = MonochromaticField(wavelength, extent_x, extent_y, Nx, Ny)
        F.set_source_amplitude(x)
        F.propagate(z)
        res = F.shortcut()
        F.diffractive_layer(self.layer[0])
        F.propagate(z)
        I = F.get_intensity()
        x = 0.5*res + 0.5*I

        F.set_source_amplitude(x)
        F.propagate(z)
        res = F.shortcut()
        F.diffractive_layer(self.layer[1])
        F.propagate(z)
        I = F.get_intensity()
        x = 0.5*res + 0.5*I

        F.set_source_amplitude(x)
        F.propagate(z)
        res = F.shortcut()
        F.diffractive_layer(self.layer[2])
        F.propagate(z)
        I = F.get_intensity()
        x = 0.5*res + 0.5*I

        return x
