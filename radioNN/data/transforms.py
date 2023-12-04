"""
Tranforms to be used by the dataloader class
"""
import numpy as np
import torch


def get_module(array):
    if isinstance(array, np.ndarray):
        module = np
    elif isinstance(array, torch.Tensor):
        module = torch
    return module


def cart2sph(antenna_pos):
    module = get_module(antenna_pos)
    x = antenna_pos.T[0]
    y = antenna_pos.T[1]
    rho = module.sqrt(x**2 + y**2) / 250
    phi = module.arctan2(y, x) / np.pi
    # TODO: Use stack
    antenna_pos.T[0] = rho
    antenna_pos.T[1] = phi
    return antenna_pos


def pulse2spec(pulse, mask):
    mm = get_module(pulse)
    fft = mm.fft.rfft(pulse.T, axis=1)[:, mask]
    spec = mm.abs(fft) / 6e-3
    phase = mm.angle(fft) / np.pi
    return mm.cat([spec, phase], axis=0).T


def sph2cart(antenna_pos):
    module = get_module(antenna_pos)
    rho = antenna_pos.T[0] * 250
    phi = antenna_pos.T[1] * np.pi
    antenna_pos.T[0] = rho * module.cos(phi)
    antenna_pos.T[1] = rho * module.sin(phi)
    return antenna_pos


class Identity:
    """
    Identity transformation
    """

    def __call__(self, event_data, meta_data, antenna_pos, output_meta, output):
        return event_data, meta_data, antenna_pos, output_meta, output


class DefaultTransform:
    def __init__(self):
        self.mask = np.logical_and(
            30 <= np.fft.rfftfreq(256, 1e-9) / 1e6,
            np.fft.rfftfreq(256, 1e-9) / 1e6 <= 80,
        )

    def __call__(self, event_data, meta_data, antenna_pos, output_meta, output):

        mm = get_module(event_data)
        # The transpose is so that this transform is generalized for various
        # input shapes

        event_data = mm.log(event_data.T[4:] + 1e-14).T
        #  2: xmax
        meta_data.T[2] /= 700
        #  3: density at xmax - Useless, barely varies in value
        #  4: Height(xmax) - from 1% data, we see div is more sensible than log
        meta_data.T[4] /= 700 * 1e3
        #  5: Eem
        meta_data.T[5] = mm.log(meta_data.T[5] + 1e-14) / 20
        # 10: PRMPAR
        # No nothing
        # TODO: (Think about what to do for the 5000 value outlier)
        # 11: Primary Energy
        meta_data.T[11] = mm.log(meta_data.T[11] + 1e-14)

        meta_data = meta_data.T[1:].T
        output_meta = mm.sign(output_meta) * mm.log(mm.abs(output_meta) + 1e-14)
        # Event shape is flipped. It should be [batch, 300, 7] but it is
        # [batch, 7, 300].
        # TODO: Fix it in the input file and stop swapaxes.
        # antenna_pos = cart2sph(antenna_pos)
        return (
            event_data.T / 30,
            meta_data,
            antenna_pos / 250,
            output_meta,
            pulse2spec(output, self.mask),
        )

    def reverse_transform(self, output_meta, output):
        return output_meta, output
