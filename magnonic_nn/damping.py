"""Spatially varying Gilbert damping with absorbing boundaries.

A finite simulation window reflects spin waves off its edges, and those
reflections dominate the probe signals long before the useful transmitted wave
arrives. The standard fix -- and what the reference implementation does -- is to
ramp the damping up towards the edges so outgoing waves are swallowed, which
makes the finite window behave like a window cut out of an extended film.
"""

from __future__ import annotations

import torch

from .config import MaterialConfig, MeshConfig

__all__ = ["absorbing_damping"]


def absorbing_damping(mesh: MeshConfig, material: MaterialConfig) -> torch.Tensor:
    """Build the damping field ``alpha(x, y)``.

    The value at a cell depends only on its distance ``d`` (in cells) to the
    nearest lateral edge::

        alpha(d) = alpha                                     for d >= W
        alpha(d) = alpha + (alpha_max - alpha) * ((W - d)/W)^p   for d < W

    so damping is ``alpha_max`` on the outermost ring and tapers smoothly to the
    bulk value ``alpha`` at depth ``W``. A gradual taper matters: a step change
    in damping is itself an impedance discontinuity and reflects.

    .. note::
       The reference implementation writes the taper onto rings ``1..W`` and
       leaves the outermost ring (``d = 0``) at the bulk value, so its profile
       is non-monotonic in the last cell. The formula above is monotonic. The
       difference is one cell deep inside an already-absorbing layer and does
       not measurably change the transmitted field.

    :returns: tensor of shape ``(nx, ny, nz, 1)``, suitable for direct use as
        the ``alpha`` material parameter.
    """
    nx, ny, nz = mesh.n
    width = int(material.abc_width)

    alpha = torch.full((nx, ny, nz, 1), float(material.alpha))
    if width <= 0:
        return alpha

    pbc = getattr(mesh, "pbc", (0, 0, 0))
    ix = torch.arange(nx).reshape(nx, 1)
    iy = torch.arange(ny).reshape(1, ny)

    # Distance to the nearest lateral edge, in cells. A periodic axis has no
    # edge to absorb at -- damping it would attenuate a wave that is physically
    # meant to wrap around.
    dists = []
    if not pbc[0]:
        dists.append(torch.minimum(ix, (nx - 1) - ix).expand(nx, ny))
    if not pbc[1]:
        dists.append(torch.minimum(iy, (ny - 1) - iy).expand(nx, ny))
    if not dists:
        return alpha
    dist = torch.stack(dists).amin(dim=0).to(alpha.dtype)

    ramp = ((width - dist) / width).clamp(min=0.0) ** material.abc_exponent
    profile = material.alpha + (material.alpha_max - material.alpha) * ramp

    return profile.reshape(nx, ny, 1, 1).expand(nx, ny, nz, 1).contiguous()
