"""Trainable scatterer geometries.

These are the "weights" of the physical network. Each geometry exposes a
``torch.nn.Parameter`` ``rho`` and turns it into the two quantities the solver
needs:

``static_field()``
    The time-independent external field, in A/m, shape ``(nx, ny, nz, 3)``.
    Always contains the uniform bias; a field-programmed scatterer adds its
    pattern on top.

``Ms_field()``
    The saturation magnetisation, in A/m, shape ``(nx, ny, nz, 1)``.

Three parameterisations are provided, mirroring the reference implementation:

:class:`FreeFormFieldGeometry`
    A per-cell field offset. Most degrees of freedom, easiest to train, and the
    variant the paper falls back on when it wants "~16000 continuous
    variables". Physically it stands in for any means of locally shifting the
    dispersion relation.

:class:`MsGeometry`
    A per-cell saturation-magnetisation multiplier -- i.e. patterning the film
    itself (etching, ion irradiation) rather than the field above it.

:class:`NanomagnetArrayGeometry`
    A lattice of perpendicularly magnetised nanomagnets whose up/down state is
    binary and trainable through a straight-through estimator. This is the
    fabricable device in the paper: the scattering field is the *stray field*
    of the nanomagnet array, computed here with the Newell kernel evaluated at
    the nanomagnets' stand-off height.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from ._compat import import_magnumnp
from .config import MU_0, FieldConfig, MaterialConfig, MeshConfig

_magnumnp = import_magnumnp()
from magnumnp.field_terms.demag import f as _newell_f  # noqa: E402
from magnumnp.field_terms.demag import g as _newell_g  # noqa: E402
from magnumnp.field_terms.demag import newell as _newell  # noqa: E402

__all__ = [
    "Geometry",
    "FreeFormFieldGeometry",
    "MsGeometry",
    "NanomagnetArrayGeometry",
    "build_geometry",
    "interior_mask",
    "binarize",
]


class _Binarize(torch.autograd.Function):
    """``sign(x)`` forward, identity backward (straight-through estimator).

    The forward pass must be binary because the nanomagnets really are either
    up or down. The backward pass pretends the operation was the identity, so
    gradients still tell ``rho`` which way to move. This is the standard trick
    for training binary weights and is what the reference implementation uses.
    """

    @staticmethod
    def forward(ctx, x):
        return torch.sign(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def binarize(x: torch.Tensor) -> torch.Tensor:
    return _Binarize.apply(x)


class Geometry(nn.Module):
    """Base class holding the mesh/field/material configuration."""

    def __init__(self, mesh: MeshConfig, fields: FieldConfig, material: MaterialConfig):
        super().__init__()
        self.mesh_cfg = mesh
        self.field_cfg = fields
        self.material_cfg = material

        axis = torch.tensor(fields.bias_axis, dtype=torch.get_default_dtype())
        axis = axis / axis.norm()
        self.register_buffer("bias_axis", axis)
        # magnum.np works in H (A/m); the config quotes fields in tesla.
        self.register_buffer("H0", torch.tensor(fields.B0 / MU_0))

    @property
    def shape(self):
        return self.mesh_cfg.n

    def _uniform_bias(self) -> torch.Tensor:
        nx, ny, nz = self.shape
        return (self.H0 * self.bias_axis).reshape(1, 1, 1, 3).expand(nx, ny, nz, 3)

    def static_field(self) -> torch.Tensor:
        raise NotImplementedError

    def Ms_field(self) -> torch.Tensor:
        nx, ny, nz = self.shape
        return torch.full((nx, ny, nz, 1), float(self.material_cfg.Ms))

    def design_image(self) -> torch.Tensor:
        """2D array for plotting the current design, shape ``(nx, ny)``."""
        raise NotImplementedError


class FreeFormFieldGeometry(Geometry):
    """Per-cell trainable field offset along the bias axis.

    ``H = (B0 + B1 * bound(rho)) / mu_0 * bias_axis``

    :param bound: how ``rho`` is constrained.
        ``'none'`` leaves it unconstrained (reference behaviour -- nothing stops
        the optimiser from asking for a 200 mT local field, so check
        ``design_image()`` before believing a result);
        ``'tanh'`` maps it smoothly into ``(-1, 1)``;
        ``'clamp'`` hard-clips to ``[-1, 1]`` with a straight-through gradient
        outside the range.
    :param design_mask: optional ``(nx, ny)`` boolean mask restricting which
        cells are trainable. Cells outside the mask keep zero offset. Use it to
        keep the absorbing boundary and the source/probe lines out of the
        design region.
    """

    def __init__(
        self,
        mesh: MeshConfig,
        fields: FieldConfig,
        material: MaterialConfig,
        bound: str = "tanh",
        design_mask: torch.Tensor | None = None,
        init: float = 0.0,
    ):
        super().__init__(mesh, fields, material)
        nx, ny, _ = self.shape
        if bound not in ("none", "tanh", "clamp"):
            raise ValueError(f"unknown bound mode {bound!r}")
        self.bound = bound
        self.rho = nn.Parameter(torch.full((nx, ny), float(init)))
        self.register_buffer("H1", torch.tensor(fields.B1 / MU_0))
        if design_mask is None:
            mask = torch.ones(nx, ny)
        else:
            mask = design_mask.to(dtype=self.rho.dtype).reshape(nx, ny)
        self.register_buffer("design_mask", mask)

    def _bounded_rho(self) -> torch.Tensor:
        if self.bound == "tanh":
            rho = torch.tanh(self.rho)
        elif self.bound == "clamp":
            # straight-through clamp: forward clipped, gradient passed through
            rho = self.rho + (self.rho.clamp(-1.0, 1.0) - self.rho).detach()
        else:
            rho = self.rho
        return rho * self.design_mask

    def static_field(self) -> torch.Tensor:
        nx, ny, nz = self.shape
        offset = (self.H1 * self._bounded_rho()).reshape(nx, ny, 1, 1)
        return self._uniform_bias() + offset * self.bias_axis.reshape(1, 1, 1, 3)

    def design_image(self) -> torch.Tensor:
        return (self.field_cfg.B1 * self._bounded_rho()).detach()


class MsGeometry(Geometry):
    """Per-cell trainable saturation magnetisation, ``Ms -> Ms * bound(rho)``.

    Patterning ``Ms`` changes both the exchange and the dipolar response, so it
    scatters more strongly per unit of design change than a field offset -- but
    it is also the harder thing to fabricate and to reverse.

    :param bound: ``'none'`` (reference behaviour), or ``'clamp'`` to keep the
        multiplier inside ``[ms_min, ms_max]`` with a straight-through gradient.
    """

    def __init__(
        self,
        mesh: MeshConfig,
        fields: FieldConfig,
        material: MaterialConfig,
        bound: str = "clamp",
        ms_min: float = 0.0,
        ms_max: float = 1.0,
        design_mask: torch.Tensor | None = None,
        init: float = 1.0,
    ):
        super().__init__(mesh, fields, material)
        nx, ny, _ = self.shape
        if bound not in ("none", "clamp"):
            raise ValueError(f"unknown bound mode {bound!r}")
        self.bound = bound
        self.ms_min = float(ms_min)
        self.ms_max = float(ms_max)
        self.rho = nn.Parameter(torch.full((nx, ny), float(init)))
        if design_mask is None:
            mask = torch.ones(nx, ny)
        else:
            mask = design_mask.to(dtype=self.rho.dtype).reshape(nx, ny)
        self.register_buffer("design_mask", mask)

    def _bounded_rho(self) -> torch.Tensor:
        rho = self.rho
        if self.bound == "clamp":
            rho = rho + (rho.clamp(self.ms_min, self.ms_max) - rho).detach()
        # outside the design region the film keeps its nominal Ms
        return rho * self.design_mask + (1.0 - self.design_mask)

    def static_field(self) -> torch.Tensor:
        return self._uniform_bias()

    def Ms_field(self) -> torch.Tensor:
        nx, ny, nz = self.shape
        return (self.material_cfg.Ms * self._bounded_rho()).reshape(nx, ny, 1, 1).expand(
            nx, ny, nz, 1
        )

    def design_image(self) -> torch.Tensor:
        return (self.material_cfg.Ms * self._bounded_rho()).detach()


def _stray_field_kernel(
    n: tuple[int, int],
    d: tuple[float, float, float],
    z_offset: float,
):
    """FFT of the demag kernel columns coupling ``M_z`` of an offset layer to
    ``H`` in the film plane.

    The nanomagnets sit ``z_offset`` metres above the film and are magnetised
    perpendicular to it, so only the third column of the demagnetisation tensor
    contributes: ``H_a = N_az * M_z``.

    Uses Newell's exact formulae (via magnum.np's ``f``/``g``) everywhere rather
    than magnum.np's near/far split, because the far-field dipole branch of
    ``demag_f``/``demag_g`` zeroes the ``(0,0,0)`` entry -- correct for the
    self-interaction of a single layer, wrong for two layers separated in z
    where that entry is a genuine finite coupling.

    :returns: ``(Nxz_fft, Nyz_fft, Nzz_fft)``, each of shape ``(2nx, nyf)``
        where ``nyf = ny + 1`` is the ``rfft`` length of a ``2ny`` axis.
    """
    nx, ny = n
    dx, dy, dz = d

    # Rescale lengths so single-precision arithmetic in the Newell formulae
    # stays well-conditioned; the kernel is scale-invariant.
    scale = min(dx, dy, dz)
    sx, sy, sz = dx / scale, dy / scale, dz / scale
    sz_off = z_offset / scale

    shape = (2 * nx, 2 * ny)
    ij = [torch.fft.fftfreq(s, 1 / s) for s in shape]
    ii, jj = torch.meshgrid(*ij, indexing="ij")

    dtype = torch.get_default_dtype()
    x = (ii * sx).to(torch.float64)
    y = (jj * sy).to(torch.float64)
    z = torch.full_like(x, float(sz_off))

    # magnum.np's convention: Nxz = g(x, z, y), Nyz = g(y, z, x), Nzz = f(z, x, y)
    Nxz = _newell(_newell_g, x, z, y, sx, sz, sy, sx, sz, sy)
    Nyz = _newell(_newell_g, y, z, x, sy, sz, sx, sy, sz, sx)
    Nzz = _newell(_newell_f, z, x, y, sz, sx, sy, sz, sx, sy)

    out = []
    for N in (Nxz, Nyz, Nzz):
        out.append(torch.fft.rfftn(N.to(dtype), dim=(0, 1)))
    return tuple(out)


class NanomagnetArrayGeometry(Geometry):
    """Binary array of perpendicularly magnetised nanomagnets above the film.

    Each lattice site carries one nanomagnet whose magnetisation points either
    ``+z`` or ``-z``; the trainable ``rho`` is passed through a straight-through
    ``sign`` so the forward pass is genuinely binary. The scattering field seen
    by the film is the array's stray field, evaluated at the film plane.

    :param lattice_origin: index of the first nanomagnet in x and y.
    :param lattice_pitch: spacing between nanomagnets, in cells.
    :param magnet_size: nanomagnet footprint, in cells (square).
    :param z_offset_cells: stand-off between film and nanomagnet layer, in units
        of ``dz``.
    :param Ms_magnet: saturation magnetisation of the nanomagnet material
        (default: CoPt, 723 kA/m).
    """

    def __init__(
        self,
        mesh: MeshConfig,
        fields: FieldConfig,
        material: MaterialConfig,
        lattice_origin: int = 15,
        lattice_pitch: int = 4,
        magnet_size: int = 2,
        z_offset_cells: int = 10,
        Ms_magnet: float = 723e3,
        nmag_x: int | None = None,
        nmag_y: int | None = None,
        init: torch.Tensor | None = None,
    ):
        super().__init__(mesh, fields, material)
        nx, ny, _ = self.shape
        self.origin = int(lattice_origin)
        self.pitch = int(lattice_pitch)
        self.magnet_size = int(magnet_size)
        self.Ms_magnet = float(Ms_magnet)

        if nmag_x is None:
            nmag_x = max(int((nx - 2 * self.origin) / self.pitch), 1)
        if nmag_y is None:
            nmag_y = max(int((ny - 2 * self.origin) / self.pitch) + 1, 1)
        self.nmag = (int(nmag_x), int(nmag_y))

        if init is None:
            # Break the symmetry: an all-zero rho has sign() == 0 everywhere,
            # which produces no stray field and no useful gradient signal.
            init = torch.empty(self.nmag).uniform_(-1.0, 1.0)
        self.rho = nn.Parameter(init.clone().detach().reshape(self.nmag))

        Nxz, Nyz, Nzz = _stray_field_kernel(
            (nx, ny), self.mesh_cfg.d, z_offset_cells * self.mesh_cfg.dz
        )
        self.register_buffer("Nxz_fft", Nxz)
        self.register_buffer("Nyz_fft", Nyz)
        self.register_buffer("Nzz_fft", Nzz)

    def _magnet_pattern(self) -> torch.Tensor:
        """Expand the lattice of binary states onto the simulation grid."""
        nx, ny, _ = self.shape
        states = binarize(self.rho)

        pattern = torch.zeros(nx, ny, dtype=states.dtype, device=states.device)
        end_x = self.origin + self.nmag[0] * self.pitch
        end_y = self.origin + self.nmag[1] * self.pitch
        pattern[self.origin : end_x : self.pitch, self.origin : end_y : self.pitch] = states

        if self.magnet_size > 1:
            # Give each nanomagnet a finite footprint by convolving the lattice
            # of delta functions with a square of ones.
            kernel = torch.ones(1, 1, self.magnet_size, self.magnet_size, dtype=pattern.dtype)
            pad = self.magnet_size // 2
            conv = torch.nn.functional.conv2d(
                pattern.reshape(1, 1, nx, ny), kernel, padding=pad
            )
            pattern = conv[0, 0, :nx, :ny]
        return pattern

    def _stray_field(self) -> torch.Tensor:
        """Stray field of the nanomagnet array at the film plane, in A/m."""
        nx, ny, nz = self.shape
        Mz = self._magnet_pattern() * self.Ms_magnet

        Mz_fft = torch.fft.rfftn(Mz, dim=(0, 1), s=(2 * nx, 2 * ny))
        hx = torch.fft.irfftn(self.Nxz_fft * Mz_fft, dim=(0, 1))[:nx, :ny]
        hy = torch.fft.irfftn(self.Nyz_fft * Mz_fft, dim=(0, 1))[:nx, :ny]
        hz = torch.fft.irfftn(self.Nzz_fft * Mz_fft, dim=(0, 1))[:nx, :ny]

        return torch.stack([hx, hy, hz], dim=-1).reshape(nx, ny, 1, 3).expand(nx, ny, nz, 3)

    def static_field(self) -> torch.Tensor:
        return self._uniform_bias() + self._stray_field()

    def design_image(self) -> torch.Tensor:
        return self._magnet_pattern().detach()

    def stray_field_image(self) -> torch.Tensor:
        """Component of the stray field along the bias axis, in tesla."""
        h = self._stray_field()[:, :, 0, :].detach()
        return MU_0 * (h * self.bias_axis).sum(dim=-1)


def build_geometry(cfg, design_mask: torch.Tensor | None = None, **kwargs) -> Geometry:
    """Instantiate the geometry named by ``cfg.geometry``."""
    kinds = {
        "freeform": FreeFormFieldGeometry,
        "ms": MsGeometry,
        "nanomagnets": NanomagnetArrayGeometry,
    }
    if cfg.geometry not in kinds:
        raise KeyError(f"unknown geometry {cfg.geometry!r}; available: {sorted(kinds)}")
    cls = kinds[cfg.geometry]
    if cls is NanomagnetArrayGeometry:
        # the nanomagnet lattice has no per-cell mask
        kwargs.pop("design_mask", None)
        return cls(cfg.mesh, cfg.fields, cfg.material, **kwargs)
    return cls(cfg.mesh, cfg.fields, cfg.material, design_mask=design_mask, **kwargs)


def interior_mask(mesh: MeshConfig, material: MaterialConfig, margin: int = 0) -> torch.Tensor:
    """Boolean ``(nx, ny)`` mask that excludes the absorbing boundary.

    Designing inside the absorbing layer is wasted capacity -- whatever the
    optimiser puts there is damped away -- and it lets the optimiser cheat by
    modulating the absorber instead of steering the wave.
    """
    nx, ny, _ = mesh.n
    pad = int(material.abc_width) + int(margin)
    mask = torch.zeros(nx, ny)
    if 2 * pad >= min(nx, ny):
        raise ValueError(
            f"absorbing boundary ({pad} cells) leaves no design region in a "
            f"{nx}x{ny} mesh"
        )
    mask[pad : nx - pad, pad : ny - pad] = 1.0
    return mask
