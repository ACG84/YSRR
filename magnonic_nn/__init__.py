"""Magnonic physical neural network training simulator, built on magnum.np.

A spin-wave scatterer is simulated with full micromagnetics and trained end to
end by backpropagating through the Landau-Lifshitz-Gilbert dynamics. The only
learnable parameters are physical: a field pattern, a saturation-magnetisation
pattern, or the up/down states of an array of nanomagnets. Signal routing and
non-linear activation are both performed by spin-wave propagation and
interference -- there are no artificial neurons anywhere.

Reproduces the setup of

    Papp, A., Porod, W. & Csaba, G. "Nanoscale neural network using non-linear
    spin-wave interference." Nat Commun 12, 6422 (2021).

using magnum.np's field terms in place of a bespoke solver, following the
inverse-micromagnetics approach of

    Abert, C. et al. "NeuralMag: an open-source nodal finite-difference code for
    inverse micromagnetics." npj Comput Mater 11, 193 (2025).

Quick start::

    import magnonic_nn as mnn

    mnn.set_precision("float32")
    task = mnn.build_focusing(mnn.get_preset("focus"))
    history = mnn.train(task.model, task.signals, task.targets,
                        task.loss_fn, epochs=10, lr=0.01,
                        metric_fns=task.metric_fns)
"""

from ._compat import (
    compile_enabled,
    get_device,
    import_magnumnp,
    set_device,
    set_precision,
)
from .config import (
    PRESETS,
    FieldConfig,
    MaterialConfig,
    MeshConfig,
    SimConfig,
    SolverConfig,
    get_preset,
)
from .damping import absorbing_damping
from .dispersion import (
    frequency_of_k,
    k_of_frequency,
    kittel_fmr,
    measure_dispersion,
    omega_of_k,
    usable_band,
    wavelength,
)
from .geometry import (
    FreeFormFieldGeometry,
    Geometry,
    MsGeometry,
    NanomagnetArrayGeometry,
    build_geometry,
    interior_mask,
)
from .losses import (
    accuracy,
    confusion_matrix,
    contrast,
    focus_loss,
    intensity_cross_entropy,
    margin_loss,
)
from .model import SpinWaveNetwork, normalize_intensities
from .plotting import (
    plot_confusion_matrix,
    plot_design,
    plot_integrated_intensity,
    plot_loss,
    plot_probe_outputs,
    plot_snapshot,
    plot_spectrum,
)
from .probes import DiskProbe, PointProbe, Probe, linear_probe_array
from .signals import (
    FORMANTS,
    chirp,
    gated_tones,
    multitone,
    time_vector,
    tone,
    vowel_dataset,
)
from .solver import LLGRollout, RolloutResult, estimate_max_timestep
from .sources import AntennaSource, LineSource, PointSource, Source
from .tasks import Task, build_demux, build_focusing, build_vowels
from .train import (
    TrainHistory,
    accumulate_gradients,
    evaluate,
    load_checkpoint,
    save_checkpoint,
    train,
)

__version__ = "0.1.0"

__all__ = [
    # setup
    "set_precision", "set_device", "get_device", "import_magnumnp", "compile_enabled",
    # config
    "SimConfig", "MeshConfig", "MaterialConfig", "FieldConfig", "SolverConfig",
    "get_preset", "PRESETS",
    # physics
    "LLGRollout", "RolloutResult", "estimate_max_timestep", "absorbing_damping",
    "kittel_fmr", "usable_band", "wavelength", "omega_of_k", "frequency_of_k",
    "k_of_frequency", "measure_dispersion",
    # design
    "Geometry", "FreeFormFieldGeometry", "MsGeometry", "NanomagnetArrayGeometry",
    "build_geometry", "interior_mask",
    # io
    "Source", "PointSource", "LineSource", "AntennaSource",
    "Probe", "PointProbe", "DiskProbe", "linear_probe_array",
    # model
    "SpinWaveNetwork", "normalize_intensities",
    # signals
    "time_vector", "tone", "multitone", "chirp", "gated_tones",
    "vowel_dataset", "FORMANTS",
    # objectives
    "focus_loss", "intensity_cross_entropy", "margin_loss",
    "contrast", "accuracy", "confusion_matrix",
    # training
    "train", "evaluate", "accumulate_gradients", "TrainHistory",
    "save_checkpoint", "load_checkpoint",
    # tasks
    "Task", "build_focusing", "build_demux", "build_vowels",
    # plotting
    "plot_design", "plot_snapshot", "plot_integrated_intensity", "plot_loss",
    "plot_probe_outputs", "plot_confusion_matrix", "plot_spectrum",
]
