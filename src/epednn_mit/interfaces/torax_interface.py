"""TORAX interface for the EPEDNN-mit pedestal structure model.

Please cite [M. Muraca et al. 2025 Nucl. Fusion 65
096010](https://doi.org/10.1088/1741-4326/adf656) in any works using the
"sparc" machine of this model.

Two trained networks ("machines") are available, selected via the
EPEDNNmitConfig.machine field:
  - "sparc": Valid for the SPARC parameter space, as specified in
    https://github.com/aaronkho/epednn_mit/tree/main/src/epednn_mit/models/sparc.
  - "scoping": Valid for a broader high-field FPP scoping parameter space, as
    specified in
    https://github.com/aaronkho/epednn_mit/tree/main/src/epednn_mit/models/scoping.

Register the model once before building a TORAX config, then select it under
``pedestal.structure_model``::

    torax.pedestal.register_pedestal_structure_model(EPEDNNmitConfig)
    config["pedestal"]["structure_model"] = {
        "model_name": "epednn_mit",
        "n_e_top": 2.0e20,
    }

This file defines the classes required to use EPEDNN-mit as a TORAX pedestal
structure model:
1. ``EPEDNNmitPedestalModelWrapper``: closes over the EPEDNN-mit ensemble,
   parameters, and normalization statistics, hashing and comparing by machine.
2. ``EpedNnStructureState``: pedestal structure state carried across timesteps
   tracking the previous pedestal-top location ``rho_norm_top``.
3. ``RuntimeParams``: dynamic runtime parameters passed to the structure model
   at each timestep.
4. ``EPEDNNmitPedestalModel``: ``torax.pedestal.PedestalStructureModel``
   implementation that initializes state and evaluates the wrapper.
5. ``EPEDNNmitConfig``: Pydantic ``torax.pedestal.StructureConfigBase`` config
   used to register and build the structure model and runtime params from a
   TORAX config.
"""

import dataclasses
import functools
import pathlib
from typing import Annotated, Any, Final, Literal, TypeAlias

import chex
from epednn_mit.models.scoping import jax_model as scoping_jax_model
from epednn_mit.models.sparc import jax_model as sparc_jax_model
import jax
from jax import numpy as jnp
import jaxtyping as jt
import numpy as np
import torax
from torax import constants
from torax import math_utils
from torax import torax_pydantic
from typing_extensions import override
import xarray as xr

# pylint: disable=invalid-name


EPEDNNmitStats: TypeAlias = dict[str, jax.Array]
EPEDNNmitParams: TypeAlias = dict[str, Any]
EPEDNNmitMachine: TypeAlias = Literal["sparc", "scoping"]

# For definitions of these parameters, see the EPEDNN-mit SPARC README:
# https://github.com/aaronkho/epednn_mit/blob/main/src/epednn_mit/models/sparc/README.txt
_SPARC_INPUT_BOUNDS: Final[dict[str, tuple[float, float]]] = {
    "Ip": (1.6, 14.3),
    "Bt": (7.2, 12.2),
    "R": (1.85, 1.85),
    "a": (0.57, 0.57),
    "kappa": (1.53, 2.29),
    "delta": (0.39, 0.59),
    "neped": (2.84, 90.235),
    "betan": (0.8, 1.6),
    "zeff": (1.3, 2.5),
}
_SPARC_DEVICE_MAJOR_RADIUS: Final[float] = 1.85
_SPARC_DEVICE_MINOR_RADIUS: Final[float] = 0.57
# In the Snyder mtanh parameterization, the mtanh knee ("pedestal") lies at
# ψ_ped = 1 - Δ and the pedestal top lies at ψ_top = 1 - 1.5 * Δ, where Δ is
# the mtanh width. Thus the pedestal-top width is w_top = 1 - ψ_top = 1.5 * Δ.
_PEDESTAL_TOP_WIDTH_FACTOR: Final[float] = 1.5
# Small offset added to the on-axis magnetic pressure [Pa] to guard the beta
# calculation against division by zero.
_MAGNETIC_PRESSURE_EPS: Final[float] = 1e-7
# Initial pedestal-top rho_norm guess before any network prediction exists.
_INITIAL_RHO_NORM_TOP_GUESS: Final[float] = 0.9
_RHO_NORM_TOP_KEY: Final[torax.output.OutputKey] = torax.output.OutputKey(
    "rho_norm_top",
    units=torax.output.Units.DIMENSIONLESS,
    grid_type=torax.output.GridType.SCALAR,
)

# For definitions of these parameters, including the qstar, fgped, and
# nsfrac derived quantities, see the EPEDNN-mit scoping README:
# https://github.com/aaronkho/epednn_mit/blob/main/src/epednn_mit/models/scoping/README.txt
_SCOPING_INPUT_BOUNDS: Final[dict[str, tuple[float, float]]] = {
    "a": (0.4, 2.2),
    "aspect": (2.0, 4.2),
    "kappa": (1.3, 2.5),
    "delta": (0.3, 0.7),
    "bt": (3.0, 17.0),  # README notes (2.0, 18.0) is not a clean boundary.
    "qstar": (3.0, 5.0),
    "betan": (0.3, 3.7),
    "zeffped": (1.2, 3.2),
    "fgped": (0.3, 1.3),
    "nsfrac": (0.2, 0.8),
    "tesep": (50.0, 500.0),
}


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class EpedNnStructureState(torax.pedestal.StructureState):
  """Structure state for EPED-NN, tracking previous pedestal top location."""

  rho_norm_top: jt.Float[jt.Scalar, ""]

  @override
  def to_xr_datatree(
      self,
      context: torax.output.OutputGridContext,
  ) -> xr.DataTree:
    """Serializes EpedNnStructureState to an xarray DataTree."""
    return xr.DataTree(
        dataset=context.build_dataset({
            _RHO_NORM_TOP_KEY: context.pack(
                _RHO_NORM_TOP_KEY,
                np.atleast_1d(np.asarray(self.rho_norm_top)),
            )
        })
    )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams(torax.pedestal.StructureRuntimeParams):
  """Runtime params for the EPEDNNmitPedestalModel."""

  n_e_top: jt.Float[jt.Scalar, ""]  # [m^-3]
  T_i_T_e_ratio: jt.Float[jt.Scalar, ""]
  p_top_multiplier: jt.Float[jt.Scalar, ""]


class EPEDNNmitPedestalModelWrapper:
  """Wrapper for the EPEDNN-mit pedestal model.

  Captures the EPEDNN-mit model, parameters, and statistics as attributes of
  the class. Hashes by value (machine name), not id.
  """

  def __init__(self, machine: EPEDNNmitMachine):
    self.machine = machine

    # Freeze in parameters of the model specific to the machine.
    match self.machine:
      case "sparc":
        model_dir = pathlib.Path(sparc_jax_model.__file__).parent
        model_weights = sorted(model_dir.glob("epednn_mit_sparc_*.pkl"))
        self._stats, self._params = (
            sparc_jax_model.load_ensemble_params_from_pickle(model_weights)
        )
        self.model = sparc_jax_model.EPEDNNmitEnsemble()
        self.input_lower_bounds = jnp.array(
            [lo for lo, _ in _SPARC_INPUT_BOUNDS.values()], dtype=jnp.float32
        )
        self.input_upper_bounds = jnp.array(
            [hi for _, hi in _SPARC_INPUT_BOUNDS.values()], dtype=jnp.float32
        )
        self.R_0 = _SPARC_DEVICE_MAJOR_RADIUS
        self.a_0 = _SPARC_DEVICE_MINOR_RADIUS
      case "scoping":
        model_dir = pathlib.Path(scoping_jax_model.__file__).parent
        model_weights = sorted(model_dir.glob("epednn_mit_scoping_*.pkl"))
        self._stats, self._params = (
            scoping_jax_model.load_ensemble_params_from_pickle(model_weights)
        )
        self.model = scoping_jax_model.EPEDNNmitEnsemble()
        self.input_lower_bounds = jnp.array(
            [lo for lo, _ in _SCOPING_INPUT_BOUNDS.values()], dtype=jnp.float32
        )
        self.input_upper_bounds = jnp.array(
            [hi for _, hi in _SCOPING_INPUT_BOUNDS.values()], dtype=jnp.float32
        )
        self.R_0 = None
        self.a_0 = None
      case _:
        raise ValueError(
            f"Unsupported machine: {machine}. Only 'sparc' and 'scoping' "
            "are supported."
        )

  def __call__(self, inputs: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Evaluates the EPEDNN-mit ensemble on clipped normalized inputs.

    Args:
      inputs: Input vector (9 elements for "sparc", 11 for "scoping") with
        n_e_ped and Z_eff_ped evaluated at ψ_ped = 1 - Δ.

    Returns:
      A tuple ``(p_top_kPa, pedestal_top_width_psi_norm)`` where ``p_top_kPa``
      is the pedestal-top pressure [kPa] at ψ_top = 1 - 1.5Δ and
      ``pedestal_top_width_psi_norm`` is the pedestal-top width
      w_top = 1 - ψ_top = 1.5Δ in normalized poloidal flux.
    """
    p_top_kPa, pedestal_top_width_psi_norm = self.model.apply(
        self._params, jnp.asarray(inputs, dtype=jnp.float32), **self._stats
    )
    return p_top_kPa, pedestal_top_width_psi_norm

  def __hash__(self) -> int:
    return hash(self.machine)

  def __eq__(self, other: object) -> bool:
    return (
        isinstance(other, EPEDNNmitPedestalModelWrapper)
        and self.machine == other.machine
    )


@functools.lru_cache(maxsize=None)
def _get_epednn_mit_wrapper(
    machine: EPEDNNmitMachine,
) -> EPEDNNmitPedestalModelWrapper:
  return EPEDNNmitPedestalModelWrapper(machine)


def _calculate_beta_N(
    core_profiles: torax.CoreProfiles,
    geo: torax.Geometry,
) -> jt.Float[jt.Array, ""]:
  """Calculates normalized toroidal plasma beta beta_N [%]."""
  p_total_volume_avg = (
      math_utils.cell_integration(
          core_profiles.pressure_total.value * geo.vpr, geo
      )
      / geo.volume_face[-1]
  )
  magnetic_pressure_on_axis = geo.B_0**2 / (2.0 * constants.CONSTANTS.mu_0)
  beta_tor = p_total_volume_avg / (
      magnetic_pressure_on_axis + _MAGNETIC_PRESSURE_EPS
  )
  return (
      1e8
      * beta_tor
      * (
          geo.a_minor
          * geo.B_0
          / (
              core_profiles.poloidal_flux.I_p_profile_face[-1]
              + constants.CONSTANTS.eps
          )
      )
  )


@dataclasses.dataclass(frozen=True, eq=False)
class EPEDNNmitPedestalModel(torax.pedestal.PedestalStructureModel):
  """Pedestal structure model using EPEDNN-mit to predict pressure and width."""

  machine: EPEDNNmitMachine = "sparc"
  # The following fields are set by __post_init__.
  model: EPEDNNmitPedestalModelWrapper = dataclasses.field(init=False)

  def __post_init__(self):
    # Need to use __setattr__  to install attributes as this is a frozen
    # dataclass.
    object.__setattr__(
        self, "model", _get_epednn_mit_wrapper(self.machine)
    )
    super().__post_init__()

  def _prepare_inputs(
      self,
      runtime_params: torax.RuntimeParams,
      geo: torax.Geometry,
      core_profiles: torax.CoreProfiles,
      previous_rho_norm_top: jt.Float[jt.Scalar, ""],
  ) -> jt.Float[jt.Array, "n_inputs"]:
    """Prepares the inputs for EPEDNN-mit.

    Z_eff_ped and n_e_ped are evaluated at the mtanh "pedestal" location
    (ψ_ped = 1 - Δ). n_e_top is read from the evolved
    ``core_profiles.kinetic.n_e`` at ``previous_rho_norm_top``, then converted
    to ψ_ped using the mtanh relationship below.

    mtanh pedestal density profile (Snyder et al., PPCF 46, 2004):

    Neglecting the contribution of the core density profile:

      n(ψ) = n_sep + a₀·[tanh(1) - tanh(2(ψ - ψ_mid)/Δ)]

    where ψ_mid = 1 - Δ/2.  Key locations:

      ψ_top = 1 - 1.5Δ  →  tanh arg = -2  →  EPED-NN output / TORAX top
      ψ_ped = 1 - Δ      →  tanh arg = -1  →  EPED-NN input location
      ψ_mid = 1 - Δ/2    →  tanh arg =  0  →  inflection point

    Profile values at these locations::

      n_top = n_sep + a₀·(tanh 1 + tanh 2)
      n_ped = n_sep + a₀·2·tanh 1

    Eliminating a₀:

      n_ped = n_sep + (n_top - n_sep)·C
      C = 2·tanh(1) / (tanh(1) + tanh(2)) ≈ 0.883

    Args:
      runtime_params: Runtime parameters.
      geo: Geometry.
      core_profiles: Core plasma profiles.
      previous_rho_norm_top: Previous timestep's rho_norm at the pedestal top.

    Returns:
      float32 array of clipped EPED-NN inputs (9 elements for "sparc", 11
      for "scoping").
    """
    assert isinstance(runtime_params.pedestal.structure, RuntimeParams)

    beta_N = _calculate_beta_N(core_profiles, geo)

    # -- Convert ρ_top (pedestal top) to ρ_ped (mtanh pedestal location) --
    # EPED-NN expects inputs evaluated at ψ_ped = 1 − Δ, while TORAX tracks
    # ρ_top (= ψ_top = 1 − 1.5Δ). This section maps ρ_top → ψ_top → Δ →
    # ψ_ped → ρ_ped so we can sample profiles at the correct location.

    # Map ρ ↔ ψ on the face grid, consistently with the mtanh pedestal profile.
    psi_norm_face = core_profiles.psi_norm_face
    psi_top = jnp.interp(
        previous_rho_norm_top, geo.rho_face_norm, psi_norm_face
    )

    # Δ = (1 − ψ_top) / 1.5  ;  ψ_ped = 1 − Δ
    delta_psi = (1.0 - psi_top) / _PEDESTAL_TOP_WIDTH_FACTOR
    psi_ped = 1.0 - delta_psi

    # Map ψ_ped back to ρ_ped.
    rho_ped = jnp.interp(psi_ped, psi_norm_face, geo.rho_face_norm)

    # -- n_e from profile at ρ_top, then mtanh conversion to ψ_ped --
    n_e_top = jnp.interp(
        previous_rho_norm_top,
        geo.rho_face_norm,
        core_profiles.kinetic.n_e.face_value(),
    )
    _C = 2.0 * jnp.tanh(1.0) / (jnp.tanh(1.0) + jnp.tanh(2.0))
    n_e_sep = core_profiles.kinetic.n_e.right_face_value[0]
    n_e_ped = n_e_sep + (n_e_top - n_e_sep) * _C

    # -- Z_eff at the pedestal location --
    Z_eff_ped = jnp.interp(
        rho_ped, geo.rho_face_norm, core_profiles.ions.Z_eff_face
    )

    I_p = core_profiles.poloidal_flux.I_p_profile_face[-1]  # [A]
    I_p_MA = I_p * 1e-6  # [MA]

    if self.machine == "scoping":
      # Definitions of qstar, fgped, and nsfrac follow the "Derived
      # quantities" section of the EPEDNN-mit scoping README:
      #   fgped = n_e_ped / n_Greenwald, with
      #           n_Greenwald = (I_p[MA] / (pi * a^2)) * 1e20 [m^-3].
      #   qstar = 5*a^2*B_t*shaping / (R * I_p[MA])  (Uckan & Sauthoff, ITER
      #           Physics Design Guidelines, 1990), where:
      #     epsilon = a / R
      #     shaping = 0.5*(1 + kappa^2*(1 + 2*delta^2 - 1.2*delta^3))
      #               * (1.17 - 0.65*epsilon) / (1 - epsilon^2)^2
      #   nsfrac = n_e,sep / n_e_ped
      kappa = geo.elongation_face[-1]
      delta = geo.delta_face[-1]
      epsilon = geo.a_minor / geo.R_major
      shaping = (
          0.5
          * (1.0 + kappa**2 * (1.0 + 2.0 * delta**2 - 1.2 * delta**3))
          * (1.17 - 0.65 * epsilon)
          / (1.0 - epsilon**2) ** 2
      )
      qstar = (
          5.0
          * geo.a_minor**2
          * geo.B_0
          * shaping
          / (geo.R_major * (I_p_MA + constants.CONSTANTS.eps))
      )
      n_GW = I_p_MA / (jnp.pi * geo.a_minor**2) * 1e20
      fgped = n_e_ped / (n_GW + constants.CONSTANTS.eps)
      T_e_sep_eV = core_profiles.kinetic.T_e.right_face_value[0] * 1e3

      raw_inputs = jnp.array(
          [
              geo.a_minor,  # [m]
              geo.R_major / geo.a_minor,  # aspect ratio []
              kappa,  # []
              delta,  # []
              geo.B_0,  # [T]
              qstar,  # shaped (engineering) edge safety factor []
              beta_N,  # [%]
              Z_eff_ped,  # []
              fgped,  # Greenwald fraction of n_e_ped []
              n_e_sep / (n_e_ped + constants.CONSTANTS.eps),  # nsfrac []
              T_e_sep_eV,  # tesep [eV]
          ],
          # Network was trained with float32
          dtype=jnp.float32,
      )
    else:
      raw_inputs = jnp.array(
          [
              I_p_MA,  # [MA]
              geo.B_0,  # [T]
              self.model.R_0,  # [m]
              self.model.a_0,  # [m]
              geo.elongation_face[-1],  # []
              geo.delta_face[-1],  # []
              n_e_ped * 1e-19,  # [10^19 m^-3]
              beta_N,  # [%]
              Z_eff_ped,  # []
          ],
          # Network was trained with float32
          dtype=jnp.float32,
      )

    clipped_inputs = jnp.clip(
        raw_inputs,
        self.model.input_lower_bounds,
        self.model.input_upper_bounds,
    )

    return clipped_inputs

  @override
  def initial_state(
      self,
      runtime_params: torax.RuntimeParams,
      geo: torax.Geometry,
      core_profiles: torax.CoreProfiles,
  ) -> EpedNnStructureState:
    """Returns the initial structure state for EPED-NN."""
    # _prepare_inputs requires an initial estimate of rho_norm_top to derive
    # the mtanh knee location (ψ_ped = 1 - Δ), where EPED-NN's input features
    # are evaluated. Evaluating self() once here with rho_norm_top = 0.9 gives a
    # model-consistent initial rho_norm_top so that PedestalModel.initialize
    # evaluates pedestal structure with a self-consistent width.
    _, structure_state = self(
        runtime_params,
        geo,
        core_profiles,
        EpedNnStructureState(
            rho_norm_top=jnp.array(
                _INITIAL_RHO_NORM_TOP_GUESS, dtype=geo.rho_face_norm.dtype
            )
        ),
    )
    return structure_state

  @override
  def initial_state_from_datatree(
      self,
      datatree: xr.DataTree | None,
      runtime_params: torax.RuntimeParams,
  ) -> EpedNnStructureState:
    """Reconstructs EpedNnStructureState from a DataTree for restarts."""
    dtype = jnp.asarray(runtime_params.t).dtype
    if (
        datatree is not None
        and _RHO_NORM_TOP_KEY in datatree.dataset.data_vars
    ):
      rho_norm_top = jnp.asarray(
          np.squeeze(datatree.dataset[_RHO_NORM_TOP_KEY].values),
          dtype=dtype,
      )
    else:
      rho_norm_top = jnp.array(_INITIAL_RHO_NORM_TOP_GUESS, dtype=dtype)
    return EpedNnStructureState(rho_norm_top=rho_norm_top)

  @override
  def __call__(
      self,
      runtime_params: torax.RuntimeParams,
      geo: torax.Geometry,
      core_profiles: torax.CoreProfiles,
      structure_state: torax.pedestal.StructureState,
  ) -> tuple[
      torax.pedestal.PedestalStructure,
      EpedNnStructureState,
  ]:
    """Computes pedestal-top values from EPEDNN-mit pressure and top width.

    Naming convention: In the EPED literature, "pedestal" (ψ_ped = 1 - Δ)
    and "pedestal top" (ψ_top = 1 - 1.5Δ) are distinct mtanh profile
    locations, where Δ is the Snyder mtanh width. EPEDNN-mit takes inputs
    (``n_e_ped``, ``Z_eff_ped``) evaluated at ψ_ped = 1 - Δ (computed in
    ``_prepare_inputs`` from ``previous_rho_norm_top`` via
    Δ = (1 - ψ_top) / 1.5) and outputs ``p_top_kPa`` (at ψ_top) together with
    the pedestal-top width ``pedestal_top_width_psi_norm`` = 1 - ψ_top = 1.5Δ.
    Hence the pedestal-top location is directly
    ``psi_norm_top = 1 - pedestal_top_width_psi_norm``, matching TORAX's
    ``*_top`` convention (``n_e_top``, ``T_e_top``, ``T_i_top``,
    ``rho_norm_top`` in ``PedestalStructure``).

    Args:
      runtime_params: Runtime parameters including pedestal config.
      geo: Geometry.
      core_profiles: Core plasma profiles.
      structure_state: Pedestal structure state carried over from the previous
        timestep.

    Returns:
      Pedestal structure properties and the updated EPED-NN structure state.
    """
    assert isinstance(runtime_params.pedestal.structure, RuntimeParams)
    assert isinstance(structure_state, EpedNnStructureState)

    # Get pedestal-top pressure and pedestal-top width from EPEDNN-mit.
    inputs = self._prepare_inputs(
        runtime_params,
        geo,
        core_profiles,
        previous_rho_norm_top=structure_state.rho_norm_top,
    )
    # p_top_kPa is pressure at the pedestal top (ψ_top = 1 - 1.5Δ).
    # pedestal_top_width_psi_norm is the pedestal-top width 1 - ψ_top = 1.5Δ
    # (NOT the mtanh width Δ; the network was trained on pedestal-top width).
    p_top_kPa, pedestal_top_width_psi_norm = self.model(inputs)
    dtype = geo.rho_face_norm.dtype
    p_top_kPa = p_top_kPa.astype(dtype)
    pedestal_top_width_psi_norm = pedestal_top_width_psi_norm.astype(dtype)

    # Pedestal-top poloidal flux location ψ_top = 1 - w_top, mapped to ρ_N on
    # the face grid.
    psi_norm_top = 1.0 - pedestal_top_width_psi_norm
    rho_norm_top = jnp.interp(
        psi_norm_top, core_profiles.psi_norm_face, geo.rho_face_norm
    )

    # Convert pressure from kPa to Pa.
    p_top = p_top_kPa * 1e3

    n_e_top = runtime_params.pedestal.structure.n_e_top

    # -- Calculate T_e and T_i at the pedestal top from p_top --
    # Evaluate composition at the pedestal-top location.
    temperature_ratio = runtime_params.pedestal.structure.T_i_T_e_ratio
    at_top = lambda face_values: jnp.interp(
        rho_norm_top, geo.rho_face_norm, face_values
    )
    Z_eff_top = at_top(core_profiles.ions.Z_eff_face)
    Z_i_top = at_top(core_profiles.ions.Z_i_face)
    Z_impurity_top = at_top(core_profiles.ions.Z_impurity_average_face)
    Z_impurity_bundled_top = at_top(
        core_profiles.ions.Z_impurity_bundled_face
    )
    # Guard dead branches of jnp.where against 0/0 so autodiff produces finite
    # gradients in pure plasmas (Z_eff == 1.0).
    safe_dilution_denom = jnp.where(
        Z_eff_top == 1.0,
        1.0,
        Z_i_top * (Z_impurity_bundled_top - Z_i_top),
    )
    dilution_factor_top = jnp.where(
        Z_eff_top == 1.0,
        1.0,
        (Z_impurity_bundled_top - Z_eff_top) / safe_dilution_denom,
    )
    n_i_top = dilution_factor_top * n_e_top
    safe_Z_impurity_top = jnp.where(
        Z_eff_top == 1.0, 1.0, Z_impurity_top
    )
    n_impurity_top = jnp.where(
        Z_eff_top == 1.0,
        0.0,
        (n_e_top - Z_i_top * n_i_top) / safe_Z_impurity_top,
    )
    # p = T_e*n_e + T_i*n_i + T_i*n_imp  (assuming T_imp = T_i)
    T_e_top = (
        p_top * runtime_params.pedestal.structure.p_top_multiplier
    ) / (
        n_e_top
        + temperature_ratio * n_i_top
        + temperature_ratio * n_impurity_top
    ) / constants.CONSTANTS.keV_to_J
    T_i_top = temperature_ratio * T_e_top

    return (
        torax.pedestal.PedestalStructure(
            n_e_top=n_e_top,
            T_i_top=T_i_top,
            T_e_top=T_e_top,
            rho_norm_top=rho_norm_top,
        ),
        EpedNnStructureState(rho_norm_top=rho_norm_top),
    )


class EPEDNNmitConfig(torax.pedestal.StructureConfigBase):
  """TORAX pedestal structure model config using EPEDNN-mit.

  Attributes:
    machine: Which trained EPEDNN-mit network to use, "sparc" or "scoping".
    n_e_top: The electron density at the pedestal top [m^-3].
    T_i_T_e_ratio: Ratio of the ion and electron temperature at the pedestal
      top [dimensionless].
    p_top_multiplier: Multiplier for the pedestal-top pressure (mostly used for
      sensitivity analysis) [dimensionless].
  """

  model_name: Annotated[Literal["epednn_mit"], torax_pydantic.JAX_STATIC] = (
      "epednn_mit"
  )
  machine: Annotated[EPEDNNmitMachine, torax_pydantic.JAX_STATIC] = "sparc"
  n_e_top: torax_pydantic.PositiveTimeVaryingScalar
  T_i_T_e_ratio: torax_pydantic.PositiveTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(1.0)
  )
  p_top_multiplier: torax_pydantic.PositiveTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(1.0)
  )

  def build_structure_model(
      self,
  ) -> EPEDNNmitPedestalModel:
    return EPEDNNmitPedestalModel(machine=self.machine)

  def build_runtime_params(self, t: chex.Numeric) -> RuntimeParams:
    base_runtime_params = super().build_runtime_params(t)
    return RuntimeParams(
        **vars(base_runtime_params),
        p_top_multiplier=self.p_top_multiplier.get_value(t),
        n_e_top=self.n_e_top.get_value(t),
        T_i_T_e_ratio=self.T_i_T_e_ratio.get_value(t),
    )
