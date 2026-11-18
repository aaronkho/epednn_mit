"""TORAX interface for the EPEDNN-mit pedestal model.

Please cite [M. Muraca et al. 2025 Nucl. Fusion 65
096010](https://doi.org/10.1088/1741-4326/adf656) in any works using this model.
Currently, this model is only valid for the SPARC parameter space, as specified
in https://github.com/aaronkho/epednn_mit/tree/main/src/epednn_mit/models/sparc.

EPEDNN-mit plugs into TORAX as a pedestal structure model. Register it once,
before building a TORAX config, then select it under ``structure_model``::

    torax.pedestal.register_pedestal_structure_model(EPEDNNmitConfig)
    config['pedestal']['structure_model'] = {'model_name': 'epednn_mit'}

This file defines the five necessary classes to use the model in TORAX:
1. A wrapper class, which closes over the EPEDNN-mit model, parameters, and
   statistics. This is necessary to make the model hashable.
2. A TORAX pedestal structure model, which calls the wrapper within its
   __call__. This is what will be called by TORAX.
3. A TORAX pedestal structure model Pydantic config, which is used to build the
   structure model and runtime params. This allows us to use the model from a
   TORAX config.
4. A runtime params class, which contains any input parameters specific to the
   EPEDNN-mit model. This will be passed to the pedestal model by TORAX at each
   time step.
5. A structure state, which carries the pedestal-top location from one call to
   the next. The network inputs are sampled relative to that location.
"""

import dataclasses
import pathlib
from typing import Annotated, Any, Final, Literal, TypeAlias

import chex
from epednn_mit.models.sparc import jax_model as epednn_mit_jax_model
import jax
from jax import numpy as jnp
import jaxtyping as jt
import torax
from torax._src import jax_utils
from torax._src.pedestal_model.structure import set_ptop_ttopratio_ntop
from torax._src.physics import formulas
from torax._src.torax_pydantic import torax_pydantic
import xarray as xr

# pylint: disable=invalid-name


EPEDNNmitStats: TypeAlias = dict[str, jax.Array]
EPEDNNmitParams: TypeAlias = dict[str, Any]
EPEDNNmitMachine: TypeAlias = Literal["sparc"]

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

# Pedestal-top guess for the first call, before any network prediction exists.
_RHO_NORM_TOP_GUESS: Final[float] = 0.9

# Splits the predicted pedestal-top pressure into T_e, T_i and n_e.
_SET_P_TOP_MODEL: Final = (
    set_ptop_ttopratio_ntop.SetPressureTemperatureRatioAndDensityPedestalModel()
)


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
        model_dir = pathlib.Path(epednn_mit_jax_model.__file__).parent
        model_weights = sorted(model_dir.glob("epednn_mit_sparc_*.pkl"))
        self._stats, self._params = (
            epednn_mit_jax_model.load_ensemble_params_from_pickle(model_weights)
        )
        self.model = epednn_mit_jax_model.EPEDNNmitEnsemble()
        bounds = _SPARC_INPUT_BOUNDS
        self.R_0 = _SPARC_DEVICE_MAJOR_RADIUS
        self.a_0 = _SPARC_DEVICE_MINOR_RADIUS
      case _:
        raise ValueError(
            f"Unsupported machine: {machine}. Only SPARC is supported."
        )

    # Evaluate the network in float32, the precision it was trained in, even
    # though TORAX enables float64. Inputs held fixed in training (R, a) have
    # variances ~1e-31, so a float32 input against a float64 mean, off by
    # ~1e-8, normalizes to ~1e7 and the output blows up. The bounds are
    # float32 too, so clipping does not promote the inputs.
    self._stats, self._params = jax.tree.map(
        lambda x: jnp.asarray(x, dtype=jnp.float32),
        (self._stats, self._params),
    )
    self.input_lower_bounds = jnp.array(
        [lo for lo, _ in bounds.values()], dtype=jnp.float32
    )
    self.input_upper_bounds = jnp.array(
        [hi for _, hi in bounds.values()], dtype=jnp.float32
    )

  def __call__(self, inputs: jax.Array) -> tuple[jax.Array, jax.Array]:
    P_top_kPa, pedestal_width_psi_norm = self.model.apply(
        self._params, inputs, **self._stats
    )
    return P_top_kPa, pedestal_width_psi_norm

  def __hash__(self):
    return hash(self.machine)

  def __eq__(self, other: Any) -> bool:
    return (
        isinstance(other, EPEDNNmitPedestalModelWrapper)
        and self.machine == other.machine
    )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class EPEDNNmitStructureState(torax.pedestal.StructureState):
  """Pedestal-top location from the previous EPEDNN-mit call.

  Attributes:
    rho_norm_top: Pedestal-top location [rho_norm]. ``inf`` until the first
      prediction, so the inputs are sampled about ``_RHO_NORM_TOP_GUESS``.
  """

  rho_norm_top: jt.Float[jt.Scalar, ""] = dataclasses.field(
      default_factory=lambda: jnp.array(jnp.inf, dtype=jax_utils.get_dtype())
  )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams(torax.pedestal.StructureRuntimeParams):
  """Runtime params for the EPEDNNmitPedestalModel."""

  n_e_top: jt.Float[jt.Scalar, ""]  # [m^-3] or Greenwald fraction
  n_e_top_is_f_GW: jt.Bool[jt.Scalar, ""]
  T_i_T_e_ratio: jt.Float[jt.Scalar, ""]
  p_top_multiplier: jt.Float[jt.Scalar, ""]


@dataclasses.dataclass(frozen=True, eq=False)
class EPEDNNmitPedestalModel(torax.pedestal.PedestalStructureModel):
  """TORAX pedestal structure model using EPEDNN-mit to predict pressure and
  width.
  """

  machine: EPEDNNmitMachine = "sparc"
  # The following fields are set by __post_init__.
  model: EPEDNNmitPedestalModelWrapper = dataclasses.field(init=False)

  def __post_init__(self):
    # Need to use __setattr__  to install attributes as this is a frozen
    # dataclass.
    object.__setattr__(
        self, "model", EPEDNNmitPedestalModelWrapper(self.machine)
    )
    super().__post_init__()

  def initial_state(
      self,
      runtime_params: torax.RuntimeParams,
      geo: torax.Geometry,
      core_profiles: torax.CoreProfiles,
  ) -> EPEDNNmitStructureState:
    del runtime_params, geo, core_profiles
    return EPEDNNmitStructureState()

  def initial_state_from_datatree(
      self,
      datatree: xr.DataTree | None,
      runtime_params: torax.RuntimeParams,
  ) -> EPEDNNmitStructureState:
    del datatree, runtime_params
    return EPEDNNmitStructureState()

  def _prepare_inputs(
      self,
      geo: torax.Geometry,
      core_profiles: torax.CoreProfiles,
      previous_rho_norm_top: jax.Array,
  ) -> jt.Float[jt.Array, "9"]:
    """Prepares the inputs for EPEDNN-mit.

    When ``previous_rho_norm_top`` is provided (and is not the
    placeholder ``inf``), Z_eff and n_e_ped are evaluated at the mtanh
    "pedestal" location (ψ_ped = 1 - Δ). n_e_top is read from the
    evolved ``core_profiles.kinetic.n_e`` at rho_norm_top, then corrected to
    ψ_ped using the assumed mtanh relationship below.

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
      geo: Geometry.
      core_profiles: Core plasma profiles.
      previous_rho_norm_top: Previous timestep's rho_norm at the pedestal
        top. ``jnp.inf`` signals first timestep (placeholder).

    Returns:
      9-element float32 array of clipped EPED-NN inputs.
    """
    _, _, beta_N = formulas.calculate_betas(core_profiles, geo)
    n_e = core_profiles.kinetic.n_e

    # -- Convert ρ_top (pedestal top) to ρ_ped (mtanh pedestal location) --
    # EPED-NN expects inputs evaluated at ψ_ped = 1 − Δ, but TORAX tracks
    # ρ_top (= ψ_top = 1 − 1.5Δ). This section maps ρ_top → ψ_top → Δ →
    # ψ_ped → ρ_ped so we can sample profiles at the correct location.

    # Default ρ_top to 0.9 when no previous pedestal output is available.
    rho_top = jnp.where(
        jnp.isinf(previous_rho_norm_top),
        _RHO_NORM_TOP_GUESS,
        previous_rho_norm_top,
    )

    # Map ρ → ψ using the normalised poloidal flux profile.
    rho_norm = _rho_norm_cells_and_boundaries(geo)
    psi_norm = _psi_norm(core_profiles)
    psi_top = jnp.interp(rho_top, rho_norm, psi_norm)

    # Δ = (1 − ψ_top) / 1.5  ;  ψ_ped = 1 − Δ
    delta_psi = (1.0 - psi_top) / 1.5
    psi_ped = 1.0 - delta_psi

    # Map ψ_ped back to ρ_ped.
    rho_ped = jnp.interp(psi_ped, psi_norm, rho_norm)

    # -- n_e from profile at ρ_top, then mtanh correction to ψ_ped --
    n_e_profile = n_e.cell_plus_boundaries()
    n_e_top = jnp.interp(rho_top, rho_norm, n_e_profile)
    C = 2.0 * jnp.tanh(1.0) / (jnp.tanh(1.0) + jnp.tanh(2.0))
    n_e_sep = n_e_profile[-1]
    n_e_ped = n_e_sep + (n_e_top - n_e_sep) * C

    # -- Z_eff at the pedestal location --
    ions = core_profiles.ions
    Z_eff = jnp.concatenate(
        [ions.Z_eff_face[:1], ions.Z_eff, ions.Z_eff_face[-1:]]
    )
    Z_eff_ped = jnp.interp(rho_ped, rho_norm, Z_eff)

    raw_inputs = jnp.array(
        [
            core_profiles.poloidal_flux.I_p_profile_face[-1] * 1e-6,  # [MA]
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

    return jnp.clip(
        raw_inputs,
        self.model.input_lower_bounds,
        self.model.input_upper_bounds,
    )

  def __call__(
      self,
      runtime_params: torax.RuntimeParams,
      geo: torax.Geometry,
      core_profiles: torax.CoreProfiles,
      structure_state: EPEDNNmitStructureState,
  ) -> tuple[torax.pedestal.PedestalStructure, EPEDNNmitStructureState]:
    """Computes pedestal-top values from EPEDNN-mit pressure and width.

    Naming convention: In the EPED literature, "pedestal" (ψ_ped = 1 - Δ)
    and "pedestal top" (ψ_top = 1 - 1.5Δ) are distinct mtanh profile
    locations.  EPEDNN-mit outputs P_top and the width to the pedestal
    top, and TORAX I/O also uses "top" to mean pedestal top
    (e.g. ``n_e_top``, ``T_e_top``, ``rho_norm_top`` in
    ``PedestalStructure``).

    Within this function, local variables use ``_top`` suffixes when
    evaluated at the pedestal-top location (rho_norm_top) to distinguish from
    EPEDNN-mit inputs evaluated at ψ_ped.

    The pressure is split into T_e, T_i and n_e at the pedestal top by
    TORAX's ``set_p_top_n_top`` model, with the prescribed n_e_top and
    T_i/T_e ratio.

    Args:
      runtime_params: Runtime parameters including pedestal config.
      geo: Geometry.
      core_profiles: Core plasma profiles.
      structure_state: Pedestal-top location from the previous call.

    Returns:
      The pedestal structure, and a state carrying its ``rho_norm_top``.
    """
    params = runtime_params.pedestal.structure
    assert isinstance(params, RuntimeParams)

    # Get pedestal pressure and width from EPEDNN-mit.
    inputs = self._prepare_inputs(
        geo, core_profiles, structure_state.rho_norm_top
    )
    # P_top_kPa is pressure at the pedestal top.
    # pedestal_width_psi_norm is Δψ_N from ψ=1 to the pedestal top (ψ_top).
    P_top_kPa, pedestal_width_psi_norm = self.model(inputs)

    # Convert pedestal-top width from ψ_N to ρ_N.
    dtype = jax_utils.get_dtype()
    rho_norm_top = jnp.interp(
        1.0 - pedestal_width_psi_norm.astype(dtype),
        _psi_norm(core_profiles),
        _rho_norm_cells_and_boundaries(geo),
    )

    set_p_top_params = set_ptop_ttopratio_ntop.RuntimeParams(
        pedestal_profile_form=params.pedestal_profile_form,
        # Convert pressure from kPa to Pa.
        p_top=P_top_kPa.astype(dtype) * 1e3,
        n_e_top=params.n_e_top,
        T_i_T_e_ratio=params.T_i_T_e_ratio,
        rho_norm_top=rho_norm_top,
        n_e_top_is_f_GW=params.n_e_top_is_f_GW,
        p_top_multiplier=params.p_top_multiplier,
    )
    set_p_top_runtime_params = dataclasses.replace(
        runtime_params,
        pedestal=dataclasses.replace(
            runtime_params.pedestal, structure=set_p_top_params
        ),
    )
    structure, _ = _SET_P_TOP_MODEL(
        set_p_top_runtime_params,
        geo,
        core_profiles,
        torax.pedestal.StructureState(),
    )
    return structure, EPEDNNmitStructureState(rho_norm_top=rho_norm_top)


def _rho_norm_cells_and_boundaries(geo: torax.Geometry) -> jax.Array:
  """rho_norm of the cell centers plus both boundary faces.

  Paired with ``CellVariable.cell_plus_boundaries``, this gives the
  piecewise-linear profile of the finite-volume solution. It reaches the LCFS,
  so a pedestal top beyond the last cell center maps without clamping, and the
  drop to the boundary value stays within the last half cell.
  """
  return jnp.concatenate(
      [geo.rho_face_norm[:1], geo.rho_norm, geo.rho_face_norm[-1:]]
  )


def _psi_norm(core_profiles: torax.CoreProfiles) -> jax.Array:
  """Normalized poloidal flux on the cell-and-boundary grid, spanning [0, 1]."""
  psi = core_profiles.poloidal_flux.psi.cell_plus_boundaries()
  return (psi - psi[0]) / (psi[-1] - psi[0])


class EPEDNNmitConfig(torax.pedestal.StructureConfigBase):
  """TORAX pedestal structure model config using EPEDNN-mit.

  Attributes:
    machine: Which trained EPEDNN-mit network to use.
    n_e_top: The electron density at the pedestal top [m^-3, or Greenwald
      fraction if ``n_e_top_is_f_GW``].
    n_e_top_is_f_GW: Whether ``n_e_top`` is a Greenwald fraction instead of
      [m^-3].
    T_i_T_e_ratio: Ratio of the ion and electron temperature at the pedestal
      [dimensionless].
    p_top_multiplier: Multiplier for the pedestal pressure (mostly used for
      sensitivity analysis) [dimensionless].
  """

  model_name: Annotated[Literal["epednn_mit"], torax_pydantic.JAX_STATIC] = (
      "epednn_mit"
  )
  machine: Annotated[EPEDNNmitMachine, torax_pydantic.JAX_STATIC] = "sparc"
  n_e_top: torax_pydantic.PositiveTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.7e20)
  )
  n_e_top_is_f_GW: bool = False
  T_i_T_e_ratio: torax_pydantic.PositiveTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(1.0)
  )
  p_top_multiplier: torax_pydantic.PositiveTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(1.0)
  )

  def build_structure_model(self) -> EPEDNNmitPedestalModel:
    return EPEDNNmitPedestalModel(machine=self.machine)

  def build_runtime_params(self, t: chex.Numeric) -> RuntimeParams:
    return RuntimeParams(
        **vars(super().build_runtime_params(t)),
        n_e_top=self.n_e_top.get_value(t),
        n_e_top_is_f_GW=self.n_e_top_is_f_GW,
        T_i_T_e_ratio=self.T_i_T_e_ratio.get_value(t),
        p_top_multiplier=self.p_top_multiplier.get_value(t),
    )
