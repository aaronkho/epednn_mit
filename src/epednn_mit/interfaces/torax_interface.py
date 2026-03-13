"""TORAX interface for the EPEDNN-mit pedestal model.

Please cite [M. Muraca et al. 2025 Nucl. Fusion 65
096010](https://doi.org/10.1088/1741-4326/adf656) in any works using this model.
Currently, this model is only valid for the SPARC parameter space, as specified
in https://github.com/aaronkho/epednn_mit/tree/main/src/epednn_mit/models/sparc.

This file defines the four necessary classes to use the model in TORAX:
1. A wrapper class, which closes over the EPEDNN-mit model, parameters, and
   statistics. This is necessary to make the model hashable.
2. A TORAX pedestal model, which calls the wrapper within its
   _call_implementation. This is what will be called by TORAX.
3. A TORAX pedestal model Pydantic config, which is used to build the pedestal
   model and runtime params. This allows us to use the model from a TORAX
   config.
4. A runtime params class, which contains any input parameters specific to the
   EPEDNN-mit model. This will be passed to the pedestal model by TORAX at each
   time step.
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
from torax._src.pedestal_model import set_pped_tpedratio_nped
from torax._src.physics import formulas
from torax._src.torax_pydantic import torax_pydantic
from typing_extensions import override

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


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams(torax.pedestal.RuntimeParams):
  """Runtime params for the EPEDNNmitPedestalModel."""

  n_e_ped: jt.Float[jt.Scalar, ""]
  T_i_T_e_ratio: jt.Float[jt.Scalar, ""]
  n_e_ped_is_fGW: jt.Bool[jt.Scalar, ""]


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
        self.input_lower_bounds = jnp.array(
            [_SPARC_INPUT_BOUNDS[key][0] for key in _SPARC_INPUT_BOUNDS]
        )
        self.input_upper_bounds = jnp.array(
            [_SPARC_INPUT_BOUNDS[key][1] for key in _SPARC_INPUT_BOUNDS]
        )
        self.R_0 = _SPARC_DEVICE_MAJOR_RADIUS
        self.a_0 = _SPARC_DEVICE_MINOR_RADIUS
      case _:
        raise ValueError(
            f"Unsupported machine: {machine}. Only SPARC is supported."
        )

  def __call__(self, inputs: jax.Array) -> tuple[jax.Array, jax.Array]:
    P_ped_kPa, pedestal_width_psi_norm = self.model.apply(
        self._params, inputs, **self._stats
    )
    return P_ped_kPa, pedestal_width_psi_norm

  def __hash__(self):
    return hash(self.machine)


@dataclasses.dataclass(frozen=True, eq=False)
class EPEDNNmitPedestalModel(
    set_pped_tpedratio_nped.SetPressureTemperatureRatioAndDensityPedestalModel
):
  """TORAX pedestal model using EPEDNN-mit to predict pressure and width."""

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

  def _prepare_inputs(
      self,
      runtime_params: torax.RuntimeParams,
      geo: torax.Geometry,
      core_profiles: torax.CoreProfiles,
  ) -> jt.Float[jt.Array, "9"]:
    """Prepares the inputs for EPEDNN-mit."""
    assert isinstance(runtime_params.pedestal, RuntimeParams)

    _, _, beta_N = formulas.calculate_betas(core_profiles, geo)

    # TODO: Get Z_eff(rho_norm_ped_top) from t+dt.
    # Currently, we approximate it from the value at rho_norm = 0.9.
    Z_eff_idx = jnp.argmin(jnp.abs(geo.rho_norm - 0.9))
    Z_eff = core_profiles.Z_eff[Z_eff_idx]

    raw_inputs = jnp.array(
        [
            core_profiles.Ip_profile_face[-1] * 1e-6,  # [MA]
            geo.B_0,  # [T]
            self.model.R_0,  # [m]
            self.model.a_0,  # [m]
            geo.elongation_face[-1],  # []
            geo.delta_face[-1],  # []
            runtime_params.pedestal.n_e_ped * 1e-19,  # [10^19 m^-3]
            beta_N,  # [%]
            Z_eff,  # []
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
  def _call_implementation(
      self,
      runtime_params: torax.RuntimeParams,
      geo: torax.Geometry,
      core_profiles: torax.CoreProfiles,
  ) -> torax.pedestal.PedestalModelOutput:
    assert isinstance(runtime_params.pedestal, RuntimeParams)

    # Get pedestal pressure and width from EPEDNN-mit.
    inputs = self._prepare_inputs(runtime_params, geo, core_profiles)
    P_ped_kPa, pedestal_width_psi_norm = self.model(inputs)

    # Convert pedestal width to rho_norm
    psi_norm = (
        core_profiles.psi.value - core_profiles.psi.left_face_value()
    ) / (
        core_profiles.psi.right_face_value - core_profiles.psi.left_face_value()
    )
    psi_norm_ped_top = 1.0 - pedestal_width_psi_norm
    rho_norm_ped_top = jnp.interp(psi_norm_ped_top, psi_norm, geo.rho_norm)

    # Convert pedestal pressure from kPa to Pa.
    P_ped = P_ped_kPa * 1e3

    # Use the set_pped_tpedratio_nped model to calculate the pedestal profiles.
    super_runtime_params = set_pped_tpedratio_nped.RuntimeParams(
        set_pedestal=runtime_params.pedestal.set_pedestal,
        mode=runtime_params.pedestal.mode,
        P_ped=P_ped,
        n_e_ped=runtime_params.pedestal.n_e_ped,
        T_i_T_e_ratio=runtime_params.pedestal.T_i_T_e_ratio,
        rho_norm_ped_top=rho_norm_ped_top,
        n_e_ped_is_fGW=runtime_params.pedestal.n_e_ped_is_fGW,
        formation=runtime_params.pedestal.formation,
        saturation=runtime_params.pedestal.saturation,
        chi_max=runtime_params.pedestal.chi_max,
        D_e_max=runtime_params.pedestal.D_e_max,
        V_e_max=runtime_params.pedestal.V_e_max,
        V_e_min=runtime_params.pedestal.V_e_min,
        pedestal_top_smoothing_width=runtime_params.pedestal.pedestal_top_smoothing_width,
    )
    modified_runtime_params = dataclasses.replace(
        runtime_params, pedestal=super_runtime_params
    )
    return super()._call_implementation(
        modified_runtime_params, geo, core_profiles
    )


class EPEDNNmitConfig(torax.pedestal.BasePedestal):
  """TORAX pedestal model config using EPEDNN-mit.

  Attributes:
    n_e_ped: The electron density at the pedestal [m^-3] or fGW.
    n_e_ped_is_fGW: Whether the electron density at the pedestal is in units of
      fGW.
    T_i_T_e_ratio: Ratio of the ion and electron temperature at the pedestal
      [dimensionless].
  """

  model_name: Annotated[Literal["epednn_mit"], torax_pydantic.JAX_STATIC] = (
      "epednn_mit"
  )
  n_e_ped: torax_pydantic.TimeVaryingScalar = torax_pydantic.ValidatedDefault(
      0.7e20
  )
  n_e_ped_is_fGW: bool = False
  T_i_T_e_ratio: torax_pydantic.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(1.0)
  )

  def build_pedestal_model(
      self,
  ) -> EPEDNNmitPedestalModel:
    return EPEDNNmitPedestalModel(
        formation_model=self.formation_model.build_formation_model(),
        saturation_model=self.saturation_model.build_saturation_model(),
    )

  def build_runtime_params(self, t: chex.Numeric) -> RuntimeParams:
    base_runtime_params = super().build_runtime_params(t)
    return RuntimeParams(
        set_pedestal=base_runtime_params.set_pedestal,
        mode=base_runtime_params.mode,
        formation=base_runtime_params.formation,
        saturation=base_runtime_params.saturation,
        chi_max=base_runtime_params.chi_max,
        D_e_max=base_runtime_params.D_e_max,
        V_e_max=base_runtime_params.V_e_max,
        V_e_min=base_runtime_params.V_e_min,
        pedestal_top_smoothing_width=base_runtime_params.pedestal_top_smoothing_width,
        n_e_ped=self.n_e_ped.get_value(t),
        n_e_ped_is_fGW=self.n_e_ped_is_fGW,
        T_i_T_e_ratio=self.T_i_T_e_ratio.get_value(t),
    )
