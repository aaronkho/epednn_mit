"""TORAX interface for the EPEDNN-mit pedestal model.

This model is only valid for the SPARC parameter space, as specified in
https://github.com/aaronkho/epednn_mit/tree/main/src/epednn_mit/models/sparc.

Please cite [M. Muraca et al. 2025 Nucl. Fusion 65
096010](https://doi.org/10.1088/1741-4326/adf656) in any works using this model.
"""

import dataclasses
import functools
import pathlib
from typing import Annotated, Any, Final, Literal, TypeAlias

import chex
from epednn_mit.models.sparc import jax_model as epednn_mit_jax_model
import jax
from jax import numpy as jnp
import jaxtyping as jt
import torax
from torax._src import jax_utils
from torax._src.pedestal_model import set_pped_tpedratio_nped
from torax._src.physics import formulas
from torax._src.torax_pydantic import torax_pydantic
from typing_extensions import override


# pylint: disable=invalid-name

EPEDNNmitStats: TypeAlias = dict[str, jax.Array]
EPEDNNmitParams: TypeAlias = dict[str, Any]

# For definitions of these parameters, see the EPEDNN-mit SPARC README:
# https://github.com/aaronkho/epednn_mit/blob/main/src/epednn_mit/models/sparc/README.txt
_INPUT_BOUNDS: Final[dict[str, tuple[float, float]]] = {
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


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams(torax.pedestal.RuntimeParams):
  """Runtime params for the EPEDNNmitPedestalModel."""

  n_e_ped: jt.Float[jt.Scalar, ""]
  T_i_T_e_ratio: jt.Float[jt.Scalar, ""]
  n_e_ped_is_fGW: jt.Bool[jt.Scalar, ""]


@dataclasses.dataclass(frozen=True, eq=False)
class EPEDNNmitPedestalModel(
    set_pped_tpedratio_nped.SetPressureTemperatureRatioAndDensityPedestalModel
):
  """Pedestal model using EPEDNN-mit to predict pedestal pressure and width."""

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

    inputs = jnp.array(
        [
            core_profiles.Ip_profile_face[-1] * 1e-6,  # [MA]
            geo.B_0,  # [T]
            geo.R_major,  # [m]
            geo.a_minor,  # [m]
            geo.elongation_face[-1],  # []
            geo.delta_face[-1],  # []
            runtime_params.pedestal.n_e_ped * 1e-19,  # [10^19 m^-3]
            beta_N,  # [%]
            Z_eff,  # []
        ],
        # Network was trained with float32
        dtype=jnp.float32,
    )
    inputs = self._check_input_bounds(inputs)
    return inputs

  def _check_input_bounds(self, epednn_mit_inputs: jax.Array) -> jax.Array:
    """Checks that the EPEDNN-mit inputs are within the bounds.

    Uses torax._src.jax_utils.error_if. For runtime checking to be enabled, set
    the environment variable TORAX_ERRORS_ENABLED to True.

    Args:
      epednn_mit_inputs: The EPEDNN-mit inputs.

    Returns:
      The EPEDNN-mit inputs.

    Raises:
      ValueError: If any of the EPEDNN-mit inputs are out of bounds.
    """
    lower = jnp.array([v[0] for v in _INPUT_BOUNDS.values()])
    upper = jnp.array([v[1] for v in _INPUT_BOUNDS.values()])

    # TODO: Propagate this error so that it is seen (but doesn't
    # exit the sim) if TORAX_ERRORS_ENABLED is False.
    epednn_mit_inputs = jax_utils.error_if(
        epednn_mit_inputs,
        jnp.logical_or(
            jnp.any(epednn_mit_inputs < lower),
            jnp.any(epednn_mit_inputs > upper),
        ),
        "One or more EPEDNN-mit inputs are out of bounds of the training"
        " distribution.",
    )
    return epednn_mit_inputs

  @functools.cached_property
  def _model_and_params(
      self,
  ) -> tuple[
      EPEDNNmitStats,
      EPEDNNmitParams,
      epednn_mit_jax_model.EPEDNNmitEnsemble,
  ]:
    """Returns the EPEDNN-mit model and parameters."""
    model_dir = pathlib.Path(epednn_mit_jax_model.__file__).parent
    model_weights = sorted(model_dir.glob("epednn_mit_sparc_*.pkl"))
    stats, params = epednn_mit_jax_model.load_ensemble_params_from_pickle(
        model_weights
    )
    model = epednn_mit_jax_model.EPEDNNmitEnsemble()
    return stats, params, model

  @override
  def _call_implementation(
      self,
      runtime_params: torax.RuntimeParams,
      geo: torax.Geometry,
      core_profiles: torax.CoreProfiles,
  ) -> torax.pedestal.PedestalModelOutput:
    assert isinstance(runtime_params.pedestal, RuntimeParams)

    # Get pedestal pressure and width from EPEDNN-mit.
    stats, params, model = self._model_and_params
    inputs = self._prepare_inputs(runtime_params, geo, core_profiles)
    P_ped_kPa, pedestal_width_psi_norm = model.apply(params, inputs, **stats)

    # Convert pedestal width to rho_norm
    psi_norm = (core_profiles.psi.value - core_profiles.psi.value[0]) / (
        core_profiles.psi.value[-1] - core_profiles.psi.value[0]
    )
    psi_norm_ped_top = 1.0 - pedestal_width_psi_norm
    rho_norm_ped_top = jnp.interp(psi_norm_ped_top, psi_norm, geo.rho_norm)

    # Convert pedestal pressure from kPa to Pa.
    P_ped = P_ped_kPa * 1e3

    # Use the set_pped_tpedratio_nped model to calculate the pedestal profiles.
    super_runtime_params = set_pped_tpedratio_nped.RuntimeParams(
        set_pedestal=runtime_params.pedestal.set_pedestal,
        P_ped=P_ped,
        n_e_ped=runtime_params.pedestal.n_e_ped,
        T_i_T_e_ratio=runtime_params.pedestal.T_i_T_e_ratio,
        rho_norm_ped_top=rho_norm_ped_top,
        n_e_ped_is_fGW=runtime_params.pedestal.n_e_ped_is_fGW,
    )
    modified_runtime_params = dataclasses.replace(
        runtime_params, pedestal=super_runtime_params
    )
    return super()._call_implementation(
        modified_runtime_params, geo, core_profiles
    )


class EPEDNNmitConfig(torax.pedestal.BasePedestal):
  """Uses EPEDNN-mit to predict pedestal pressure and width.

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
    return EPEDNNmitPedestalModel()

  def build_runtime_params(self, t: chex.Numeric) -> RuntimeParams:
    return RuntimeParams(
        set_pedestal=self.set_pedestal.get_value(t),
        n_e_ped=self.n_e_ped.get_value(t),
        n_e_ped_is_fGW=self.n_e_ped_is_fGW,
        T_i_T_e_ratio=self.T_i_T_e_ratio.get_value(t),
    )
