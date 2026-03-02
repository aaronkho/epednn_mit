"""Tests for the Torax EPEDNN-mit interface."""

import dataclasses

from absl.testing import absltest
from absl.testing import parameterized
from epednn_mit.interfaces import torax_interface as epednn_mit_torax_interface
import jax
import numpy as np
import torax
from torax._src import jax_utils

# pylint: disable=invalid-name


class EPEDNNmitPedestalModelTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()

    # Register the EPEDNN-mit pedestal model with TORAX.
    torax.pedestal.register_pedestal_model(
        epednn_mit_torax_interface.EPEDNNmitConfig
    )

    # Create a TORAX config with the EPEDNN-mit pedestal model.
    config = {
        'profile_conditions': {},
        'plasma_composition': {},
        'numerics': {},
        'geometry': {'geometry_type': 'circular', 'n_rho': 4},
        'sources': {},
        'solver': {},
        'transport': {},
        'pedestal': {
            'model_name': 'epednn_mit',
            'set_pedestal': True,
            'n_e_ped': 0.7e20,
            'n_e_ped_is_fGW': False,
            'T_i_T_e_ratio': 1.0,
        },
    }
    torax_config = torax.ToraxConfig.from_dict(config)
    self.pedestal_model = torax_config.pedestal.build_pedestal_model()
    self.jitted_pedestal_model = jax.jit(self.pedestal_model)
    step_fn = torax.experimental.make_step_fn(torax_config)
    state, _ = torax.experimental.get_initial_state_and_post_processed_outputs(
        step_fn
    )
    self.runtime_params = step_fn.runtime_params_provider(t=0.0)
    self.geo = state.geometry
    self.core_profiles = state.core_profiles
    self.source_profiles = state.core_sources

  def test_build_and_call_pedestal_model(self):
    """Tests the EPEDNN-mit pedestal model.

    Note that the EPEDNN-mit is only valid for SPARC parameter space, but we're
    testing here with a generic config. Hence, we don't perform checks on
    the values of the model predictions, but only that the values set in the
    config are passed through to the output.
    """
    assert isinstance(
        self.runtime_params.pedestal, epednn_mit_torax_interface.RuntimeParams
    )
    assert isinstance(
        self.pedestal_model, epednn_mit_torax_interface.EPEDNNmitPedestalModel
    )

    pedestal_model_output = self.jitted_pedestal_model(
        runtime_params=self.runtime_params,
        geo=self.geo,
        core_profiles=self.core_profiles,
        source_profiles=self.source_profiles,
    )

    # These values come from the config, not the model.
    np.testing.assert_allclose(pedestal_model_output.n_e_ped, 0.7e20)
    np.testing.assert_allclose(
        pedestal_model_output.T_i_ped / pedestal_model_output.T_e_ped, 1.0
    )

  def test_out_of_bounds_input_raises_error(self):
    modified_geo = dataclasses.replace(
        self.geo,
        R_major=1000.0,
    )
    with jax_utils.enable_errors(True):
      with self.assertRaises(RuntimeError):
        self.jitted_pedestal_model(
            runtime_params=self.runtime_params,
            geo=modified_geo,
            core_profiles=self.core_profiles,
            source_profiles=self.source_profiles,
        )


if __name__ == '__main__':
  absltest.main()
