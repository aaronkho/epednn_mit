"""Tests for the Torax EPEDNN-mit interface."""

import copy
import dataclasses
from absl.testing import absltest
from absl.testing import parameterized
from epednn_mit.interfaces import torax_interface as epednn_mit_torax_interface
import jax
from jax import numpy as jnp
import numpy as np
import torax

# pylint: disable=invalid-name

# Register the EPEDNN-mit pedestal structure model with TORAX.
torax.pedestal.register_pedestal_structure_model(
    epednn_mit_torax_interface.EPEDNNmitConfig
)

# Create a TORAX config with the EPEDNN-mit pedestal structure model.
_CONFIG = {
    'profile_conditions': {
        'I_p': 15e6,
        'I_p_source': 'profile_conditions',
        'psi_source': 'j',
    },
    'plasma_composition': {},
    'numerics': {},
    'time_step_calculator': {
        'calculator_type': 'fixed',
        'dt': 0.1,
        't_final': 0.3,
    },
    'geometry': {'model_name': 'circular', 'n_rho': 25},
    'sources': {},
    'solver': {},
    'transport': {},
    'pedestal': {
        'mode': 'internal_boundary_condition',
        'formation_model': {'model_name': 'prescribed'},
        'structure_model': {
            'model_name': 'epednn_mit',
            'n_e_top': 0.7e20,
            'T_i_T_e_ratio': 1.0,
        },
    },
}


class EPEDNNmitPedestalModelTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.torax_config = torax.ToraxConfig.from_dict(copy.deepcopy(_CONFIG))
    self.structure_model = (
        self.torax_config.pedestal.structure_model.build_structure_model()
    )
    self.jitted_structure_model = jax.jit(self.structure_model)
    step_fn = torax.experimental.make_step_fn(self.torax_config)
    state, _ = torax.experimental.get_initial_state_and_post_processed_outputs(
        step_fn
    )
    self.runtime_params = step_fn.runtime_params_provider(t=0.0)
    self.geo = state.geometry
    self.core_profiles = state.core_profiles
    self.structure_state = self.structure_model.initial_state(
        self.runtime_params, self.geo, self.core_profiles
    )

  def _call(self, runtime_params=None, structure_state=None):
    return self.jitted_structure_model(
        runtime_params or self.runtime_params,
        self.geo,
        self.core_profiles,
        structure_state or self.structure_state,
    )

  def test_build_and_call_structure_model(self):
    """Tests the EPEDNN-mit pedestal structure model.

    Note that the EPEDNN-mit is only valid for SPARC parameter space, but we're
    testing here with a generic config. Hence, we don't perform checks on
    the values of the model predictions, but only that the values set in the
    config are passed through to the output and the outputs are physical.
    """
    assert isinstance(
        self.structure_model, epednn_mit_torax_interface.EPEDNNmitPedestalModel
    )
    structure, _ = self._call()

    # These values come from the config, not the model.
    np.testing.assert_allclose(structure.n_e_top, 0.7e20)
    np.testing.assert_allclose(structure.T_i_top / structure.T_e_top, 1.0)
    self.assertTrue(np.isfinite(structure.T_e_top))
    self.assertGreater(structure.T_e_top, 0.0)
    self.assertBetween(float(structure.rho_norm_top), 0.0, 1.0)

  def test_pedestal_pressure_multiplier(self):
    """Tests that p_top_multiplier scales the temperature outputs."""
    # Get outputs with multiplier 1.0
    structure_1, _ = self._call()

    # Replace the multiplier in runtime params and re-run
    runtime_params = dataclasses.replace(
        self.runtime_params,
        pedestal=dataclasses.replace(
            self.runtime_params.pedestal,
            structure=dataclasses.replace(
                self.runtime_params.pedestal.structure, p_top_multiplier=1.5
            ),
        ),
    )
    structure_2, _ = self._call(runtime_params=runtime_params)

    np.testing.assert_allclose(structure_2.T_e_top, structure_1.T_e_top * 1.5)
    np.testing.assert_allclose(structure_2.T_i_top, structure_1.T_i_top * 1.5)
    np.testing.assert_allclose(structure_2.n_e_top, structure_1.n_e_top)

  def test_state_carries_pedestal_top(self):
    """The returned state holds rho_norm_top, and feeds the next call."""
    self.assertTrue(np.isinf(self.structure_state.rho_norm_top))
    structure, state = self._call()
    np.testing.assert_array_equal(state.rho_norm_top, structure.rho_norm_top)

    # The first call samples about the 0.9 guess. A state elsewhere moves the
    # sampled n_e_ped and Z_eff, so the inputs differ.
    prepare = self.structure_model._prepare_inputs  # pylint: disable=protected-access
    inputs_guess = prepare(self.geo, self.core_profiles, jnp.inf)
    inputs_state = prepare(self.geo, self.core_profiles, jnp.array(0.7))
    self.assertFalse(np.allclose(inputs_guess, inputs_state))

  def test_n_e_ped_at_last_cell_center(self):
    """A pedestal top at the last cell center samples that cell's n_e."""
    n_e = self.core_profiles.kinetic.n_e
    n_e_sep = n_e.face_value()[-1]
    C = 2.0 * np.tanh(1.0) / (np.tanh(1.0) + np.tanh(2.0))
    inputs = self.structure_model._prepare_inputs(  # pylint: disable=protected-access
        self.geo, self.core_profiles, self.geo.rho_norm[-1]
    )
    np.testing.assert_allclose(
        inputs[6] * 1e19, n_e_sep + (n_e.value[-1] - n_e_sep) * C, rtol=1e-6
    )

  def test_inputs_are_float32(self):
    """Inputs reach the float32-trained network in float32, inside bounds."""
    model = self.structure_model.model
    inputs = self.structure_model._prepare_inputs(  # pylint: disable=protected-access
        self.geo, self.core_profiles, jnp.inf
    )
    self.assertEqual(inputs.dtype, jnp.float32)
    self.assertTrue(np.all(inputs >= model.input_lower_bounds))
    self.assertTrue(np.all(inputs <= model.input_upper_bounds))

  def test_wrapper_matches_reference_vector(self):
    """The wrapper reproduces the network's own reference vector.

    Input and expected output are those of
    ``models.sparc.jax_model.test_epednn_mit_sparc_jax``.
    """
    wrapper = epednn_mit_torax_interface.EPEDNNmitPedestalModelWrapper('sparc')
    test_in = jnp.array(
        [10.0, 9.0, 1.85, 0.57, 2.0, 0.45, 30.0, 1.2, 2.0], dtype=jnp.float32
    )
    P_top_kPa, pedestal_width_psi_norm = wrapper(test_in)
    np.testing.assert_allclose(P_top_kPa, 3.19934668e02, rtol=1e-4)
    np.testing.assert_allclose(
        pedestal_width_psi_norm, 4.11556657e-02, rtol=1e-4
    )

  def test_run_simulation(self):
    """A short simulation with EPEDNN-mit as the pedestal reaches t_final."""
    state_history = torax.run_simulation(
        self.torax_config, progress_bar=False
    )
    self.assertEqual(state_history.sim_error, torax.SimError.NO_ERROR)


if __name__ == '__main__':
  absltest.main()
