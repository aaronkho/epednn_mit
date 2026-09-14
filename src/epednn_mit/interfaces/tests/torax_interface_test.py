"""Tests for the Torax EPEDNN-mit interface."""

import dataclasses
from absl.testing import absltest
from absl.testing import parameterized
from epednn_mit.interfaces import torax_interface as epednn_mit_torax_interface
import jax
import numpy as np
import pydantic
import torax

# pylint: disable=invalid-name


class EPEDNNmitPedestalModelTest(parameterized.TestCase):

  def _build(self, machine: str):
    # Register the EPEDNN-mit pedestal model with TORAX.
    torax.pedestal.register_pedestal_structure_model(
        epednn_mit_torax_interface.EPEDNNmitConfig
    )

    # Create a TORAX config with the EPEDNN-mit pedestal model.
    config = {
        'profile_conditions': {
            'psi_source': 'j',
            'I_p': 15e6,
            'I_p_source': 'profile_conditions',
        },
        'plasma_composition': {},
        'numerics': {},
        'geometry': {'model_name': 'circular', 'n_rho': 4},
        'sources': {},
        'solver': {},
        'transport': {},
        'pedestal': {
            'structure_model': {
                'model_name': 'epednn_mit',
                'machine': machine,
                'n_e_top': 0.7e20,
                'T_i_T_e_ratio': 1.0,
            },
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
    self.pedestal_state = state.model_state.pedestal

  def setUp(self):
    super().setUp()
    self._build('sparc')

  @parameterized.parameters('sparc', 'scoping')
  def test_build_and_call_pedestal_model(self, machine):
    """Tests the EPEDNN-mit pedestal model.

    Note that the EPEDNN-mit machines are only valid within their respective
    training parameter spaces, but we're testing here with a generic config.
    Hence, we don't perform checks on the values of the model predictions,
    but only that the values set in the config are passed through to the
    output.
    """
    self._build(machine)

    assert isinstance(
        self.runtime_params.pedestal.structure,
        epednn_mit_torax_interface.RuntimeParams,
    )
    assert isinstance(
        self.pedestal_model.structure_model,
        epednn_mit_torax_interface.EPEDNNmitPedestalModel,
    )
    assert self.pedestal_model.structure_model.machine == machine

    pedestal_model_output, _ = self.jitted_pedestal_model(
        runtime_params=self.runtime_params,
        geo=self.geo,
        core_profiles=self.core_profiles,
        source_profiles=self.source_profiles,
        pedestal_state=self.pedestal_state,
    )

    # These values come from the config, not the model.
    np.testing.assert_allclose(pedestal_model_output.structure.n_e_top, 0.7e20)
    np.testing.assert_allclose(
        pedestal_model_output.structure.T_i_top
        / pedestal_model_output.structure.T_e_top,
        1.0,
    )

  def test_pedestal_pressure_multiplier(self):
    """Tests that p_top_multiplier scales the temperature outputs."""
    assert isinstance(
        self.runtime_params.pedestal.structure,
        epednn_mit_torax_interface.RuntimeParams,
    )
    assert isinstance(
        self.pedestal_model.structure_model,
        epednn_mit_torax_interface.EPEDNNmitPedestalModel,
    )

    # Get outputs with multiplier 1.0
    pedestal_model_output_1, _ = self.jitted_pedestal_model(
        runtime_params=self.runtime_params,
        geo=self.geo,
        core_profiles=self.core_profiles,
        source_profiles=self.source_profiles,
        pedestal_state=self.pedestal_state,
    )

    # Replace the multiplier in runtime params and re-run
    modified_runtime_params = dataclasses.replace(
        self.runtime_params,
        pedestal=dataclasses.replace(
            self.runtime_params.pedestal,
            structure=dataclasses.replace(
                self.runtime_params.pedestal.structure,
                p_top_multiplier=1.5,
            ),
        ),
    )

    pedestal_model_output_2, _ = self.jitted_pedestal_model(
        runtime_params=modified_runtime_params,
        geo=self.geo,
        core_profiles=self.core_profiles,
        source_profiles=self.source_profiles,
        pedestal_state=self.pedestal_state,
    )

    np.testing.assert_allclose(
        pedestal_model_output_2.structure.T_e_top,
        pedestal_model_output_1.structure.T_e_top * 1.5,
    )
    np.testing.assert_allclose(
        pedestal_model_output_2.structure.T_i_top,
        pedestal_model_output_1.structure.T_i_top * 1.5,
    )
    np.testing.assert_allclose(
        pedestal_model_output_2.structure.n_e_top,
        pedestal_model_output_1.structure.n_e_top,
    )

  def test_structure_state_tracks_pedestal_top(self):
    """Tests that EPED-NN updates rho_norm_top in EpedNnStructureState."""
    structure_model = self.pedestal_model.structure_model
    init_state = structure_model.initial_state(
        self.runtime_params, self.geo, self.core_profiles
    )
    self.assertIsInstance(
        init_state, epednn_mit_torax_interface.EpedNnStructureState
    )
    self.assertFalse(np.isinf(init_state.rho_norm_top))

    pedestal_model_output, updated_structure_state = structure_model(
        self.runtime_params,
        self.geo,
        self.core_profiles,
        init_state,
    )
    np.testing.assert_allclose(
        updated_structure_state.rho_norm_top,
        pedestal_model_output.rho_norm_top,
    )

  def test_pedestal_top_location_from_top_width(self):
    """Tests that psi_norm_top = 1 - pedestal_top_width_psi_norm directly."""
    structure_model = self.pedestal_model.structure_model
    assert isinstance(
        structure_model, epednn_mit_torax_interface.EPEDNNmitPedestalModel
    )
    init_state = structure_model.initial_state(
        self.runtime_params, self.geo, self.core_profiles
    )
    inputs = structure_model._prepare_inputs(
        self.runtime_params,
        self.geo,
        self.core_profiles,
        previous_rho_norm_top=init_state.rho_norm_top,
    )
    _, pedestal_top_width_psi_norm = structure_model.model(inputs)
    structure_output, _ = structure_model(
        self.runtime_params, self.geo, self.core_profiles, init_state
    )
    expected_psi_norm_top = 1.0 - pedestal_top_width_psi_norm
    expected_rho_norm_top = np.interp(
        expected_psi_norm_top,
        self.core_profiles.psi_norm_face,
        self.geo.rho_face_norm,
    )
    np.testing.assert_allclose(
        structure_output.rho_norm_top, expected_rho_norm_top
    )

  def test_prepare_inputs_evaluates_at_mtanh_pedestal_location(self):
    """Tests _prepare_inputs maps rho_norm_top -> psi_ped = 1 - Delta."""
    structure_model = self.pedestal_model.structure_model
    assert isinstance(
        structure_model, epednn_mit_torax_interface.EPEDNNmitPedestalModel
    )
    # Construct profiles strictly within SPARC input bounds (neped in [2.84,
    # 90.235] x 1e19 m^-3, zeff in [1.3, 2.5]) with spatial variation so both
    # the mtanh density conversion and the psi_ped = 1 - Delta Z_eff lookup are
    # distinguishable from psi_top = 1 - 1.5 * Delta.
    n_e_cell = 6.0e20 - 3.0e20 * self.geo.rho_norm
    n_e_sep = 2.5e20
    z_eff_face = 1.4 + 0.9 * self.geo.rho_face_norm
    custom_core_profiles = dataclasses.replace(
        self.core_profiles,
        kinetic=dataclasses.replace(
            self.core_profiles.kinetic,
            n_e=dataclasses.replace(
                self.core_profiles.kinetic.n_e,
                value=n_e_cell,
                right_face_constraint=jax.numpy.asarray(n_e_sep),
                right_face_grad_constraint=None,
            ),
        ),
        ions=dataclasses.replace(
            self.core_profiles.ions,
            Z_eff_face=z_eff_face,
        ),
    )
    previous_rho_norm_top = jax.numpy.asarray(0.88)
    inputs = structure_model._prepare_inputs(
        self.runtime_params,
        self.geo,
        custom_core_profiles,
        previous_rho_norm_top=previous_rho_norm_top,
    )

    psi_norm_face = np.asarray(custom_core_profiles.psi_norm_face)
    rho_face_norm = np.asarray(self.geo.rho_face_norm)
    psi_top = np.interp(0.88, rho_face_norm, psi_norm_face)
    delta_psi = (1.0 - psi_top) / 1.5
    psi_ped = 1.0 - delta_psi
    rho_ped = np.interp(psi_ped, psi_norm_face, rho_face_norm)

    n_e_face = np.asarray(custom_core_profiles.kinetic.n_e.face_value())
    n_e_top = np.interp(0.88, rho_face_norm, n_e_face)
    c_mtanh = 2.0 * np.tanh(1.0) / (np.tanh(1.0) + np.tanh(2.0))
    expected_n_e_ped_1e19 = (n_e_sep + (n_e_top - n_e_sep) * c_mtanh) * 1e-19
    expected_z_eff_ped = np.interp(
        rho_ped, rho_face_norm, np.asarray(z_eff_face)
    )

    self.assertEqual(inputs.dtype, jax.numpy.float32)
    np.testing.assert_allclose(inputs[6], expected_n_e_ped_1e19, rtol=1e-5)
    np.testing.assert_allclose(inputs[8], expected_z_eff_ped, rtol=1e-5)

  def test_pedestal_model_equality_and_wrapper_caching(self):
    """Tests equality, hashing, and wrapper caching across instantiations."""
    model_1 = epednn_mit_torax_interface.EPEDNNmitPedestalModel()
    model_2 = epednn_mit_torax_interface.EPEDNNmitPedestalModel()

    self.assertIs(model_1.model, model_2.model)
    self.assertEqual(model_1, model_2)
    self.assertEqual(hash(model_1), hash(model_2))

    wrapper_1 = epednn_mit_torax_interface.EPEDNNmitPedestalModelWrapper(
        'sparc'
    )
    wrapper_2 = epednn_mit_torax_interface.EPEDNNmitPedestalModelWrapper(
        'sparc'
    )
    self.assertIsNot(wrapper_1, wrapper_2)
    self.assertEqual(wrapper_1, wrapper_2)
    self.assertEqual(hash(wrapper_1), hash(wrapper_2))

  def test_wrapper_float32_inference(self):
    wrapper = epednn_mit_torax_interface.EPEDNNmitPedestalModelWrapper('sparc')
    mid_f64 = jax.numpy.array(
        [
            0.5 * (lo + hi)
            for lo, hi in (
                epednn_mit_torax_interface._SPARC_INPUT_BOUNDS.values()
            )
        ],
        dtype=jax.numpy.float64,
    )
    mid_f32 = mid_f64.astype(jax.numpy.float32)
    p_top_f32, width_f32 = wrapper(mid_f32)
    p_top_f64, width_f64 = wrapper(mid_f64)
    # Without float32 stats/params in the wrapper, the outputs are float64 and
    # evaluating mid_f32 against float64 stats blows up to ~1e9 kPa (because
    # fixed training inputs R, a have variance ~1e-31, amplifying float32-vs-
    # float64 rounding error by ~1e15), failing both the dtype and allclose
    # assertions below. Existing production runs escaped this only because
    # _prepare_inputs clipped against float64 bounds, implicitly promoting the
    # float32 inputs back to float64 before calling the wrapper.
    self.assertEqual(p_top_f32.dtype, jax.numpy.float32)
    self.assertEqual(width_f32.dtype, jax.numpy.float32)
    np.testing.assert_allclose(p_top_f32, p_top_f64)
    np.testing.assert_allclose(width_f32, width_f64)

  def test_initial_state_from_datatree(self):
    structure_model = self.pedestal_model.structure_model
    restored_state = structure_model.initial_state_from_datatree(
        None, self.runtime_params
    )
    self.assertIsInstance(
        restored_state, epednn_mit_torax_interface.EpedNnStructureState
    )
    structure_output, updated_state = structure_model(
        self.runtime_params, self.geo, self.core_profiles, restored_state
    )
    np.testing.assert_allclose(
        updated_state.rho_norm_top, structure_output.rho_norm_top
    )

  def test_structure_state_datatree_roundtrip(self):
    structure_model = self.pedestal_model.structure_model
    stacked_state = epednn_mit_torax_interface.EpedNnStructureState(
        rho_norm_top=jax.numpy.array([0.85, 0.87])
    )
    context = torax.output.OutputGridContext(
        times=np.array([0.0, 0.1]),
        rho_face_norm=self.geo.rho_face_norm,
        rho_cell_norm=self.geo.rho_norm,
        rho_cell_plus_boundaries_norm=np.concatenate([
            [0.0],
            np.asarray(self.geo.rho_norm),
            [1.0],
        ]),
    )
    datatree = stacked_state.to_xr_datatree(context)
    restored = structure_model.initial_state_from_datatree(
        datatree.isel(time=-1), self.runtime_params
    )
    self.assertIsInstance(
        restored, epednn_mit_torax_interface.EpedNnStructureState
    )
    np.testing.assert_allclose(restored.rho_norm_top, 0.87)


  def test_config_validation(self):
    # n_e_top is required.
    with self.assertRaises(pydantic.ValidationError):
      epednn_mit_torax_interface.EPEDNNmitConfig.from_dict(
          {'model_name': 'epednn_mit'}
      )
    # Invalid machine name is rejected.
    with self.assertRaises(pydantic.ValidationError):
      epednn_mit_torax_interface.EPEDNNmitConfig.from_dict(
          {'model_name': 'epednn_mit', 'machine': 'iter', 'n_e_top': 2.0e20}
      )
    # Non-positive scalar fields are rejected.
    for field in ('n_e_top', 'T_i_T_e_ratio', 'p_top_multiplier'):
      kwargs = {'model_name': 'epednn_mit', 'n_e_top': 2.0e20, field: 0.0}
      with self.assertRaises(pydantic.ValidationError):
        epednn_mit_torax_interface.EPEDNNmitConfig.from_dict(kwargs)

  def test_pure_plasma_dilution_gradient_is_finite(self):
    """Tests that Z_eff=1 with Z_impurity_bundled=Z_i yields finite grads."""
    structure_model = self.pedestal_model.structure_model
    init_state = structure_model.initial_state(
        self.runtime_params, self.geo, self.core_profiles
    )
    ones_face = jax.numpy.ones_like(self.core_profiles.ions.Z_eff_face)
    pure_profiles = dataclasses.replace(
        self.core_profiles,
        ions=dataclasses.replace(
            self.core_profiles.ions,
            Z_eff_face=ones_face,
            Z_i_face=ones_face,
            Z_impurity_average_face=ones_face,
            Z_impurity_bundled_face=ones_face,
        ),
    )

    def eval_T_e_top(n_e_top):
      rp = dataclasses.replace(
          self.runtime_params,
          pedestal=dataclasses.replace(
              self.runtime_params.pedestal,
              structure=dataclasses.replace(
                  self.runtime_params.pedestal.structure,
                  n_e_top=n_e_top,
              ),
          ),
      )
      out, _ = structure_model(rp, self.geo, pure_profiles, init_state)
      return out.T_e_top

    val, grad = jax.value_and_grad(eval_T_e_top)(
        jax.numpy.asarray(2.0e20, dtype=self.geo.rho_face_norm.dtype)
    )
    self.assertTrue(np.isfinite(val))
    self.assertTrue(np.isfinite(grad))

  def test_run_simulation(self):
    config = {
        'profile_conditions': {
            'psi_source': 'j',
            'I_p': 15e6,
            'I_p_source': 'profile_conditions',
        },
        'plasma_composition': {},
        'numerics': {},
        'time_step_calculator': {
            'calculator_type': 'fixed',
            'dt': 0.1,
            't_final': 0.3,
        },
        'geometry': {'model_name': 'circular', 'n_rho': 10},
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
    torax_config = torax.ToraxConfig.from_dict(config)
    state_history = torax.run_simulation(torax_config, progress_bar=False)
    self.assertEqual(state_history.sim_error, torax.SimError.NO_ERROR)


if __name__ == '__main__':
  absltest.main()
