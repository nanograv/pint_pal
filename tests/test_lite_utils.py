import pytest
from pathlib import Path
import numpy as np
import astropy.units as u
import pint.toa as toa
import pint.models as models
from pint.models.parameter import maskParameter
from pint.models.timing_model import Component

from pint_pal.lite_utils import (
    add_feJumps,
    add_dips_to_model,
    add_chromatic_gaussians_to_model,
    convert_equad_convention,
)

@pytest.fixture
def model():
    parent = Path(__file__).parent
    parfile = parent / "par/J2022+2534.basic.par"
    return models.get_model(parfile)
@pytest.fixture
def toas():
    parent = Path(__file__).parent
    timfile = parent / "tim/J2022+2534_15y_L-S_nb.tim"
    return toa.get_TOAs(timfile)


@pytest.mark.filterwarnings("ignore:PINT only supports 'T2CMETHOD IAU2000B'")
def test_add_feJump(model, toas):
    """Check N-1 jumps added if N receivers present"""
    assert not any('JUMP' in p for p in model.params)

    receivers = set(toas.get_flag_value('fe')[0])
    add_feJumps(model, list(receivers))
        
    # Assert proper number of fe jumps have been added (Nrec-1)
    all_jumps = model.components['PhaseJump'].get_jump_param_objects()
    jump_rcvrs = [x.key_value[0] for x in all_jumps if x.key == '-fe']
    assert len(jump_rcvrs) == len(receivers)-1


# ---------------------------------------------------------------------------
# EQUAD convention conversion
# ---------------------------------------------------------------------------

PSR = 'J2022+2534'
EFACS = {'Rcvr1_2_GUPPI': 1.12, 'Rcvr_800_GUPPI': 0.97}
EQUADS = {'Rcvr1_2_GUPPI': -6.5, 'Rcvr_800_GUPPI': -6.9}


def _wn_dict(suffix):
    """White-noise dictionary with EQUADs named by *suffix*."""
    d = {}
    for be in EFACS:
        d[f'{PSR}_{be}_efac'] = EFACS[be]
        d[f'{PSR}_{be}_{suffix}'] = EQUADS[be]
    return d


@pytest.mark.parametrize('suffix', ['log10_tnequad', 'log10_equad'])
def test_convert_equad_convention_tn_to_t2(suffix):
    """Temponest EQUADs (v3.3.0+ and pre-v3.3.0 names) are divided by their own EFAC."""
    converted = convert_equad_convention(_wn_dict(suffix), convention='t2equad')

    assert len(converted) == len(EFACS) * 2
    for be in EFACS:
        assert f'{PSR}_{be}_{suffix}' not in converted
        assert converted[f'{PSR}_{be}_efac'] == EFACS[be]
        assert 10 ** converted[f'{PSR}_{be}_log10_t2equad'] == pytest.approx(
            10 ** EQUADS[be] / EFACS[be]
        )


def test_convert_equad_convention_t2_to_tn():
    """T2 EQUADs are multiplied by their own EFAC going the other way."""
    converted = convert_equad_convention(_wn_dict('log10_t2equad'), convention='tnequad')

    for be in EFACS:
        assert 10 ** converted[f'{PSR}_{be}_log10_tnequad'] == pytest.approx(
            10 ** EQUADS[be] * EFACS[be]
        )


def test_convert_equad_convention_defaults_to_toggle():
    """With no target convention the existing one is toggled."""
    assert all('_log10_t2equad' in k for k in convert_equad_convention(_wn_dict('log10_tnequad'))
               if 'equad' in k)
    assert all('_log10_tnequad' in k for k in convert_equad_convention(_wn_dict('log10_t2equad'))
               if 'equad' in k)


def test_convert_equad_convention_noop_when_already_target():
    """Requesting the convention already in use leaves the dictionary untouched."""
    wn_dict = _wn_dict('log10_t2equad')
    assert convert_equad_convention(wn_dict, convention='t2equad') == wn_dict


def test_convert_equad_convention_pairs_each_equad_with_its_own_efac():
    """EQUADs are matched to EFACs by backend, not by dictionary ordering."""
    wn_dict = {
        f'{PSR}_Rcvr1_2_GUPPI_log10_tnequad': -6.5,
        f'{PSR}_Rcvr_800_GUPPI_efac': 0.97,
        f'{PSR}_Rcvr_800_GUPPI_log10_tnequad': -6.9,
        f'{PSR}_Rcvr1_2_GUPPI_efac': 1.12,
    }
    converted = convert_equad_convention(wn_dict, convention='t2equad')

    for be in EFACS:
        assert 10 ** converted[f'{PSR}_{be}_log10_t2equad'] == pytest.approx(
            10 ** EQUADS[be] / EFACS[be]
        )


def test_convert_equad_convention_leaves_dmequads_alone():
    """DM EQUADs are not TOA EQUADs and must not be converted."""
    wn_dict = _wn_dict('log10_tnequad')
    wn_dict[f'{PSR}_Rcvr1_2_GUPPI_log10_dmequad'] = -5.0
    converted = convert_equad_convention(wn_dict, convention='t2equad')

    assert converted[f'{PSR}_Rcvr1_2_GUPPI_log10_dmequad'] == -5.0


def test_convert_equad_convention_missing_efac_assumes_unity():
    """An EQUAD without a matching EFAC is carried over unchanged."""
    wn_dict = {f'{PSR}_Rcvr1_2_GUPPI_log10_tnequad': -6.5}
    converted = convert_equad_convention(wn_dict, convention='t2equad')

    assert converted[f'{PSR}_Rcvr1_2_GUPPI_log10_t2equad'] == -6.5


def test_convert_equad_convention_rejects_bad_convention():
    with pytest.raises(ValueError):
        convert_equad_convention(_wn_dict('log10_tnequad'), convention='tempo2')



    


# ---------------------------------------------------------------------------
# Exponential dips
# ---------------------------------------------------------------------------

_DIPS = dict(
    epochs=[54757.0, 57510.0],
    ampls=[1.5e-6, 6.0e-7],
    gammas=[-2.0, -1.0],
    taus=[60.0, 30.0],
)


def test_add_dips_to_model_sets_values_and_freezing(model):
    add_dips_to_model(model, **_DIPS)
    comp = model.components['SimpleExponentialDip']
    assert sorted(comp.get_indices()) == [1, 2]
    for i in (1, 2):
        assert model[f'EXPDIPEP_{i}'].value == pytest.approx(_DIPS['epochs'][i - 1])
        assert model[f'EXPDIPAMP_{i}'].value == pytest.approx(_DIPS['ampls'][i - 1])
        assert model[f'EXPDIPIDX_{i}'].value == pytest.approx(_DIPS['gammas'][i - 1])
        assert model[f'EXPDIPTAU_{i}'].value == pytest.approx(_DIPS['taus'][i - 1])
        assert model[f'EXPDIPEP_{i}'].frozen
        for prefix in ('EXPDIPAMP_', 'EXPDIPIDX_', 'EXPDIPTAU_'):
            assert not model[f'{prefix}{i}'].frozen
            assert f'{prefix}{i}' in model.free_params


def test_add_dips_to_model_frozen_and_replaces_existing(model):
    add_dips_to_model(model, **_DIPS)
    add_dips_to_model(model, [55000.0], [1e-6], [-1.5], [40.0], frozen=True)
    comp = model.components['SimpleExponentialDip']
    assert list(comp.get_indices()) == [1]
    assert model.EXPDIPEP_1.value == pytest.approx(55000.0)
    assert not any(p.startswith('EXPDIP') for p in model.free_params)


@pytest.mark.filterwarnings("ignore:PINT only supports 'T2CMETHOD IAU2000B'")
def test_add_dips_to_model_produces_delay(model, toas):
    base = model.delay(toas).to_value(u.s)
    add_dips_to_model(model, [toas.get_mjds().value.mean()], [1e-5], [-2.0], [100.0])
    diff = model.delay(toas).to_value(u.s) - base
    assert np.any(np.abs(diff) > 1e-7)


def test_add_dips_to_model_rejects_mismatched_lengths(model):
    with pytest.raises(ValueError):
        add_dips_to_model(model, [55000.0, 56000.0], [1e-6], [-1.0], [30.0])


# ---------------------------------------------------------------------------
# Chromatic Gaussian events
# ---------------------------------------------------------------------------

_GAUSS = dict(
    epochs=[56000.0, 57500.0],
    log10_amps=[-5.5, -6.0],
    log10_sigmas=[1.5, 2.0],
    signs=[1.0, -1.0],
    chrom_idxs=[4.0, 2.0],
)


def test_add_chromatic_gaussians_sets_values_and_freezing(model):
    add_chromatic_gaussians_to_model(model, **_GAUSS)
    comp = model.components['ChromaticGaussianEvent']
    assert sorted(comp.get_indices()) == [1, 2]
    for i in (1, 2):
        assert model[f'CHROMGAUSS_EPOCH_{i}'].value == pytest.approx(_GAUSS['epochs'][i - 1])
        assert model[f'CHROMGAUSS_LOGAMP_{i}'].value == pytest.approx(_GAUSS['log10_amps'][i - 1])
        assert model[f'CHROMGAUSS_LOGSIG_{i}'].value == pytest.approx(_GAUSS['log10_sigmas'][i - 1])
        assert model[f'CHROMGAUSS_SIGN_{i}'].value == _GAUSS['signs'][i - 1]
        assert model[f'CHROMGAUSS_CHROMIDX_{i}'].value == pytest.approx(_GAUSS['chrom_idxs'][i - 1])
        assert model[f'CHROMGAUSS_EPOCH_{i}'].frozen
        assert model[f'CHROMGAUSS_SIGN_{i}'].frozen
        for prefix in ('CHROMGAUSS_LOGAMP_', 'CHROMGAUSS_LOGSIG_', 'CHROMGAUSS_CHROMIDX_'):
            assert f'{prefix}{i}' in model.free_params


def test_add_chromatic_gaussians_frozen_and_replaces_existing(model):
    add_chromatic_gaussians_to_model(model, **_GAUSS)
    add_chromatic_gaussians_to_model(model, 55000.0, -6.0, 1.0, -1.0, 4.0, frozen=True)
    comp = model.components['ChromaticGaussianEvent']
    assert list(comp.get_indices()) == [1]
    assert model.CHROMGAUSS_EPOCH_1.value == pytest.approx(55000.0)
    assert not any(p.startswith('CHROMGAUSS') for p in model.free_params)


@pytest.mark.filterwarnings("ignore:PINT only supports 'T2CMETHOD IAU2000B'")
def test_add_chromatic_gaussians_matches_discovery_formula(model, toas):
    """PINT delay should equal discovery's chromatic_gaussian:
    sign * 10**log10_Amp * exp(-(t - t0)**2 / (2 sigma**2)) * (fref / f)**alpha."""
    base = model.delay(toas).to_value(u.s)
    t0 = float(np.median(toas.get_mjds().value))
    log10_amp, log10_sigma, sign, alpha = -5.0, 2.0, -1.0, 4.0
    add_chromatic_gaussians_to_model(model, t0, log10_amp, log10_sigma, sign, alpha)
    diff = model.delay(toas).to_value(u.s) - base

    comp = model.components['ChromaticGaussianEvent']
    fnorm = 1.0 / np.asarray(comp.get_ffac(toas))  # fref / f
    t = np.asarray(toas['tdbld'], dtype=float)
    expected = (sign * 10**log10_amp
                * np.exp(-(t - t0)**2 / (2 * (10**log10_sigma)**2))
                * fnorm**alpha)
    assert np.any(np.abs(expected) > 1e-7)
    np.testing.assert_allclose(diff, expected, rtol=1e-6, atol=1e-12)


def test_add_chromatic_gaussians_rejects_bad_input(model):
    with pytest.raises(ValueError):
        add_chromatic_gaussians_to_model(model, [55000.0, 56000.0], [-6.0], [1.0], [1.0], [4.0])
    with pytest.raises(ValueError):
        add_chromatic_gaussians_to_model(model, 55000.0, -6.0, 1.0, 0.0, 4.0)
