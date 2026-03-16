'''
Unit testing for summarizer.py

These tests are performed on modified versions of the
NANOGrav 12.5-year data set (version 3) parameter files
'''

import sys
import pytest
from pathlib import Path
from astropy import log
import pint.models as models
import pint.toa as toa
from pint.fitter import DownhillGLSFitter
from summarizer import Summarizer

from loguru import logger

logger.remove(0)
logger.add(sys.stderr, level="ERROR") # do not show PINT warnings here to avoid clutter

@pytest.fixture(params=['B1855+09', 'J1024-0719'])
def TestSummarizer(request):
    psr = request.param
    parent = Path(__file__).parent
    parfile = str(parent / f"par/{psr}_NANOGrav_12yv4.gls.par")
    timfile = str(parent / f"tim/{psr}_NANOGrav_12yv4.tim")

    model = models.get_model(parfile)
    toas = toa.get_TOAs(timfile)

    fitter = DownhillGLSFitter(toas, model)
    s = Summarizer(fitter, parfile, autorun=False)
    return s

@pytest.mark.filterwarnings("ignore:PINT only supports 'T2CMETHOD IAU2000B'")
def test_init(TestSummarizer):
    summarizer = TestSummarizer
    assert summarizer.fitter.model.PSR.value == summarizer.psr # by construction


@pytest.mark.filterwarnings("ignore:PINT only supports 'T2CMETHOD IAU2000B'")
def test_check_error(TestSummarizer):
    summarizer = TestSummarizer
    pass

@pytest.mark.filterwarnings("ignore:PINT only supports 'T2CMETHOD IAU2000B'")
def test_generate_timing_model_comparison(TestSummarizer):
    summarizer = TestSummarizer
    pass


@pytest.mark.filterwarnings("ignore:PINT only supports 'T2CMETHOD IAU2000B'")
def test_generate_residual_stats(TestSummarizer):
    summarizer = TestSummarizer
    summarizer.generate_residual_stats()
    text = summarizer.report.generate()
    assert "Statistic" in text
    assert "Reduced" in text


@pytest.mark.filterwarnings("ignore:PINT only supports 'T2CMETHOD IAU2000B'")
def test_generate_timing_model_comparison(TestSummarizer):
    summarizer = TestSummarizer
    pass

@pytest.mark.filterwarnings("ignore:PINT only supports 'T2CMETHOD IAU2000B'")
def test_generate_summary_info(TestSummarizer):
    summarizer = TestSummarizer
    summarizer.generate_summary_info()
    text = summarizer.report.generate()
    assert "GMT" in text
    assert "Span" in text


@pytest.mark.filterwarnings("ignore:PINT only supports 'T2CMETHOD IAU2000B'")
def test_generate_software_versions(TestSummarizer):
    summarizer = TestSummarizer
    summarizer.generate_software_versions()
    text = summarizer.report.generate()
    assert "astropy" in text
    assert "enterprise" in text

@pytest.mark.filterwarnings("ignore:PINT only supports 'T2CMETHOD IAU2000B'")
def test_add_summary_plots(TestSummarizer):
    summarizer = TestSummarizer
    summarizer.add_summary_plots()
    text = summarizer.report.generate()
    for i in range(1, 5):
        assert f"summary_plot_{i}" in text


@pytest.mark.filterwarnings("ignore:PINT only supports 'T2CMETHOD IAU2000B'")
def test_output_pdf(TestSummarizer):
    summarizer = TestSummarizer
    pass
        

