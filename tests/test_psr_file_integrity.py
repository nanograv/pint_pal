from os.path import join, exists
from glob import glob
import yaml
import pytest

yamls = sorted(glob("configs/[BJ]*.yaml"))
results = sorted(glob("results/*.par"))

@pytest.fixture(scope="module")
def find_bad():
    bad_yamls = {}
    results = {p:[] for p in glob("results/*.par")}
    bad_results = {}
    bad_comparisons = {}
    for c in sorted(glob("configs/[BJ]*.yaml")):
        try:
            with open(c) as f:
                y = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.parser.ParserError as e:
            bad_yamls[c] = e
            continue
        p = join("results", y["timing-model"])
        try:
            results[p].append(c)
        except KeyError:
            bad_results[c] = p

        if y["compare-model"] is None:
            bad_comparisons[c] = None
        else:
            ap = join("results", y["compare-model"])
            if not exists(ap):
                bad_comparisons[c] = ap

    return bad_yamls, bad_results, bad_comparisons, results

@pytest.mark.parametrize("config", yamls)
def test_name(config):
    with open(config) as f:
        y = yaml.load(f, Loader=yaml.FullLoader)
    source = y["source"]
    assert source[0] in "BJ"
    toa_type = y["toa-type"].lower()
    assert toa_type in {"wb", "nb"}
    assert config == f"configs/{source}.{toa_type}.yaml"

@pytest.mark.parametrize("result_par", results)
def test_results_all_used(result_par, find_bad):
    _, _, _, configs = find_bad
    assert configs[result_par]

@pytest.mark.parametrize("result_par", results)
def test_result_not_shared(result_par, find_bad):
    _, _, _, configs = find_bad
    assert len(configs[result_par])<=1

@pytest.mark.parametrize("config", yamls)
def test_par_exists(config):
    with open(config) as f:
        y = yaml.load(f, Loader=yaml.FullLoader)
    assert exists(join("results", y["timing-model"]))

@pytest.mark.parametrize("config", yamls)
def test_comparison_par(config):
    with open(config) as f:
        y = yaml.load(f, Loader=yaml.FullLoader)
    assert y["compare-model"] is not None
    assert exists(join("results", y["compare-model"]))
