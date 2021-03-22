from os.path import join, exists
from glob import glob
import yaml
import pytest

yamls = sorted(glob("configs/[BJ]*.yaml"))

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

def test_yaml_ok(find_bad):
    bad_yamls, _, _, _ = find_bad
    if bad_yamls:
        raise ValueError(f"Some yamls do not parse correctly: {bad_yamls}")

def test_par_files_exist(find_bad):
    _, bad_results, _, _ = find_bad
    if bad_results:
        raise ValueError(f"Some par files are referenced but do not exist: {bad_results}")

def test_compare_par_files_exist(find_bad):
    _, _, bad_comparisons, _ = find_bad
    if bad_comparisons:
        raise ValueError(f"Some comparison par files are referenced but do not exist: {bad_comparisons}")

def test_par_files_not_shared(find_bad):
    _, _, _, results = find_bad
    wrong_number = {r:l for r,l in results.items() if len(l)!=1}
    if wrong_number:
        raise ValueError(f"Some par files are referenced by no or more than one config yaml: {wrong_number}")

@pytest.mark.parametrize("config", yamls)
def test_name(config):
    with open(config) as f:
        y = yaml.load(f, Loader=yaml.FullLoader)
    source = y["source"]
    assert source[0] in "BJ"
    toa_type = y["toa-type"].lower()
    assert toa_type in {"wb", "nb"}
    assert config == f"configs/{source}.{toa_type}.yaml"

