from os.path import join, exists
from glob import glob
import yaml

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

if bad_yamls:
    raise ValueError(f"Some yamls do not parse correctly: {bad_yamls}")
if bad_results:
    raise ValueError(f"Some par files are referenced but do not exist: {bad_results}")
if bad_comparisons:
    raise ValueError(f"Some comparison par files are referenced but do not exist: {bad_comparisons}")
wrong_number = {r:l for r,l in results.items() if len(l)!=1}
if wrong_number:
    raise ValueError(f"Some par files are referenced by no or more than one config yaml: {wrong_number}")
