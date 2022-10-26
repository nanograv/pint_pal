from glob import glob
import os.path
import pickle
import timing_analysis.noise_utils as nu

chain_dir = "/nanograv/share/15yr/timing/intermediate/20220314.Release.nb.78afc797/"

chain_files = glob(chain_dir+"*.nb.*chain_1.txt")
psrs = sorted([os.path.split(x)[1].split(".")[0] for x in chain_files])
results = {}
for psr in psrs:
    results[psr] = nu.analyze_noise(chain_dir+psr+".nb.", save_corner=False, no_corner_plot=True)

with open("noise_results_20220314.pickle", "wb") as f:
    pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)
