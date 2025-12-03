import pandas as pd
from scipy.stats import shapiro, ttest_rel, wilcoxon
import numpy as np
import glob

def collect_results(path_pattern):
    files = glob.glob(path_pattern)
    all_sr = []
    for f in files:
        df = pd.read_csv(f)
        final = df[df.get("final") == 1]["success_rate"].values
        if len(final) > 0:
            all_sr.append(final[0])
    return np.array(all_sr)

baseline = collect_results("results/dqn_small_seed*/eval_log.csv")
ours     = collect_results("results/gat_rnn_dqn_small_seed*/eval_log.csv")

print("Shapiro p =", shapiro(ours).pvalue)

if shapiro(ours).pvalue > 0.05:
    print("t-test p =", ttest_rel(ours, baseline).pvalue)
else:
    print("Wilcoxon p =", wilcoxon(ours - baseline).pvalue)

d = (ours.mean() - baseline.mean()) / np.std(ours - baseline)
print("Effect size d =", d)
