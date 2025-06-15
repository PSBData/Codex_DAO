import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.stats import entropy, pearsonr


def gini_coefficient(v):
    v = np.array(v)
    v = v[v > 0]
    if len(v) == 0:
        return None
    v_sorted = np.sort(v)
    n = len(v)
    cumvals = np.cumsum(v_sorted)
    gini = (2 * np.sum((np.arange(1, n + 1)) * v_sorted)) / (n * cumvals[-1]) - (n + 1) / n
    return gini


def shannon_entropy(v):
    v = np.array(v)
    v = v[v > 0]
    if len(v) == 0:
        return None
    p = v / v.sum()
    return entropy(p)


def theil_index(v):
    v = np.array(v)
    v = v[v > 0]
    if len(v) == 0:
        return None
    mean = v.mean()
    if mean == 0:
        return None
    ratios = v / mean
    return np.mean(ratios * np.log(ratios))


def top_ratio(v, frac):
    v = np.array(v)
    v = v[v > 0]
    if len(v) == 0:
        return None
    v_sorted = np.sort(v)[::-1]
    k = max(1, int(np.ceil(len(v_sorted) * frac)))
    return v_sorted[:k].sum() / v_sorted.sum()


DIRECTORY = "votes_data_karina"

metrics = []
for filename in tqdm(os.listdir(DIRECTORY), desc="Processing files for disparity metrics"):
    if not filename.endswith('.csv'):
        continue
    file_path = os.path.join(DIRECTORY, filename)
    try:
        df = pd.read_csv(file_path)
    except (pd.errors.ParserError, pd.errors.EmptyDataError):
        continue
    df['voting_power'] = pd.to_numeric(df.get('voting_power'), errors='coerce')
    df = df.dropna(subset=['voting_power'])
    if len(df) < 10:
        # ignore proposals with fewer than 10 voters
        continue
    vp = df['voting_power'].values
    vp_norm = (vp - vp.min()) / (vp.max() - vp.min())
    metrics.append({
        'file': filename,
        'gini': gini_coefficient(vp_norm),
        'entropy': shannon_entropy(vp_norm),
        'theil': theil_index(vp_norm),
        'top1': top_ratio(vp_norm, 0.01),
        'top10': top_ratio(vp_norm, 0.10),
        'top50': top_ratio(vp_norm, 0.50)
    })

metrics_df = pd.DataFrame(metrics).dropna()
metrics_df.to_csv('voting_power_disparity_metrics.csv', index=False)

cols = ['gini', 'entropy', 'theil', 'top1', 'top10', 'top50']
corr = pd.DataFrame(index=cols, columns=cols, dtype=float)
pvals = pd.DataFrame(index=cols, columns=cols, dtype=float)
for i in cols:
    for j in cols:
        if i == j:
            corr.loc[i, j] = 1.0
            pvals.loc[i, j] = 0.0
        else:
            r, p = pearsonr(metrics_df[i], metrics_df[j])
            corr.loc[i, j] = r
            pvals.loc[i, j] = p


def significance_stars(p):
    if p <= 0.001:
        return '***'
    elif p <= 0.01:
        return '**'
    elif p <= 0.05:
        return '*'
    else:
        return ''

annotated = corr.copy()
for i in cols:
    for j in cols:
        annotated.loc[i, j] = f"{corr.loc[i, j]:.2f}{significance_stars(pvals.loc[i, j])}"

corr.to_csv('metrics_corr_matrix.csv')
pvals.to_csv('metrics_corr_pvalues.csv')
annotated.to_csv('metrics_corr_annotated.csv')

latex_table = annotated.to_latex(escape=False)
with open('metrics_corr_table.tex', 'w') as f:
    f.write(latex_table)

print(latex_table)
