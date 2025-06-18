import pandas as pd
import scipy.stats as stats
import scikit_posthocs as sp

# Load data
metadata_path = "/Users/charles/Documents/Articles/2024/DAO/Article_Files/DAO_Governance_data.csv"

# Compute groups and capture thresholds
lifespan_groups, lifespan_bins = pd.qcut(
    df['Lifespan (Days)'],
    q=3,
    labels=["court", "moyen", "long"],
    retbins=True
)
member_groups, member_bins = pd.qcut(
    df['Member Count'],
    q=3,
    labels=["faible", "moyen", "élevé"],
    retbins=True
)

# Assign groups back to dataframe
df['Lifespan_Group'] = lifespan_groups
df['Member_Group'] = member_groups

# Display thresholds
print("Lifespan thresholds:")
print([round(x, 2) for x in lifespan_bins])
print("Member count thresholds:")
print([round(x, 2) for x in member_bins])

# Descriptive statistics by group

def print_stats_by_group(df, group_col, metric):
    grouped = df.groupby(group_col)[metric].agg(['mean', 'std', 'count'])
    print(f"\nAnalyse pour '{metric}' segmenté par '{group_col}':\n")
    print(grouped.round(4))

print_stats_by_group(df, 'Lifespan_Group', 'Average Gini Coefficient')
print_stats_by_group(df, 'Member_Group', 'Average Gini Coefficient')

# Kruskal-Wallis and Dunn post-hoc tests

def kruskal_dunn(df, group_col, metric):
    df_clean = df[[group_col, metric]].dropna().reset_index(drop=True)
    print(f"\nTest de Kruskal-Wallis sur '{metric}' par '{group_col}' :")
    groups = [group[metric] for name, group in df_clean.groupby(group_col)]
    stat, p = stats.kruskal(*groups)
    print(f"Statistique H = {stat:.4f}, p-value = {p:.4g}")
    if p < 0.05:
        print("\nTest post-hoc de Dunn (correction Bonferroni) :")
        posthoc = sp.posthoc_dunn(df_clean, val_col=metric, group_col=group_col, p_adjust='bonferroni')
        print(posthoc.round(4))
    else:
        print("Pas de différence significative entre les groupes, test post-hoc non effectué.")

kruskal_dunn(df, 'Lifespan_Group', 'Average Gini Coefficient')
kruskal_dunn(df, 'Member_Group', 'Average Gini Coefficient')
