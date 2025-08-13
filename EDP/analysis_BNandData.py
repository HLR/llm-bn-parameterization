import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import create_custom_palette

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
})

families = {
        "gpt-4o-mini": "GPT-4o-mini",
        "gpt-4o":      "GPT-4o",
        "o3-mini":     "O3-mini",
        "o3":          "O3",
        "gemini-pro":  "Gemini Pro",
        "claude-3.5":  "Claude 3.5",
        "deepseek":    "DeepSeek",
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="gpt-4o", type=str, help='Path to the new CSV file.')
    parser.add_argument('--csv_file', default="comparison_results.csv", type=str,help='Path to the new CSV file.')
    parser.add_argument('--learning_method', default="counting", type=str,choices=["counting", "Dirichlet"],help='Filter for which learning method to plot.')
    parser.add_argument('--metric', dest='metric', default='BN KL Divergence', choices=['CPT KL Divergence', 'BN KL Divergence'], help="Specify the metric to plot.")
    parser.add_argument('--title_fontsize', type=int, default=24,help="Font size for plot titles.")
    parser.add_argument('--ax_fontsize', type=int, default=17,help="Font size for axis labels and ticks.")
    parser.add_argument('--legend_fontsize', type=int, default=21,help="Font size for legend text.")
    parser.add_argument('--min_args', type=int, default=11,help="Models with sample_num are selected for data. Default: 10")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_file)
    df = df[(df["llm_model"] == args.model) | (df["llm_model"] == "Mean")].copy()

    df = df[df['learning_method'].str.lower() == args.learning_method.lower()]
    df['ModelSample'] = (df['llm_model'].str.replace("Mean", "Uniform")+ ' ('+ df['sample_num'].astype(str)+ ')')
    if args.metric == 'CPT KL Divergence':
        chosen_metric = 'combined_kl_avg'
    else:
        chosen_metric = 'combined_BN_KL_divergence'
    data_rows = []
    for idx, row in df.iterrows():

        if int(row['sample_num'])==10:
            new_row = row.copy()
            new_row['ModelSample'] = f"{row['llm_model'].replace(args.model, 'SepState').replace('Mean', 'Uniform')}"
            if args.metric == 'CPT KL Divergence':
                new_row[chosen_metric] = row['initial_kl_avg']
            else:
                new_row[chosen_metric] = row['initial_BN_KL_divergence']
            data_rows.append(new_row)
        if int(row['sample_num'])<args.min_args or row['llm_model']=='Uniform':
            continue
        new_row = row.copy()
        new_row['llm_model'] = 'Data'
        new_row['ModelSample'] = f"MLE-{row['sample_num']}"
        if args.metric == 'CPT KL Divergence':
            new_row[chosen_metric] = row['learned_kl_avg']
        else:
            new_row[chosen_metric] = row['learned_BN_KL_divergence']
        data_rows.append(new_row)
    df_extra = pd.DataFrame(data_rows)
    df = pd.concat([df, df_extra], ignore_index=True)

    plot_overall(df, chosen_metric, args)


def plot_overall(df, metric, args):
    """
    Create a boxplot across all ModelSamples (including newly added 'data_S#').
    """
    # 1) Compute median for sorting
    df["ModelSample"] = df["ModelSample"].str.replace(args.model, "EDP").str.replace("(", "-").str.replace(")", "").str.replace(" ", "")
    #df["ModelSample"] = df["ModelSample"].str.replace("GPT-4o", "EDP").str.replace("(", "-").str.replace(")","").str.replace(" ", "")
    model_medians = {}
    for ms in df['ModelSample'].unique():
        ms_data = df[df['ModelSample'] == ms]
        if not ms_data.empty:
            median_val = ms_data[metric].median()
            model_medians[ms] = median_val

    sorted_ms = sorted(model_medians.keys(), key=lambda x: model_medians[x])
    palette = create_custom_palette(sorted_ms)
    spacer = pd.DataFrame({metric: [100], 'ModelSample': ['']})
    df_plot = pd.concat([df, spacer])

    order_gap = sorted_ms[:sorted_ms.index("SepState")] + [''] + sorted_ms[sorted_ms.index("SepState"):]
    palette[''] = (0, 0, 0, 0)  # fully transparent

    plt.figure(figsize=(15, 10), dpi=120)

    ax = sns.boxplot(
        data=df_plot,
        x='ModelSample',
        y=metric,
        hue='ModelSample',
        palette=palette,
        order=order_gap,
        width=0.7,
        linewidth=1.5,
        showfliers=True
    )
    ax.axvline(order_gap.index('') , ls='--', c='k', lw=2.5, alpha=0.8)

    method_nice_name = {
        'Dirichlet': 'Linear Pooling',
        'counting': 'Priors as Psuedocounts'
    }
    unique_networks = df['model_name'].nunique()
    main_title = f"Distribution of {args.metric} of Bayesian Networks by Model"
    subtitle = f"({unique_networks} unique networks)"

    plt.suptitle(main_title, fontsize=args.title_fontsize, y=0.98)
    plt.title(subtitle, fontsize=args.title_fontsize - 6)
    plt.xlabel("Models (Sorted by Median KL Divergence)", fontsize=args.ax_fontsize)
    plt.ylabel(f"{args.metric} Distribution", fontsize=args.ax_fontsize)
    plt.xticks(rotation=45, ha='right', fontsize=args.ax_fontsize)
    plt.yticks(fontsize=args.ax_fontsize)

    plt.axhline(y=0, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
    min_val = df[metric].min()
    max_val = df[metric].max()
    plt.ylim(bottom=0, top=min(1.0 if args.metric == 'CPT KL Divergence' else 2.5,max_val * 1.1))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for spine in ax.spines.values():
        spine.set_edgecolor('#333333')
        spine.set_linewidth(1.5)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
