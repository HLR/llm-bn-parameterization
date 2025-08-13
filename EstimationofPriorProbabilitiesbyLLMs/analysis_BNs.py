import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from utils.helpers import get_display_name, create_feature_data, \
    model_name_translation
from utils.helpers import create_model_palette

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
})

llm_models = ['GPT-4o (SepState)', 'GPT-4o (FullDist)', 'GPT-4o (Token Probability)', 'GPT-4o (Without Context)', 'GPT-4o (Random)',\
 'GPT-4o-mini (SepState)', 'GPT-4o-mini (FullDist)', 'GPT-4o-mini (Token Probability)', 'GPT-4o-mini (Without Context)', 'GPT-4o-mini (Random)',\
 'DeekSeek-V3 (SepState)', 'DeekSeek-V3 (FullDist)', 'DeekSeek-V3 (Token Probability)', 'DeekSeek-V3 (Without Context)', 'DeekSeek-V3 (Random)',\
"Gemini Pro (SepState)","Claude 3.5 (SepState)","Gemini Pro (FullDist)","Claude 3.5 (FullDist)",\
 'O3 (SepState)', 'O3 (FullDist)',\
 'O3-mini (SepState)', 'O3-mini (FullDist)',\
 'DeekSeek-R1 (SepState)', 'DeekSeek-R1 (FullDist)',\
 'Uniform', 'Random']

parser = argparse.ArgumentParser()
parser.add_argument('--models',dest='models',nargs='+',default=llm_models, help="Specify one or more models.")
parser.add_argument('--features', dest='features', nargs='+', default=[],choices=["Number of States",'Number of Parents','Type', 'Structure', 'Probabilities', 'Graph', 'Area', 'entropy_category','predicted_entropy_category'], help="Specify one or more features.")
parser.add_argument('--metric', dest='metric', default='BN KL Divergence',choices=['CPT KL Divergence','BN KL Divergence'], help="Specify the metric to plot.")
parser.add_argument('--ax_fontsize', type=int, default=17, help="Font size for axis labels and tick labels.")
parser.add_argument('--legend_fontsize', type=int, default=21, help="Font size for legend text.")
parser.add_argument('--title_fontsize', type=int, default=24, help="Font size for plot titles.")
args = parser.parse_args()

args.models = [model_name_translation[name] for name in args.models]
metric = args.metric

def create_category_plot(feature_data, feature, category_value):
    filtered_data = feature_data[feature_data[feature] == category_value]

    if filtered_data.empty:
        print(f"No data found for {feature} = {category_value}")
        return

    model_medians = {}
    for model in filtered_data['Model'].unique():
        model_data = filtered_data[filtered_data['Model'] == model]
        if not model_data.empty:
            median_value = model_data[metric].median()
            model_medians[model] = median_value
            display_name = filtered_data[filtered_data['Model'] == model]['DisplayName'].iloc[0]
            print(f"Model: {display_name} - {feature.replace('_', ' ').title()}={category_value} - Median: {median_value:.4f}")

    sorted_models = sorted(model_medians.keys(), key=lambda x: model_medians[x])
    palette = create_model_palette(sorted_models)
    plt.figure(figsize=(15, 10), dpi=120)
    ax = sns.boxplot(
        data=filtered_data,
        x='Model',
        y=metric,
        hue='Model',
        palette=palette,
        order=sorted_models,
        legend=False,
        width=0.7,
        linewidth=1.5
    )

    unique_networks = filtered_data['dataset'].nunique()
    title = f"KL Divergence by {feature.replace('_', ' ').title()}: {category_value}"
    subtitle = f"({unique_networks} unique networks)"

    plt.suptitle(title, fontsize=args.title_fontsize, y=0.98)
    plt.title(subtitle, fontsize=args.title_fontsize-6)

    plt.xlabel("Models (Sorted by Median KL Divergence)", fontsize=args.ax_fontsize)
    plt.ylabel(f"{metric} Distribution", fontsize=args.ax_fontsize)

    x_tick_labels = []
    for model in sorted_models:
        display_name = filtered_data[filtered_data['Model'] == model]['DisplayName'].iloc[0]
        x_tick_labels.append(display_name)

    x_tick_labels = [r"$\mathbf{Uniform}$" if label == "Uniform" else label for label in x_tick_labels]
    ax.set_xticklabels(x_tick_labels, rotation=45, ha='right', fontsize=args.ax_fontsize)
    plt.yticks(fontsize=args.ax_fontsize)


    plt.axhline(y=0, color='#FF5733', linestyle='--', alpha=0.7, linewidth=1.5, label='Zero Reference')
    plt.ylim(0, min(15.0 if metric == 'BN KL Divergence' else 1.0, filtered_data[metric].max() * 1.1))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for spine in ax.spines.values():
        spine.set_edgecolor('#333333')
        spine.set_linewidth(1.5)
    plt.tight_layout()
    plt.show()

def create_overall_feature_plot(feature_data):
    model_medians = {}
    for model in args.models:
        model_data = feature_data[feature_data['Model'] == model]
        if not model_data.empty:
            median_value = model_data[metric].median()
            model_medians[model] = median_value
            display_name = feature_data[feature_data['Model'] == model]['DisplayName'].iloc[0]
            print(f"Model: {display_name} - Overall Mean: {model_data[metric].mean():.4f}")

    sorted_models = sorted(model_medians.keys(), key=lambda x: model_medians[x])
    palette = create_model_palette(sorted_models)
    plt.figure(figsize=(15, 10), dpi=120)
    ax = sns.boxplot(data=feature_data,x='Model',y=metric,hue='Model',palette=palette,order=sorted_models,legend=False,width=0.7,linewidth=1.5)
    unique_networks = feature_data['dataset'].nunique()
    main_title = f"Distribution of {metric} of Bayesian Networks by Model"
    subtitle = f"({unique_networks} unique networks)"

    plt.suptitle(main_title, fontsize=args.title_fontsize, y=0.98)
    plt.title(subtitle, fontsize=args.title_fontsize-6)

    plt.xlabel("Models (Sorted by Median KL Divergence)", fontsize=args.ax_fontsize)
    plt.ylabel(f"{metric} Distribution", fontsize=args.ax_fontsize)

    x_tick_labels = []
    for model in sorted_models:
        display_name = feature_data[feature_data['Model'] == model]['DisplayName'].iloc[0]
        x_tick_labels.append(display_name)

    x_tick_labels = [r"$\mathbf{Uniform}$" if label == "Uniform" else label for label in x_tick_labels]
    ax.set_xticklabels(x_tick_labels, rotation=45, ha='right', fontsize=args.ax_fontsize)
    plt.yticks(fontsize=args.ax_fontsize)

    plt.axhline(y=0, color='#FF5733', linestyle='--', alpha=0.7, linewidth=1.5)
    plt.ylim(0, min(15.0 if metric == 'BN KL Divergence' else 1.0, feature_data[metric].max() * 1.1))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for spine in ax.spines.values():
        spine.set_edgecolor('#333333')
        spine.set_linewidth(1.5)

    plt.tight_layout()
    plt.show()

feature_data = create_feature_data(args)
feature_data['DisplayName'] = feature_data['Model'].apply(get_display_name)

create_overall_feature_plot(feature_data)
for feature in args.features:
    feature_data = create_feature_data(args, feature)
    feature_data['DisplayName'] = feature_data['Model'].apply(get_display_name)
    print(f"\n===== Processing {feature.replace('_', ' ').title()} ===== ({feature_data[feature].nunique()} unique values)")
    unique_categories = feature_data[feature].unique()
    for category_value in unique_categories:
        create_category_plot(feature_data, feature, category_value)
