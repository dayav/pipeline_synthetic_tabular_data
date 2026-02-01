import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from ipywidgets import Image
import io


def plot_correlation_ratio_diff_(corr_real: dict, corr_synth: dict, norm_diff: float, spearman: float) -> plt.Figure:
    """
    Plots the correlation ratio matrices for the real and synthetic datasets, along with their absolute difference.

    This function generates a three-panel heatmap plot that visualizes the correlation ratio matrices for both 
    the real and synthetic datasets. It also shows the absolute differences between the two matrices and displays 
    the Frobenius norm difference as a title on the third plot. The correlation matrices describe the relationships 
    between numerical and categorical variables.

    Args:
        corr_real (dict): The correlation ratio matrix for the real dataset.
        corr_synth (dict): The correlation ratio matrix for the synthetic dataset.
        norm_diff (float): The Frobenius norm difference between the real and synthetic correlation ratio matrices.

    Returns:
        plt.Figure: A Matplotlib figure object containing the three heatmaps: real, synthetic, and difference matrices.

    Example:
        fig = plot_correlation_ratio_diff_(corr_real, corr_synth, norm_diff)
        This function generates a heatmap plot that compares the real and synthetic correlation ratio matrices 
        and their differences.

    Notes:
        - The first heatmap shows the correlation matrix for the real dataset.
        - The second heatmap shows the correlation matrix for the synthetic dataset.
        - The third heatmap displays the absolute differences between the two matrices and includes the Frobenius norm difference.
        - The correlation ratio measures the relationship between numerical and categorical variables.
    """

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 3.5), sharey=True)

    axs[0].set_title('Real')

    g2 = sns.heatmap(corr_real['corr'], annot=True, annot_kws={"fontsize": 8}, linewidths=.3, ax=axs[0], cbar=False, vmin=0, vmax=1, cmap='Blues', yticklabels=corr_real['corr'].index)
    g2.set_yticklabels(corr_real['corr'].index)
    axs[0].set_xticklabels(axs[0].get_xticklabels(), rotation=45, ha="right")
    axs[0].set_yticklabels(axs[0].get_yticklabels(), rotation=0)

    axs[1].set_title('Synth')
    g2 = sns.heatmap(corr_synth['corr'], annot=True, annot_kws={"fontsize": 8}, linewidths=.3, ax=axs[1], cbar=False, vmin=0, vmax=1, cmap='Blues')
    # g2.set_yticklabels([])
    axs[1].set_xticklabels(axs[1].get_xticklabels(), rotation=45, ha="right")
    axs[1].set_yticklabels(axs[1].get_yticklabels(), rotation=0)

    diff = pd.Series.abs(corr_real['corr'] - corr_synth['corr'])
    axs[2].set_title('Diff - Norm Diff : {}'.format(norm_diff))
    axs[2].set_title('Différence des Matrices - Norme  : {:.2f}\n Correlation Spearman : {:.2f}'.format(norm_diff, spearman))
    g2 = sns.heatmap(diff, annot_kws={"fontsize": 8}, linewidths=.3, ax=axs[2], cbar=True, vmin=0, vmax=1, cmap='Blues')
    # g2.set_yticklabels([])
    axs[2].set_xticklabels(axs[2].get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    plt.close(fig)  # Close the figure to avoid displaying it directly in the notebook
    return fig


def plot_correlation_diff_(corr_real, corr_synth, norm_diff, spearman):
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 3.5), sharey=True)

    cors = corr_real.iloc[1:, 0:-1]
    cors_mask = np.triu(np.ones_like(cors, dtype=bool)) - np.identity(len(cors))
    axs[0].set_title('Données Réelles')
    sns.heatmap(cors, annot=True, annot_kws={"fontsize": 8}, linewidths=.3, ax=axs[0], mask=cors_mask, cbar=False, vmin=0, vmax=1, cmap='Blues')
    axs[0].set_xticklabels(axs[0].get_xticklabels(), rotation=45, ha="right")

    cors = corr_synth.iloc[1:, 0:-1]
    cors_mask = np.triu(np.ones_like(cors, dtype=bool)) - np.identity(len(cors))
    axs[1].set_title('Données Synthétiques')
    g2 = sns.heatmap(cors, annot=True, annot_kws={"fontsize": 8}, linewidths=.3, ax=axs[1], mask=cors_mask, cbar=False, vmin=0, vmax=1, cmap='Blues')
    g2.set(yticklabels=[])
    axs[1].set_xticklabels(axs[1].get_xticklabels(), rotation=45, ha="right")

    diff = pd.Series.abs(corr_real - corr_synth)
    cors = diff.iloc[1:, 0:-1]
    cors_mask = np.triu(np.ones_like(cors, dtype=bool)) - np.identity(len(cors))
    axs[2].set_title('Différence des Matrices - Norme  : {:.2f}\n Correlation Spearman : {:.2f}'.format(norm_diff, spearman))
    g2 = sns.heatmap(cors, annot=True, annot_kws={"fontsize": 8}, linewidths=.3, ax=axs[2], mask=cors_mask, cbar=True, vmin=0, vmax=1, cmap='Blues')
    g2.set(yticklabels=[])
    axs[2].set_xticklabels(axs[2].get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    plt.close(fig)  # Close the figure to avoid displaying it directly in the notebook
    return fig


def box_plot_results(results, name, x_labels) :
    
    fig = plt.figure()
    fig.suptitle(name)
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(x_labels)
    plt.show()

def histo_plot_utility_compare(utilities: dict, score: str) -> Image:
    """
    Generates a bar plot comparing F1 scores of different models across multiple datasets.

    This function creates a bar plot that compares the F1 scores of models across different datasets. 
    It merges the data from the `utilities` dictionary, where each key represents a dataset, and the 
    values are DataFrames containing F1 scores and model names. The function plots the F1 scores for each 
    model and highlights the differences between datasets.

    Args:
        utilities (dict): A dictionary where the keys are dataset names and the values are DataFrames. 
            Each DataFrame contains F1 scores for different models and is indexed by model name.

    Returns:
        Image: A PNG image object displaying the bar plot of F1 score comparisons for the models across datasets.

    Example:
        image = histo_plot_utility_compare(utilities)
        This function generates a bar plot that compares the F1 scores of various models across different datasets.

    Notes:
        - Each dataset in the `utilities` dictionary is added as a new category in the plot, 
          and F1 scores are plotted for each model.
        - The `utilities` DataFrames should have F1 scores as a column and models as the index.
        - The bar plot uses Seaborn for visualization, and the plot is rotated for better readability of model names.
        - The resulting plot is saved as a PNG image in memory and returned as an `Image` object for display.
    """

    for utility_key in utilities:
        utilities[utility_key]['Model'] = utilities[utility_key].index
        utilities[utility_key]['data'] = utility_key
    
    plt.figure(figsize=(8, 6)) 
    sns.barplot(y=score, x='Model', hue='data', data=pd.concat([df for df in utilities.values()]))
    plt.xticks(rotation=45)
    plt.legend(loc='lower right')
    plt.xlabel('Model')
    plt.ylabel('F1 Score')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    return Image(value=buf.read(), format='png', width=600, height=400)




def accuracy_compare(utilities) :
    acc_dict = {}

    for utility_key in utilities :
        key = utility_key + "_acc"
        acc_dict[key] = utilities[utility_key]['accuracy'].to_list()

    return pd.DataFrame.from_dict(acc_dict)