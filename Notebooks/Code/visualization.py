import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
from matplotlib.ticker import PercentFormatter

GRAY_COLOR = '#333F4B'
plt.rcParams['axes.edgecolor'] = GRAY_COLOR
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['xtick.color'] = GRAY_COLOR
plt.rcParams['ytick.color'] = GRAY_COLOR


def label_bar(ax, bar, round_by=2, bar_label_font_size=14):

    height = round(bar.get_height(), round_by)
    ax.annotate('{}'.format(height), xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=bar_label_font_size)


def label_grouped_bars(rects, ax):
    for rect in rects:
        for bar in rect:
            label_bar(ax, bar)


def label_bars(ax, round_by, bar_label_font_size):

    for bar in ax.patches:
        label_bar(ax, bar, round_by, bar_label_font_size)


def create_grouped_barplots(results, bar_groups, bars, out_path, bar_width=0.15, fig_size=(14, 6), y_label='Scores',
                            x_ticks_font_size=20, legend_loc='upper right', y_lim=[0.1, 1.05], save=True):

    fig, ax = plt.subplots(figsize=fig_size)

    rects = []
    for i, bar in enumerate(bars):
        vals = [results[bar][bar_group] for bar_group in bar_groups]
        bar_positions_in_group = [x + i * bar_width for x in np.arange(len(bar_groups))]
        rect = ax.bar(bar_positions_in_group, vals, bar_width, label=bar, align='edge', alpha=0.6)
        rects.append(rect)

    ax.set_ylabel(y_label, fontsize=20)
    center_x = bar_width * len(bars) / 2
    ax.set_xticks([x + center_x for x in np.arange(len(bar_groups))])
    ax.set_xticklabels(bar_groups, fontsize=x_ticks_font_size, color='black')
    ax.set_ylim(y_lim)
    ax.tick_params(axis='y', which='major', labelsize=14)

    ax.legend(loc=legend_loc, fontsize=14)

    label_grouped_bars(rects, ax)

    fig.tight_layout()

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    if save:
        plt.savefig(out_path)


def create_barplots(results, width, x_label, y_label, out_path, round_by=2, bar_label_font_size=14, figsize=(10, 6),
                    y_lim=[0.87, 0.905], save=True):

    fig, ax = plt.subplots(figsize=figsize)

    ax.set_xticks(list(results.keys()))
    ax.bar(*zip(*results.items()), width=width, alpha=0.4)
    ax.set_ylim(y_lim)
    ax.set_ylabel(y_label, fontsize=18)
    ax.set_xlabel(x_label, fontsize=18)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.tick_params(axis='both', which='major', labelsize=14)

    label_bars(ax, round_by, bar_label_font_size)

    plt.tight_layout()

    if save:
        plt.savefig(out_path)


def create_correlation_plot(corr_dataset1, corr_dataset2, dataset_names, out_path, fig_size=(9, 6), save=True):

    fig, ax = plt.subplots(figsize=fig_size)

    bins = 10
    _, _, patches = plt.hist([corr_dataset1, corr_dataset2], bins=bins, density=True, label=dataset_names, alpha=0.6)

    # set heights of bars relatively to the total bar heights
    for patch in patches:
        sum_items = sum(item.get_height() * item.get_width() for item in patch)
        for bar in patch:
            v = bar.get_height() * bar.get_width()
            bar.set_height(v / sum_items)

    # Plot formatting
    plt.legend(loc='upper left', fontsize=14)

    ax.set_ylabel('Percentage', fontsize=18)
    ax.set_xlabel('Correlation', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=14)

    plt.xlim(0, 1)
    plt.ylim(0, 0.4)

    x_ticks = np.arange(0.05, 1, 0.1)
    plt.xticks(x_ticks)

    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.tight_layout()

    if save:
        plt.savefig(out_path)