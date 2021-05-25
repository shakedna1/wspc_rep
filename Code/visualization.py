import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
from matplotlib.ticker import PercentFormatter


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
                            x_ticks_font_size=20, legend_loc='upper right'):

    fig, ax = plt.subplots(figsize=fig_size)

    rects = []
    for i, bar in enumerate(bars):
        vals = [results[bar][bar_group] for bar_group in bar_groups]
        bar_positions_in_group = [x + i * bar_width for x in np.arange(len(bar_groups))]
        rect = ax.bar(bar_positions_in_group, vals, bar_width, label=bar, align='edge')
        rects.append(rect)

    ax.set_ylabel(y_label, fontsize=20)
    center_x = bar_width * len(bars) / 2
    ax.set_xticks([x + center_x for x in np.arange(len(bar_groups))])
    ax.set_xticklabels(bar_groups, fontsize=x_ticks_font_size)
    ax.set_ylim([0.1, 1.05])

    ax.legend(loc=legend_loc, prop=FontProperties().set_size('xx-large'))

    label_grouped_bars(rects, ax)

    fig.tight_layout()

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    if out_path:
        plt.savefig(out_path)


def create_barplots(results, width, x_label, y_label, out_path, round_by=2, bar_label_font_size=14):

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.set_xticks(list(results.keys()))
    ax.bar(*zip(*results.items()), width=width)
    ax.set_ylim([0.87, 0.905])
    ax.set_ylabel(y_label, fontsize=18)
    ax.set_xlabel(x_label, fontsize=12)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    label_bars(ax, round_by, bar_label_font_size)

    plt.tight_layout()
    plt.savefig(out_path)


def create_correlation_plot(corr_dataset1, corr_dataset2, dataset_names, out_path):

    fig, ax = plt.subplots(figsize=(8, 4))

    bins = 10
    _, _, patches = plt.hist([corr_dataset1, corr_dataset2], bins=bins, density=True, label=dataset_names)

    # set heights of bars relatively to the total bar heights
    for patch in patches:
        sum_items = sum(item.get_height() * item.get_width() for item in patch)
        for bar in patch:
            v = bar.get_height() * bar.get_width()
            bar.set_height(v / sum_items)

    # Plot formatting
    plt.legend(loc='upper left')
    plt.xlabel('Correlation')
    plt.ylabel('Percentage')
    plt.xlim(0, 1)
    plt.ylim(0, 0.4)

    x_ticks = np.arange(0.05, 1, 0.1)
    plt.xticks(x_ticks)

    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path)