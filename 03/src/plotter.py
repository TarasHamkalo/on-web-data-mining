from typing import List

import math
from collections import Counter
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator, LogFormatter, LogLocator


PLOTTER_CONFIG = {
    'rcParams': {
        'font.size': 14,
        'axes.titlesize': 14,
        'axes.labelsize': 14,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
    },
    'style': 'seaborn-v0_8-pastel'
}

class Plotter:
    def _plot_around(
        self,
        handler,
        size=(20, 6),
        y_major_locator=5,
        x_major_locator=5,
        x_minor_locator=True,
        output_file: str | Path | None = None,
    ):
        plt.rcParams.update(PLOTTER_CONFIG['rcParams'])
        plt.style.use(PLOTTER_CONFIG['style'])

        fig, ax = plt.subplots(figsize=size, dpi=150)

        title, x_label, y_label = handler(ax)

        if y_major_locator is not None:
            ax.yaxis.set_major_locator(MultipleLocator(y_major_locator))

        if x_major_locator is not None:
            ax.xaxis.set_major_locator(MultipleLocator(x_major_locator))

        if x_minor_locator:
            ax.xaxis.set_minor_locator(MultipleLocator(5))

        ax.set_facecolor((0, 0, 0, 0))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend()

        plt.grid(False)

        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        plt.tight_layout()

        if output_file is not None:
            output_file = Path(output_file)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            fig.savefig(
                output_file,
                dpi=150,
                bbox_inches="tight",
                transparent=False,
            )
            plt.close(fig)
        else:
            plt.show()



    def plot_histogram_bin_count(
        self,
        sequence: List[int],
        bins_count,
        size=(20, 6),
        x_majors=10,
        y_majors=10,
        left=None,
        right=None,
        log_y_scale=False,
        title="",
        x_label="",
        y_label="",
        label="",
    ):

        def handler(ax):
            local_y_label = y_label

            counts = np.array(sequence)
            counts = np.where(counts == 0, 1e-10, counts)
            if right is not None and left is not None:
                start, end = left, right
            else:
                start = 0
                end = len(counts)

            dx = end - start

            step = math.ceil(dx / bins_count)
            print(start, end, step)

            bins = np.arange(start, end + step, step)
            bins[-1] = end + 1e-10

            counts, bin_edges, patches = ax.hist(
                counts,
                bins=bins,
                alpha=0.7,
                linewidth=0.9,
                edgecolor="black",
                label=label,
            )

            if log_y_scale:
                ax.set_yscale("log", nonpositive="clip")
                ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0,)))
                ax.yaxis.set_major_formatter(LogFormatter())
                ax.set_ylim(bottom=1)
                local_y_label = y_label + "(log scale)"
            else:
                ax.yaxis.set_major_locator(MultipleLocator(counts.max() // y_majors))

            ax.xaxis.set_major_locator(MultipleLocator(dx // x_majors))
            ax.set_xlim(left=left, right=bins[-1] + bins[-1] * 0.01)
            return title, x_label, local_y_label

        self._plot_around(
            handler, size, y_major_locator=None, x_major_locator=None, x_minor_locator=False
        )
