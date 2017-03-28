#!/usr/bin/env python

import json
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
from matplotlib.ticker import MultipleLocator
import matplotlib.transforms
import seaborn as sns
import sys
import os
import re

# rc("font", **{
#     "family": "monospace",
# })
plt.rc("text", usetex=True)
plt.rcParams["text.latex.preamble"] = [
    r"\usepackage{siunitx}",
    r"\sisetup{detect-all}",
]
sns.set_style("ticks")
sns.set_context("poster")
sns.set_palette([
    (202 / 256, 51 / 256, 0),
    (49 / 256, 99 / 256, 206 / 256),
    (255 / 256, 155 / 256, 0)
])

class SubplotManager(object):
    def __init__(self, rows, cols, magic_func=lambda x: x):
        self.rows = rows
        self.cols = cols
        self.n = 1
        self.magic_func = magic_func

    def __call__(self):
        x = self.rows, self.cols, self.n
        x = self.magic_func(x)
        self.n += 1
        return x


class Datum(object):
    def __init__(self, data, time):
        self.dilution = data[-1]
        self.CFU = data[:-1]
        self.time = time

    def cfu(self):
        return ((np.mean(self.CFU) * (10 ** self.dilution)) / 20) * 1000

    def std(self):
        return ((np.std(self.CFU) * (10 ** self.dilution)) / 20) * 1000

    def __repr__(self):
        return repr(self.__dict__)


class Data(object):
    def __init__(self, data, timings):
        timings = list(timings)
        self.tp = []
        for x in data:
            if x:
                t = timings.pop(0)
            else:
                timings.pop(0)
                continue
            self.tp.append(
                Datum(x, t)
            )

    def timepoints(self):
        return np.array([(x.time, x.cfu(), x.std()) for x in self.tp])

    def get_timepoint(self, t):
        for x in self.tp:
            if x.time == t:
                return x


class DataManager(object):
    def __init__(self, data, timings):
        self.description = data.pop()
        self.negative = Data(data[0], timings)
        self.positive = Data(data[1], timings)


class GrowthCurve(object):
    def stripcomments(self, fn):
        regex = re.compile("\<\+(.*?)\+\>")
        f = open(fn).read()
        s = ""
        for line in f.split("\n"):
            if not line:
                pass
            elif line.strip()[0] == "#":
                pass
            else:
                if regex.search(line):
#                    for m in regex.finditer(line):
                    key = regex.search(line).group(1)
                    if key == "last":
                        prev_line = s.split("\n")[-2]
                        if prev_line.strip().endswith(","):
                            replacement = prev_line.strip()[:-1]
                        else:
                            replacement = prev_line
                    else:
                        replacement = "null"

                    if line.strip().endswith(","):
                        s += "{0},\n".format(replacement)
                    else:
                        s += "{0}\n".format(replacement)
                else:
                    s += "{0}\n".format(line)
        return s

    def __init__(self, data="data.json"):
        self.INIT = False

        self.data = json.loads(self.stripcomments(data))
        self.timings = [
            time.mktime(
                time.strptime("{0} {1}".format(
                    a, b
                ), "%d.%m.%y %H:%M")
            ) for a, b in self.data["times"]
        ]

        self.start_time = self.timings[0]
        self.timings = [
            (x - self.start_time) / 60 / 60 for x in self.timings
        ]

        self.add_time = (time.mktime(
            time.strptime("{0} {1}".format(
                *self.data["add_date"]
            ), "%d.%m.%y %H:%M")
        ) - self.start_time) / 60 / 60

        self.remove_time = (time.mktime(
            time.strptime("{0} {1}".format(
                *self.data["remove_date"]
            ), "%d.%m.%y %H:%M")
        ) - self.start_time) / 60 / 60

        self.counts = self.data["data"]
        self.order = self.data["order"]
        self.resources = [
            DataManager(
                self.counts[x],
                self.timings,
            ) for x in self.order
        ]
        self.resource_count = len(self.resources)

    def initialise(self, resource_add=0, SP=None, magic_func=lambda x: x, label=None):
        if not SP:
            resource_count = self.resource_count + resource_add
            if resource_count < 2:
                cols = 1
            else:
                cols = 2
            rows = resource_count // 2
            rem = resource_count % 2
            if rem > 0:
                rows += 1

            self.SP = SubplotManager(rows, cols, magic_func=magic_func)
        else:
            self.INIT = True
            self.SP = SP

        self.label = label
        self.survival = []
        self.survival_min = []
        self.survival_max = []
        self.survival_points = []
        for x in self.resources:
            self.plot_x(x)

    def plot_x(self, x):
        if not self.INIT:
            plt.figure(figsize=(16, 12))
            sp = self.SP()
            ax1 = plt.subplot(*sp)
        else:
            prev_ax = plt.gca()
            sp = self.SP()
            ax1 = plt.subplot(*sp, sharex=prev_ax, sharey=prev_ax)
        self.INIT = True

        ax1.spines["right"].set_color("none")
        ax1.spines["top"].set_color("none")
        ax1.xaxis.set_ticks_position("bottom")
        ax1.yaxis.set_ticks_position("left")
        ax1.xaxis.set_minor_locator(MultipleLocator(2))

        tp_m = x.negative.timepoints()
        if tp_m.any():
            if self.label:
                label = "{0} -RIF".format(self.label)
            else:
                label = "{0} -RIF".format(x.description)
            plt.errorbar(tp_m[:, 0], tp_m[:, 1], yerr=tp_m[:, 2], label=label)

        ref_delta = self.add_time
        rif_delta = self.remove_time - self.add_time

        tp = x.positive.timepoints()
        if tp.any():
            if self.label:
                label = "{0} +RIF".format(self.label)
            else:
                label = "{0} +RIF".format(x.description)
            plt.errorbar(tp[:, 0], tp[:, 1], yerr=tp[:, 2], label=label)

        plt.yscale("log")
        # plt.ylim([10**6, 10**10])
        # plt.ylim([10**2, 10**10])
        # plt.xlim([0, 73])

        plt.ylabel("CFU (ml$^{-1}$)")
#        plt.legend(
#            l[:2], leg[:2], loc=3, numpoints=1
#        )
#
        plt.legend(loc="lower left", numpoints=1)
        plt.xlabel("Time (h)")

        trans = matplotlib.transforms.blended_transform_factory(
            ax1.transData, ax1.transAxes
        )
        rect = matplotlib.patches.Rectangle(
            (ref_delta, 0),
            rif_delta,
            1,
            facecolor="y",
            edgecolor=None,
            alpha=.3,
            transform=trans
        )
        ax1.add_patch(rect)

        t0_index = np.argmin(np.abs([(_ - self.add_time) for _ in self.timings]))
#        t0_index = ([(_ < self.add_time) for _ in self.timings]).index(False)
        addition = 0
        while True:
            t0_time = self.timings[t0_index + addition]
            try:
                t0_cfu = x.positive.get_timepoint(t0_time).cfu()
                break
            except AttributeError:
                addition += 1

        try:
#            t1_index = ([(_ < self.remove_time) for _ in self.timings]).index(False) - 1
            t1_index = np.argmin(np.abs([(_ - self.remove_time) for _ in self.timings]))
        except ValueError:
            t1_index = len(self.timings) - 1
        t1_time = self.timings[t1_index]
        try:
            t1_cfu = x.positive.get_timepoint(t1_time).cfu()
        except AttributeError:
            try:
                t1_time = self.timings[t1_index + 1]
                t1_cfu = x.positive.get_timepoint(t1_time).cfu()
            except (IndexError, AttributeError):
                t1_subtract = 1
                while True:
                    try:
                        t1_time = self.timings[t1_index - t1_subtract]
                        t1_cfu = x.positive.get_timepoint(t1_time).cfu()
                    except AttributeError:
                        t1_subtract += 1
                    else:
                        break

        # determine lowest cfu value during RIF period
        t0_index = self.timings.index(t0_time)
        t1_index = self.timings.index(t1_time)
        positives = x.positive.tp[t0_index:t1_index+1]
        cfus = [_.cfu() for _ in positives]
        t2_index = cfus.index(min(cfus))
        t2_time = positives[t2_index].time
        t2_cfu = cfus[t2_index]

        all_positives = x.positive.tp[:t1_index + 1]
        all_positive_cfus = [_.cfu() for _ in all_positives]
        t3_index = all_positive_cfus.index(max(all_positive_cfus))
        t3_time = all_positives[t3_index].time
        t3_cfu = all_positive_cfus[t3_index]

        self.survival_points.append(
            (
                ax1,
                [t0_time, t0_cfu],
                {
                    "marker": "o",
                    "color": "r",
                    "markersize": 10,
                    "markeredgewidth": 2,
                    "fillstyle": "none"
                }
            )
        )

        self.survival_points.append(
            (
                ax1,
                [t1_time, t1_cfu],
                {
                    "marker": "o",
                    "color": "b",
                    "markersize": 10,
                    "markeredgewidth": 2,
                    "fillstyle": "none"
                }
            )
        )

        self.survival_points.append(
            (
                ax1,
                [t2_time, t2_cfu],
                {
                    "marker": "o",
                    "color": "r",
                    "markersize": 10,
                    "markeredgewidth": 2,
                    "fillstyle": "none"
                }
            )
        )

        self.survival_points.append(
            (
                ax1,
                [t3_time, t3_cfu],
                {
                    "marker": "o",
                    "color": "g",
                    "markersize": 10,
                    "markeredgewidth": 2,
                    "fillstyle": "none"
                }
            )
        )

        if self.label:
            self.survival.append(
                (self.label, t1_cfu / t0_cfu)
            )
        else:
            self.survival.append(
                (x.description, t1_cfu / t0_cfu)
            )
        self.survival_min.append(
            (t2_cfu / t0_cfu)
        )
        self.survival_max.append(
            (t2_cfu / t3_cfu)
        )

    def plot_survival(self, ax=None, circle=True, add=0, y1locadd=[], y2locadd=[], y3locadd=[]):
        if type(y1locadd) is not list:
            y1locadd = [y1locadd]
        if type(y2locadd) is not list:
            y2locadd = [y2locadd]
        if type(y3locadd) is not list:
            y3locadd = [y3locadd]

        if not ax:
            sp = self.SP()
            p = plt.subplot(*sp)
        else:
            p = ax
#            p = plt.subplot(*sp, sharey=ax)
        p.spines["right"].set_color("none")
        p.spines["top"].set_color("none")
        p.xaxis.set_ticks_position("bottom")
        p.yaxis.set_ticks_position("left")

        width = 0.35
        spacing = 0.5
        xlocs = (np.arange(len(self.resources)) / 2) + (add * spacing)
        ylocs = np.array([y for x, y in self.survival]) * 100
        y2locs = np.array(self.survival_min) * 100
        y3locs = np.array(self.survival_max) * 100

        if not y1locadd:
            y1locadd = [0] * len(ylocs)
        if not y2locadd:
            y2locadd = [0] * len(ylocs)
        if not y3locadd:
            y3locadd = [0] * len(ylocs)

        if circle:
            for sp_ax, args, kwargs in self.survival_points:
                sp_ax.plot(*args, **kwargs)

        plt.bar(
            xlocs,
            ylocs,
            width,
            color="y",
            edgecolor="k",
            alpha=.3,
        )
        plt.bar(
            xlocs,
            y2locs,
            width,
            color="y",
            edgecolor="k",
            alpha=.3
        )
        plt.bar(
            xlocs,
            y3locs,
            width,
            color="y",
            edgecolor="k",
            alpha=.3,
        )

        val_idx = 0
        for val in ylocs:
            plt.text(
                xlocs[val_idx] + (width / 2),
                ylocs[val_idx] + y1locadd[val_idx],
                r"\textbf{{{0:.3f}\%}}".format(ylocs[val_idx]),
                horizontalalignment="center",
                verticalalignment="bottom",
                color="blue",
            )
            plt.text(
                xlocs[val_idx] + (width / 2),
                y2locs[val_idx] + y2locadd[val_idx],
                r"\textbf{{{0:.3f}\%}}".format(y2locs[val_idx]),
                horizontalalignment="center",
                verticalalignment="bottom",
                color="red",
            )
            plt.text(
                xlocs[val_idx] + (width / 2),
                y3locs[val_idx] + y3locadd[val_idx],
                r"\textbf{{{0:.3f}\%}}".format(y3locs[val_idx]),
                horizontalalignment="center",
                verticalalignment="bottom",
                color="green",
            )
            val_idx += 1


#        self._val_idx = 0
#        def val_idx():
#            self._val_idx += 1
#            return self._val_idx - 1

        if add:
            xlocs_prior, xdata_prior = plt.xticks()
            xdata_prior = [_.get_text() for _ in xdata_prior]
            xlocs_post = np.append(
                xlocs_prior,
                xlocs + (width / 2)
            )
            xdata_post = np.append(
                xdata_prior,
                [x for x, y in self.survival]
            )
        else:
            xlocs_post = xlocs + width / 2
            xdata_post = [x for x, y in self.survival]

        plt.xticks(
            xlocs_post,
            xdata_post
        )
#        plt.xticks(
#            xlocs + width / 2 + (add * width),
#            [x for x, y in self.survival]
#            # [r"{0} (\textbf{{{1:.2f}\%}})".format(x, ylocs[val_idx()]) for x, y in self.survival]
#        )

        plt.xlim([0, 0.5 * 5])
        plt.ylabel("\% Survival")
        return plt.gca(), len(xlocs)

if __name__ == "__main__":
    try:
        fn = sys.argv[1]
        if not os.path.exists(fn):
            fn = "data.json"
    except:
        fn = "data.json"

    print("Generating plot for {0}".format(fn))
    try:
        out = sys.argv[2]
    except:
        out = "population-cfu.pdf"

    g = GrowthCurve(data=fn)
#    g2 = GrowthCurve("data-150310.json")
#    g3 = GrowthCurve("data-141114.json")

    def modifyorder(x):
        if x[2] == 2:
            return x[0], x[1], 3
        elif x[2] == 3:
            return x[0], x[1], 2
        else:
            return x

    g.initialise(
        resource_add=(
            # g2.resource_count +
            # g3.resource_count +
            1
        )
    )
#    g2.initialise(SP=g.SP, label="10/03 (N)")
#    g3.initialise(SP=g.SP, label="14/11")
    ax, add = g.plot_survival()
#    ax, add2 = g2.plot_survival(ax=ax, add=add)
#    ax, add3 = g3.plot_survival(ax=ax, add=add + add2)

#    g.initialise(
#        resource_add=g2.resource_count + g3.resource_count + 1,
#        magic_func=modifyorder
#    )
#    g2.initialise(SP=g.SP)
#    g3.initialise(SP=g2.SP)
#    # g.initialise(resource_add=1)  # add one resource if plotting survival
#    ax, add = g.plot_survival(y1locadd=0.1)
#    ax, add2 = g2.plot_survival(ax=ax, add=add, y1locadd=0.4)
#    ax, add3 = g3.plot_survival(
#        ax=ax, add=add+add2,
#        y1locadd=[0.4, 0]
#    )
    plt.tight_layout()
    print("Saving figure to {0}".format(out))
    plt.savefig(out)
    plt.close()

    fig = plt.figure(figsize=(9.42, 4.71))
    fig.patch.set_facecolor("none")
    fig.patch.set_alpha(0)

    ax1 = fig.add_subplot(121)
    sns.despine()
    #ax1.set_title("With RIF")
    ax1.set_xlabel("Time (h)")
    ax1.set_ylabel("CFU (ml$^{-1}$)")
    ax1.set_yscale("log")
    ax1.patch.set_facecolor("none")
    ax1.patch.set_alpha(0)

    #ax2 = fig.add_subplot(122, sharex=ax1, sharey=ax1)
    ax2 = fig.add_subplot(122)
    sns.despine()
    #ax2.set_title("No RIF")
    ax2.set_xlabel("Time (h)")
    ax2.set_ylabel("CFU (ml$^{-1}$)")
    ax2.set_yscale("log")
    ax2.patch.set_facecolor("none")
    ax2.patch.set_alpha(0)

    c = sns.color_palette()
    for x in g.resources:
        print(x.description)
        if x.description == "HdB/pyruvate":
            colour = c[2]
        elif x.description == "HdB/acetate":
            colour = c[1]
        elif x.description == "HdB/glycerol":
            colour = c[0]

        pos = x.positive.timepoints()
        neg = x.negative.timepoints()
        ax1.plot(pos[:, 0], pos[:, 1], label=x.description, color=colour)
        ax2.errorbar(
            neg[:, 0], neg[:, 1], neg[:, 2], label=x.description, color=colour,
            marker=".",
            ms=8,
            ls="-",
            lw=2,
        )

#        import code
#        print("label:", x.description)
#        variables = {
#            "data": neg,
#            "plt": plt,
#            "np": np,
#        }
#        code.interact(local=variables)


#    ax1.legend(loc="lower left")
#    ax2.legend(loc="lower left")

#    ax1.set_xlim([0, 80])
#    ax2.set_xlim([0, 80])
#    ax1.set_ylim([10**3, 10**9])
#    ax2.set_ylim([10**3, 10**9])

    fig.tight_layout()
    fn = "traces.pdf"
    print("Saving figure to {0}".format(fn))
    fig.savefig(fn, facecolor="none", edgecolor="none", transparent=True)

    fn = "traces-rif.pdf"
    extent = matplotlib.transforms.Bbox([[0, 0], [4.71, 4.71]])
    fig.savefig(fn, facecolor="none", edgecolor="none", transparent=True, bbox_inches=extent)

    fn = "traces-no-rif.pdf"
    extent = matplotlib.transforms.Bbox([[4.71, 0], [9.42, 4.71]])
    fig.savefig(fn, facecolor="none", edgecolor="none", transparent=True, bbox_inches=extent)
