import matplotlib.pyplot as plt
import numpy
from matplotlib import colors
import matplotlib.patheffects as pe


def main(scores, diff_xy, diff_z, reference, res, ref_name, res_name):

    nb = min(len(diff_xy), 50)

    ref_x, ref_y, ref_z, ref_abc = reference.T
    res_x, res_y, res_z, res_abc = res.T

    indexes = numpy.around(numpy.linspace(0, len(res), 10))

    mid_x = numpy.median(ref_x)
    mid_y = numpy.median(ref_y)

    fig, axs = plt.subplots(2, 2)

    axs[0][0].plot(ref_x - mid_x, ref_y - mid_y, color="black")

    axs[0][0].fill_between(
        ref_x - mid_x,
        ref_y - mid_y - 1.5,
        ref_y - mid_y + 1.5,
        color="red",
        interpolate=True,
        alpha=1,
        # edgecolor="white",
        label=f"XY 100% = {(scores['XY 100%'] * 100):.2f} / 150 cm ({(scores['XY 100% 150cm']*100):.2f} %)",
    )

    fig.legend().get_texts()[-1].set_color(["red", "green"][scores['XY 100% 150cm'] > 0])

    axs[0][0].fill_between(
        ref_x - mid_x,
        ref_y - mid_y - 0.4,
        ref_y - mid_y + 0.4,
        color="blue",
        interpolate=True,
        alpha=1,
        # edgecolor="white",
        label=f"XY 90%  = {(scores['XY 90%'] * 100):.2f} /  40 cm ({(scores['XY 90% 40cm']*100):.2f} %)",
    )
    axs[0][0].fill_between(
        ref_x - mid_x,
        ref_y - mid_y - 0.2,
        ref_y - mid_y + 0.2,
        color="green",
        interpolate=True,
        alpha=1,
        # edgecolor="white",
        label=f"XY 60%  = {(scores['XY 60%'] * 100):.2f} /  20 cm ({(scores['XY 60% 20cm']*100):.2f} %)",
    )

    axs[0][0].scatter(
        res_x - mid_x,
        res_y - mid_y,
        c=diff_xy,
        cmap=plt.cm.turbo,
        edgecolor="white"
    )

    for i, point in enumerate(numpy.array([res_x, res_y]).T):
        if i not in indexes:
            continue
        distance = diff_xy[i]
        x, y = point
        txt = f"{i} : {(distance * 100):.2f} cm"
        axs[0][0].annotate(
            text=txt,
            xy=(x - mid_x, y - mid_y),
            xytext=(5, 0),
            textcoords='offset pixels',
            path_effects=[pe.withStroke(linewidth=4, foreground="white")]
        )

    axs[1][0].axvspan(0, 1.5, color="red", alpha=1)
    axs[1][0].axvline(scores["XY 100%"], color="red", linewidth=1, path_effects=[pe.withStroke(linewidth=3, foreground="white")])
    axs[1][0].axvspan(0, 0.4, color="blue", alpha=1)
    axs[1][0].axvline(scores["XY 90%"], color="blue", linewidth=1, path_effects=[pe.withStroke(linewidth=3, foreground="white")])
    axs[1][0].axvspan(0, 0.2, color="green", alpha=1)
    axs[1][0].axvline(scores["XY 60%"], color="green", linewidth=1, path_effects=[pe.withStroke(linewidth=3, foreground="white")])

    n_xy, bins_xy, patches_xy = axs[1][0].hist(diff_xy, bins=nb, edgecolor='black')
    fracs = numpy.array([v for v in abs(bins_xy[:-1])])
    for thisfrac, thispatch in zip(fracs, patches_xy):
        thispatch.set_facecolor(plt.cm.turbo(colors.Normalize(fracs.min(), fracs.max())(thisfrac)))

    axs[1][1].axvspan(0, 0.7, color="red", alpha=1)
    axs[1][1].axvline(scores["Z 100%"], color="red", linewidth=1, path_effects=[pe.withStroke(linewidth=3, foreground="white")])
    axs[1][1].axvspan(0, 0.4, color="blue", alpha=1)
    axs[1][1].axvline(scores["Z 90%"], color="blue", linewidth=1, path_effects=[pe.withStroke(linewidth=3, foreground="white")])

    axs[0][1].plot(ref_abc, ref_z, color="black")
    axs[0][1].fill_between(
        ref_abc,
        ref_z - 0.7,
        ref_z + 0.7,
        color="red",
        interpolate=True,
        alpha=1,
        # edgecolor="white",
        label=f"Z 100%  = {(scores['Z 100%'] * 100):.2f} /  70 cm ({(scores['Z 100% 70cm']*100):.2f} %)",
    )
    axs[0][1].fill_between(
        ref_abc,
        ref_z - 0.4,
        ref_z + 0.4,
        color="blue",
        interpolate=True,
        alpha=1,
        # edgecolor="white",
        label=f"Z 90%   = {(scores['Z 90%'] * 100):.2f} /  40 cm ({(scores['Z 90% 40cm']*100):.2f} %)",
    )

    n_z, bins_z, patches_z = axs[1][1].hist(diff_z, bins=nb, edgecolor='black')
    fracs = numpy.array([v for v in abs(bins_z[:-1])])
    for thisfrac, thispatch in zip(fracs, patches_z):
        thispatch.set_facecolor(plt.cm.turbo(colors.Normalize(fracs.min(), fracs.max())(thisfrac)))

    axs[0][1].scatter(
        res_abc,
        res_z,
        c=diff_z,
        cmap=plt.cm.turbo,
        edgecolor="white"
    )
    for i, point in enumerate(numpy.array([res_abc, res_z]).T):
        if i not in indexes:
            continue
        distance = diff_z[i]
        a, z = point
        txt = f"{i} : {(distance * 100):.2f} cm"
        axs[0][1].annotate(
            text=txt,
            xy=(a, z),
            xytext=(0, 10),
            ha='left',
            rotation=90,
            textcoords='offset pixels',
            path_effects=[pe.withStroke(linewidth=4, foreground="white")]
        )

    axs[1][0].set_xlim(left=0, right=max(diff_xy))
    axs[1][1].set_xlim(left=0, right=max(diff_z))
    axs[0][0].set_title('XY View')
    axs[0][0].set_xlabel('Coordinates X (meter)')
    axs[0][0].set_ylabel('Coordinates Y (meter)')
    axs[1][0].set_title(f'XY Distances : Score = {scores["percent_xy"]:.2f} %', color=["red", "green"][scores['ok_xy']])
    axs[1][0].set_xlabel('Distance XY (m)')
    axs[1][0].set_ylabel('Population')
    axs[0][1].set_title('Z View')
    axs[1][0].set_xlabel('Curvilign Abcisse (m)')
    axs[1][0].set_ylabel('Elevation (m)')
    axs[1][1].set_title(f'Z Distances : Score = {scores["percent_z"]:.2f} %', color=["red", "green"][scores['ok_z']])
    axs[1][0].set_xlabel('Distance XY (m)')
    axs[1][0].set_ylabel('Population')

    txt = f"Compare layer <{res_name}> to reference <{ref_name}>\n"
    if not scores['ok']:
        txt += f"OUT OF CLASS A\n"
    txt += f"Score = {scores['percent']:.2f} %"
    fig.suptitle(txt, fontsize=16, color=["red", "green"][scores['ok']])

    fig.legend(loc='best')
    plt.show()
