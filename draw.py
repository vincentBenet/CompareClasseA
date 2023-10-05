import matplotlib.pyplot as plt
import numpy
from matplotlib import colors


def main(scores, diff_xy, diff_z, reference, computed):

    nb = min(len(diff_xy), 100)

    ref_x, ref_y, ref_z = reference.T
    res_x, res_y, res_z = computed.T

    mid_x = numpy.median(ref_x)
    mid_y = numpy.median(ref_y)

    fig, axs = plt.subplots(2, 2)

    axs[0][0].plot(ref_x - mid_x, ref_y - mid_y, color="black")
    axs[0][0].scatter(res_x[1:] - mid_x, res_y[1:] - mid_y, c=diff_xy, cmap=plt.cm.turbo)

    n_xy, bins_xy, patches_xy = axs[1][0].hist(diff_xy, bins=nb)
    fracs = numpy.array([v for v in abs(bins_xy[:-1])])
    for thisfrac, thispatch in zip(fracs, patches_xy):
        thispatch.set_facecolor(plt.cm.turbo(colors.Normalize(fracs.min(), fracs.max())(thisfrac)))

    n_z, bins_z, patches_z = axs[1][1].hist(diff_z, bins=nb)
    fracs = numpy.array([v for v in abs(bins_z[:-1])])
    for thisfrac, thispatch in zip(fracs, patches_z):
        thispatch.set_facecolor(plt.cm.turbo(colors.Normalize(fracs.min(), fracs.max())(thisfrac)))

    plt.show()
