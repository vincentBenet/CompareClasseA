import numpy
from scipy import spatial
import numpy as np


def compare(ref, data):
    *_, distances = compare_curves(ref, data)
    dists_xy = distances["xy"]
    dists_z = distances["axis"]
    if not len(distances):
        return 100000000
    xy60 = numpy.percentile(dists_xy, 60)
    xy90 = numpy.percentile(dists_xy, 90)
    xy100 = numpy.percentile(dists_xy, 100)
    z90 = numpy.percentile(dists_z, 90)
    z100 = numpy.percentile(dists_z, 100)
    xy60p20cm = (0.2 - xy60) / 0.2
    xy90p40cm = (0.4 - xy90) / 0.4
    xy100p150cm = (1.5 - xy100) / 1.5
    z90p40cm = (0.4 - z90) / 0.4
    z100p70cm = (0.7 - z100) / 0.7
    score = 5 - (xy60p20cm + xy90p40cm + xy100p150cm + z90p40cm + z100p70cm)
    return {
        "xy60": xy60,
        "xy90": xy90,
        "xy100": xy100,
        "z90": z90,
        "z100": z100,
        "xy60p20cm": xy60p20cm,
        "xy90p40cm": xy90p40cm,
        "xy100p150cm": xy100p150cm,
        "z90p40cm": z90p40cm,
        "z100p70cm": z100p70cm,
        "score": score,
    }


def compare_curves(
    reference,
    computed,
    axis="z",
    step=0.01
):
    abs_cur = np.insert(np.cumsum(np.nansum((reference[:-1] - reference[1:]) ** 2, axis=1) ** 0.5), 0, 0)
    interp_abs_cur = np.linspace(abs_cur[0], abs_cur[-1], int(abs_cur[-1] / step) + 1)
    interp_reference = np.empty((interp_abs_cur.size, reference.shape[1]))
    for i in range(reference.shape[1]):
        interp_reference[:, i] = np.interp(interp_abs_cur, abs_cur, reference[:, i])
    abs_cur_interp_ref = np.insert(np.cumsum(np.nansum((interp_reference[:-1] - interp_reference[1:]) ** 2, axis=1) ** 0.5), 0, 0)
    tree = spatial.KDTree(interp_reference[..., :2])
    _, index_original = tree.query(reference[..., :2], k=1)
    min_dist, index_min_dist = tree.query(computed[..., :2], k=1)
    axis_index = -2 if axis == "z" else -1
    axis_min_dist = interp_reference[index_min_dist, axis_index]
    dist_axis = np.abs(computed[..., axis_index] - axis_min_dist)
    mask = ~np.logical_or(index_min_dist == 0, index_min_dist == len(interp_reference) - 1)
    return (
        np.fromiter(
            zip(
                reference[..., 0],
                reference[..., 1],
                reference[..., axis_index],
                abs_cur_interp_ref[index_original],
                strict=True,
            ),
            dtype=[
                ("x", np.float_),
                ("y", np.float_),
                ("axis", np.float_),
                ("curvilinear_abs", np.float_),
            ],
        ),
        np.fromiter(
            zip(
                computed[mask, 0],
                computed[mask, 1],
                computed[mask, axis_index],
                abs_cur_interp_ref[index_min_dist][mask],
                strict=True,
            ),
            dtype=[
                ("x", np.float_),
                ("y", np.float_),
                ("axis", np.float_),
                ("curvilinear_abs", np.float_),
            ],
        ),
        np.fromiter(zip(min_dist[mask], dist_axis[mask], strict=True), dtype=[
            ("xy", np.float_),
            ("axis", np.float_),
        ]),
    )
