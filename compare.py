import math
import numpy
import geopandas
from pathlib import Path
from scipy import spatial


def coordinates_from_gis(gis_file_paths):
    res = []
    epsg = ""
    for i, path in enumerate(gis_file_paths):
        data = geopandas.read_file(path)
        epsg_from = str(data.geometry.crs)
        if i == 0:
            epsg = epsg_from
        elif epsg_from != epsg:
            raise Exception(f"Files must be in same EPSG")
        if epsg_from == "EPSG:4326":
            raise Exception(f"Guess GPKG must be in projected EPSG")
        print(f"{epsg_from} - {path}")
        x = data.geometry.x
        y = data.geometry.y
        pipe = numpy.array([x, y, data.geometry.z]).T
        res_pipe = []
        for point in pipe:
            if not (
                math.isinf(point[0]) or
                math.isinf(point[1]) or
                math.isinf(point[2]) or
                numpy.isnan(point[0]) or
                numpy.isnan(point[1]) or
                numpy.isnan(point[2])
            ):
                res_pipe.append(point)
        if not len(res_pipe):
            raise Exception(f"Empty 3D geometry for file {path}")
        res.append(numpy.array(res_pipe))
        print(f"\tLoading {path = }: {len(res_pipe)} points")
    return res


def compare_gis_files(
    reference_path: Path,
    computed_path: Path,
):
    computed, reference = coordinates_from_gis([computed_path, reference_path])
    if computed.shape[0] == 0 or reference.shape[0] == 0:
        return None
    dists_xy, dists_z = compare_curves(reference, computed)
    scores = compare_pipe_to_pipe(dists_xy, dists_z)
    print(scores)
    return scores, dists_xy, dists_z, reference, computed


def compare_pipe_to_pipe(dists_xy, dists_z):
    if not len(dists_xy):
        print(f"Exit because not len(dists_XY)")
    xy60 = numpy.percentile(dists_xy, 60)
    xy90 = numpy.percentile(dists_xy, 90)
    xy100 = numpy.percentile(dists_xy, 100)
    z90 = numpy.percentile(dists_z, 90)
    z100 = numpy.percentile(dists_z, 100)
    return {
        "XY 60%": xy60,  # Percentile 60% XY
        "XY 90%": xy90,  # Percentile 90% XY
        "XY 100%": xy100,  # Percentile 100% XY
        "Z 90%": z90,  # Percentile 90% Z
        "Z 100%": z100,  # Percentile 100% Z
        "XY 60% 20cm": (0.2 - xy60) / 0.2,  # Critere à 60% XY 20cm, positif si classe A, [-inf, 1]
        "XY 90% 40cm": (0.4 - xy90) / 0.4,  # Critere à 90% XY 40cm, positif si classe A, [-inf, 1]
        "XY 100% 150cm": (1.5 - xy100) / 1.5,  # Critere à 100% XY 150cm, positif si classe A, [-inf, 1]
        "Z 90% 40cm": (0.4 - z90) / 0.4,  # Critere à 90% Z 40cm, positif si classe A, [-inf, 1]
        "Z 100% 70cm": (0.7 - z100) / 0.7  # Critere à 100% Z 70cm, positif si classe A, [-inf, 1]
    }


def compare_curves(reference, computed):
    interp_reference = interp1d_curve_step(reference, 0.01)
    tree = spatial.KDTree(interp_reference[..., :2])
    min_dist, index_min_dist = tree.query(computed[..., :2], k=1)
    axis_min_dist = interp_reference[index_min_dist, 2]
    dist_axis = numpy.abs(computed[..., 2] - axis_min_dist)
    mask = ~numpy.logical_or(index_min_dist == 0, index_min_dist == len(interp_reference) - 1)
    return min_dist[mask], dist_axis[mask]


def interp1d_curve_step(points, step):
    abs_cur = curvilinear_abs(points)
    interp_abs_cur = numpy.linspace(abs_cur[0], abs_cur[-1], int(abs_cur[-1] / step) + 1)
    interp_points = numpy.empty((interp_abs_cur.size, points.shape[1]))
    for i in range(points.shape[1]):
        interp_points[:, i] = numpy.interp(interp_abs_cur, abs_cur, points[:, i])
    return interp_points


def curvilinear_abs(points):
    return numpy.insert(numpy.cumsum(numpy.nansum((points[:-1] - points[1:]) ** 2, axis=1) ** 0.5), 0, 0)


if __name__ == "__main__":
    import draw
    draw.main(
        *compare_gis_files(
            Path(r"D:/GSUNL/Input/magmotor/magmotor_result_z3.gpkg"),
            Path(r"D:/GSUNL/Input/magmotor/ref_z3.gpkg"),
            )
    )