import math
import numpy
import geopandas
import pyproj
from pathlib import Path
from scipy import spatial
import os


def coordinates_from_gis(gis_file_paths, unit="metre", espg_global="EPSG:3857"):

    elevation_conversions = {  # https://en.wikipedia.org/wiki/Foot_(unit)
        "meter": 1,
        "international_foot": 0.3048,
        "us_survey_foot": 1200/3937,
        "indian_survey_foot": 0.3047996,
        "metric_foot": 1/3,
        "hesse_foot": 1/4,
        "baden foot": 0.3,
    }

    res = []
    for path, unit_layer in gis_file_paths:
        data = geopandas.read_file(path)
        if data.crs.axis_info[0].unit_name == unit:
            epsg = str(data.geometry.crs)
            break
    else:
        epsg = espg_global

    print(f"Projection EPSG: {epsg = }")

    for i, layer in enumerate(gis_file_paths):
        path, unit_layer = layer
        data = geopandas.read_file(path)
        epsg_from = str(data.geometry.crs)
        print(f"Loading {epsg_from} - {path}")
        x = data.geometry.x
        y = data.geometry.y
        z = data.geometry.z * elevation_conversions[unit_layer]
        pipe = numpy.array([x, y, z]).T
        mask = []
        for i, point in enumerate(pipe):
            if not (
                math.isinf(point[0]) or
                math.isinf(point[1]) or
                math.isinf(point[2]) or
                numpy.isnan(point[0]) or
                numpy.isnan(point[1]) or
                numpy.isnan(point[2])
            ):
                mask.append(i)
        if epsg_from != epsg:
            print(f"\tReproject coordinates from {epsg_from} to {epsg}")
            lon, lat = pipe[mask].T[:2]
            print(
                "\tCoordinates median (lon, lat) in degree: ",
                round(numpy.median(lon), 1),
                round(numpy.median(lat), 1))
            x, y = pyproj.transform(
                pyproj.Proj(epsg_from),
                pyproj.Proj(epsg),
                lat, lon)
            print(
                "\tCoordinates median (x, y) in meter: ",
                round(numpy.median(x), 1),
                round(numpy.median(y), 1))
            pipe = numpy.array([x, y, pipe[mask].T[2]]).T
        abc = curvilinear_abs(pipe)
        print("\tCurvilign abcisse distance: ", round(abc[-1], 1), "meter")
        if not len(pipe):
            raise Exception(f"Empty 3D geometry for file {path}")
        x, y, z = pipe.T
        res.append(numpy.array([x, y, z, abc]).T)
        print(f"\tLoaded {len(pipe)} points")
        print("_"*50)
    return res


def compare_gis_files(
    ref_path: Path,
    res_path: Path,
    ref_unit,
    res_unit,
    max_dist=0
):
    reference, computed = coordinates_from_gis(
        gis_file_paths=[
            [ref_path, ref_unit],
            [res_path, res_unit]])

    if computed.shape[0] == 0 or reference.shape[0] == 0:
        return None
    dists_xy, dists_z, res = compare_curves(reference.T[:3].T, computed.T[:3].T, max_dist)
    scores = compare_pipe_to_pipe(dists_xy, dists_z)
    print("Results: ")
    print(f"\tXY 60%  = {(scores['XY 60%'] * 100):.2f} /  20 cm ({(scores['XY 60% 20cm']*100):.2f} %)")
    print(f"\tXY 90%  = {(scores['XY 90%'] * 100):.2f} /  40 cm ({(scores['XY 90% 40cm']*100):.2f} %)")
    print(f"\tXY 100% = {(scores['XY 100%'] * 100):.2f} / 150 cm ({(scores['XY 100% 150cm']*100):.2f} %)")
    print(f"\tZ 90%   = {(scores['Z 90%'] * 100):.2f} /  40 cm ({(scores['Z 90% 40cm']*100):.2f} %)")
    print(f"\tZ 100%  = {(scores['Z 100%'] * 100):.2f} /  70 cm ({(scores['Z 100% 70cm']*100):.2f} %)")
    ref_name = ".".join(os.path.basename(ref_path).split(".")[:-1])
    res_name = ".".join(os.path.basename(res_path).split(".")[:-1])

    return scores, dists_xy, dists_z, reference, res, ref_name, res_name


def compare_pipe_to_pipe(dists_xy, dists_z):
    if not len(dists_xy):
        print(f"Exit because not len(dists_XY)")
    scores = {
        "XY 60%": numpy.percentile(dists_xy, 60),
        "XY 90%": numpy.percentile(dists_xy, 90),
        "XY 100%": numpy.percentile(dists_xy, 100),
        "Z 90%": numpy.percentile(dists_z, 90),
        "Z 100%": numpy.percentile(dists_z, 100),
    }
    scores["XY 60% 20cm"] = 1 - scores['XY 60%'] / 0.2
    scores["XY 90% 40cm"] = 1 - scores['XY 90%'] / 0.4
    scores["XY 100% 150cm"] = 1 - scores['XY 100%'] / 1.5
    scores["Z 90% 40cm"] = 1 - scores['Z 90%'] / 0.4
    scores["Z 100% 70cm"] = 1 - scores['Z 100%'] / 0.7
    scores['score_xy'] = scores['XY 60% 20cm'] + scores['XY 90% 40cm'] + scores['XY 100% 150cm']
    scores['score_z'] = scores['Z 90% 40cm'] + scores['Z 100% 70cm']
    scores['score'] = scores['score_xy'] + scores['score_z']
    scores["percent"] = scores['score'] / 5 * 100
    scores["percent_xy"] = scores['score_xy'] / 3 * 100
    scores["percent_z"] = scores['score_z'] / 2 * 100
    scores['ok_xy'] = not (
            scores['XY 60% 20cm'] < 0 or
            scores['XY 90% 40cm'] < 0 or
            scores['XY 100% 150cm'] < 0
    )
    scores["ok_z"] = not (
            scores['Z 90% 40cm'] < 0 or
            scores['Z 100% 70cm'] < 0
    )
    scores["ok"] = scores['ok_xy'] * scores["ok_z"]
    return scores


def compare_curves(reference, computed, max_dist, step=0.01):
    interp_reference = interp1d_curve_step(reference, step)
    abc_interp_reference = curvilinear_abs(interp_reference)
    tree = spatial.KDTree(interp_reference[..., :2])
    min_dist, index_min_dist = tree.query(computed[..., :2], k=1)
    axis_min_dist = interp_reference[index_min_dist, 2]
    dist_axis = numpy.abs(computed[..., 2] - axis_min_dist)
    mask = ~numpy.logical_or(index_min_dist == 0, index_min_dist == len(interp_reference) - 1)
    diff_xy = min_dist[mask]
    diff_z = dist_axis[mask]
    abc = abc_interp_reference[index_min_dist][mask]
    x, y, z = computed[mask].T
    res = numpy.array([x, y, z, abc]).T
    if max_dist > 0:
        mask = diff_xy < max_dist
        res = res[mask]
        diff_xy = diff_xy[mask]
        diff_z = diff_z[mask]
    return diff_xy, diff_z, res


def interp1d_curve_step(points, step):
    abs_cur = curvilinear_abs(points)
    interp_abs_cur = numpy.linspace(abs_cur[0], abs_cur[-1], int(abs_cur[-1] / step) + 1)
    interp_points = numpy.empty((interp_abs_cur.size, points.shape[1]))
    for i in range(points.shape[1]):
        interp_points[:, i] = numpy.interp(interp_abs_cur, abs_cur, points[:, i])
    return interp_points


def curvilinear_abs(points):
    return numpy.insert(
        numpy.cumsum(
            numpy.nansum(
                (
                    points[:-1] - points[1:]
                ) ** 2, axis=1
            ) ** 0.5),
        0, 0
    )


if __name__ == "__main__":
    import draw
    draw.main(
        *compare_gis_files(
            Path(os.path.join(os.path.dirname(__file__), "data_test", "magmotor_result_z3.gpkg")),
            Path(os.path.join(os.path.dirname(__file__), "data_test", "z3_ref.gpkg")),
            "meter",
            "meter",
            0.25
        )
    )
