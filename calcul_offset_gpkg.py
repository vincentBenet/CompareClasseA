import scipy
import numpy
import os
import matplotlib.pyplot as plt
import utils_gpkg
import compare_pipes


def minimize(ref, data):
    return scipy.optimize.minimize(
        func_score,
        x0=[0, 0, 0],
        args=(ref, data),
        method="CG",
    ).x


def smooth(pipeline, step, width):
    width_size = int(width / step)
    abs_cur = numpy.r_[[0], numpy.cumsum(numpy.linalg.norm(numpy.diff(pipeline[..., :3], axis=0), axis=1))]
    abs_to = numpy.arange(0, abs_cur[-1], step)
    interp_pipe = scipy.interpolate.interp1d(abs_cur, pipeline[..., :3], axis=0)(abs_to)
    return numpy.array([
        scipy.ndimage.uniform_filter1d(interp_pipe[..., 0], size=width_size, mode="nearest"),
        scipy.ndimage.uniform_filter1d(interp_pipe[..., 1], size=width_size, mode="nearest"),
        scipy.ndimage.uniform_filter1d(interp_pipe[..., 2], size=width_size, mode="nearest"),
    ]).T


def func_score(x0, ref, data):
    ref_smooth = smooth(ref, 0.5, 0.5)
    data_offset = numpy.array([[x + x0[0], y + x0[1], z + x0[2]] for x, y, z in data])
    score = compare_pipes.compare(ref_smooth, data_offset)
    return score


def main(path_ref, path_data):
    if isinstance(path_ref, list):
        path_ref, start_ref, stop_ref = path_ref
    else:
        start_ref, stop_ref = 0, -1
    if isinstance(path_data, list):
        path_data, start_data, stop_data = path_ref
    else:
        start_data, stop_data = 0, -1
    ref, epsg_ref = utils_gpkg.load_gpkg(path_ref, start_ref, stop_ref)
    data, epsg_data = utils_gpkg.load_gpkg(path_data, start_data, stop_data)
    assert epsg_ref == epsg_data
    print(f"{len(ref) = }")
    print(f"{len(data) = }")
    score_init = func_score([0, 0, 0], ref, data)
    print(f"{score_init = }")
    compare_pipes.compare(ref, data)
    offset = minimize(ref, data)
    print(f"{offset = }")
    score_result = func_score(offset, ref, data)
    result = numpy.array([[x + offset[0], y + offset[1], z + offset[2]] for x, y, z in data])
    compare_pipes.compare(ref, result)
    print(f"{offset = }")
    print(f"{score_result = }")
    utils_gpkg.write_gpkg(
        os.path.join(os.path.dirname(path_data), '.'.join(os.path.basename(path_data).split('.')[:-1]) + "_offset.gpkg"), 
        result,
        epsg_data,
    )
    ax = plt.figure().add_subplot(projection='3d')
    xref, yref, zref = ref.T
    xdata, ydata, zdata = data.T    
    xresult, yresult, zresult = result.T
    ax.scatter(xref, yref, zref, label="ref")
    ax.plot(xdata, ydata, zdata, label="data")
    ax.plot(xresult, yresult, zresult, label="result")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    main(
        r"D:\Share - SkipperNDT\RVC_US-Colorado-Hamilton_2023-09-05\Delivrables\EPSG 32613\ref_z2_32613.gpkg", 
        r"D:\Share - SkipperNDT\RVC_US-Colorado-Hamilton_2023-09-05\outputs\result_f65c28e8.gpkg",
    )
    
    # main(
        # r"D:\Share - SkipperNDT\RVC_US-Colorado-Hamilton_2023-09-05\Delivrables\EPSG 32613\ref_z2_32613.gpkg", 
        # r"D:\Share - SkipperNDT\RVC_US-Colorado-Hamilton_2023-09-05\outputs\result_f65c28e8.gpkg",
    # )
    
    # main(
        # r"D:\Share - SkipperNDT\RVC_US-Colorado-Hamilton_2023-09-05\Delivrables\EPSG 32613\ref_z2-2_32613.gpkg", 
        # r"D:\Share - SkipperNDT\RVC_US-Colorado-Hamilton_2023-09-05\outputs\result_2822f301.gpkg",
    # )
    
    # main(
        # r"D:\Share - SkipperNDT\RVC_US-Colorado-Hamilton_2023-09-05\Delivrables\EPSG 32613\ref_z4_32613.gpkg", 
        # r"D:\Share - SkipperNDT\RVC_US-Colorado-Hamilton_2023-09-05\outputs\result_ceaac239.gpkg",
    # )

