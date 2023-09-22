from shapely.geometry import Point
import geopandas
import numpy


def load_gpkg(path, start, stop):
    data = geopandas.read_file(path)
    return numpy.array([
        data.geometry.x,
        data.geometry.y, 
        data.geometry.z,
    ]).T[start:stop], data.geometry.crs


def write_gpkg(path, data, epsg):
    geopandas.GeoDataFrame(
        geometry=[Point(xyz) for xyz in data],
        crs=epsg
    ).to_file(path, driver="GPKG")
    print(f"File saved in {path}")
