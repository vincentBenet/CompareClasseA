# -*- coding: utf-8 -*-
"""
/***************************************************************************
 CompareClasseA
                                 A QGIS plugin
 Compare two datasets of localisations points based on french legislation formatting
 Generated by Plugin Builder: http://g-sherman.github.io/Qgis-Plugin-Builder/
                             -------------------
        begin                : 2023-09-22
        copyright            : (C) 2023 by Vincent Bénet
        email                : vincent.benet@outlook.fr
        git sha              : $Format:%H$
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
 This script initializes the plugin, making it known to QGIS.
"""


# noinspection PyPep8Naming
def classFactory(iface):  # pylint: disable=invalid-name
    """Load CompareClasseA class from file CompareClasseA.

    :param iface: A QGIS interface instance.
    :type iface: QgsInterface
    """
    #
    from .CompareClasseA import CompareClasseA
    return CompareClasseA(iface)
