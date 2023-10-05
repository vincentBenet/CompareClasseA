# -*- coding: utf-8 -*-
from pathlib import Path

from qgis.PyQt.QtCore import QSettings, QTranslator, QCoreApplication
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction
from qgis.core import QgsProject, Qgis, QgsVectorLayer, QgsField, QgsFeature, QgsGeometry, QgsPoint, QgsVectorFileWriter, QgsWkbTypes, QgsMapLayer

from . import compare, draw
from .resources import *
from .CompareClasseA_dialog import CompareClasseADialog
import os.path


class CompareClasseA:
    """QGIS Plugin Implementation."""

    def __init__(self, iface):
        self.iface = iface
        self.plugin_dir = os.path.dirname(__file__)
        locale = QSettings().value('locale/userLocale')[0:2]
        locale_path = os.path.join(
            self.plugin_dir,
            'i18n',
            'CompareClasseA_{}.qm'.format(locale))

        if os.path.exists(locale_path):
            self.translator = QTranslator()
            self.translator.load(locale_path)
            QCoreApplication.installTranslator(self.translator)

        self.actions = []
        self.menu = self.tr(u'&CompareClasseA')
        self.first_start = None
        self.layers = []

    def init(self):
        layers = QgsProject.instance().mapLayers().values()
        print(layers)
        self.layers = {
            layer.name(): layer.source().split("|")[0]
            for layer in layers
            if (
                layer.geometryType() == QgsMapLayer.VectorLayer
            )
        }
        self.dlg.comboBox_reference.clear()
        self.dlg.comboBox_result.clear()
        self.dlg.comboBox_reference.addItems(self.layers)
        self.dlg.comboBox_result.addItems(self.layers)

    # noinspection PyMethodMayBeStatic
    def tr(self, message):
        return QCoreApplication.translate('CompareClasseA', message)

    def add_action(
        self,
        icon_path,
        text,
        callback,
        enabled_flag=True,
        add_to_menu=True,
        add_to_toolbar=True,
        status_tip=None,
        whats_this=None,
        parent=None):
        icon = QIcon(icon_path)
        action = QAction(icon, text, parent)
        action.triggered.connect(callback)
        action.setEnabled(enabled_flag)
        if status_tip is not None:
            action.setStatusTip(status_tip)
        if whats_this is not None:
            action.setWhatsThis(whats_this)
        if add_to_toolbar:
            self.iface.addToolBarIcon(action)
        if add_to_menu:
            self.iface.addPluginToMenu(
                self.menu,
                action)
        self.actions.append(action)
        return action

    def initGui(self):
        icon_path = ':/plugins/CompareClasseA/icon.png'
        self.add_action(
            icon_path,
            text=self.tr(u'CompareClasseA'),
            callback=self.run,
            parent=self.iface.mainWindow())
        self.first_start = True

    def unload(self):
        for action in self.actions:
            self.iface.removePluginMenu(
                self.tr(u'&CompareClasseA'),
                action)
            self.iface.removeToolBarIcon(action)

    def run(self):
        if self.first_start:
            self.first_start = False
            self.dlg = CompareClasseADialog()
        self.init()
        self.dlg.show()
        result = self.dlg.exec_()
        if result:
            print(self.layers)
            path_ref = self.dlg.comboBox_reference.currentText()
            path_result = self.dlg.comboBox_result.currentText()
            scores, diff_xy, diff_z, reference, computed = compare.compare_gis_files(
                Path(self.layers[path_ref]),
                Path(self.layers[path_result])
            )
            draw.main(scores, diff_xy, diff_z, reference, computed)
