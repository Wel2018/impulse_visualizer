import sys
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMainWindow
import numpy as np
from pyvistaqt import QtInteractor
import pyvista as pv

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QHBoxLayout, QVBoxLayout, QPushButton, QLineEdit
)

import os

os.environ["QT_API"] = "pyside6"

# 圆柱的直径是80mm，高是150mm，
# 上下面的圆心坐标分别是（0，0，150）和（0，0，0）
# 这里面的第一行就是，具体的含义为:
# 前三个是中心点的坐标
# 4-6为矩形的长、宽、以及旋转绕x轴旋转的角度
# 7-21每三个为法向量的x，y，z的方向，如果xyz都为0则不存在这个冲击
# 一共有300多组数据，前面的都是两个冲击的，后面有81个4个冲击的

heat_spots = [
    # {"theta": np.pi,     "z": 100.0,  "diameter_theta": np.pi / 8, "diameter_z": 7},
    # {"theta": np.pi,     "z": 60.0,   "diameter_theta": np.pi / 8, "diameter_z": 7},
    # {"theta": .5,     "z": 50.0,   "diameter_theta": np.pi / 8, "diameter_z": 20},
    # {"theta": .06,     "z": 60.0,   "diameter_theta": np.pi / 8, "diameter_z": 7},
    # {"theta": .06,     "z": 50.0,   "diameter_theta": np.pi / 8, "diameter_z": 7},
    {"theta": .06,     "z": 20.0,   "diameter_theta": np.pi / 4, "diameter_z": 7},
]

marker_points = [
    # {"theta": np.pi / 2, "z": 150},
    # {"theta": np.pi / 2, "z": 100},
    # {"theta": np.pi / 2, "z": 50},
    # {"theta": np.pi / 2, "z": 10},
    # {"theta": 3 * np.pi / 4, "z": 2.8},
    {"theta": 3 * np.pi / 4, "z": 2.9},
    # {"theta": 3 * np.pi / 4, "z": 3.0},
    # {"theta": 3 * np.pi / 4, "z": 3.1},
    # {"theta": 3 * np.pi / 4, "z": 3.2},
    # {"theta": 3 * np.pi / 4, "z": 3.3},
    # {"theta": 2.9 * np.pi / 4, "z": 3.3},
    # {"theta": 2.8 * np.pi / 4, "z": 3.3},
    # {"theta": 2.7 * np.pi / 4, "z": 3.3},
    # {"theta": 2.6 * np.pi / 4, "z": 3.3},
    # {"theta": 2.5 * np.pi / 4, "z": 3.3},
]


#####################################################################


class ControlWidget(QWidget):
    def __init__(self):
        super().__init__()
        from ui_control import Ui_Form
        self.ui = Ui_Form()
        # 将 UI 安装到当前 widget 上
        self.ui.setupUi(self)


class PyVistaWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # PyVista in PySide6
        self.setWindowTitle("Cylinder Impulse Visualizer (2025-08-28)")
        self.setGeometry(100, 100, 1000, 600)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        # 水平布局：左侧 PyVista，右侧控制面板
        h_layout = QHBoxLayout()
        main_widget.setLayout(h_layout)

        # # 创建 PyVista 对应的 QWidget 控件
        self.pl = QtInteractor(self) # type: ignore
        h_layout.addWidget(self.pl, stretch=3)

        # # 将控制面板加入水平布局
        self.control_panel = ControlWidget()
        h_layout.addWidget(self.control_panel, stretch=1)

        # 初始化时默认加载第一条样本，可以选择加载其他样本
        self.control_panel.ui.opacity.valueChanged.connect(self.set_opacity)
        self.control_panel.ui.btn_clean.clicked.connect(self.clean_all)
        self.control_panel.ui.btn_load_sample.clicked.connect(self.load_sample)
        self.control_panel.ui.btn_prev.clicked.connect(self.load_sample_prev)
        self.control_panel.ui.btn_next.clicked.connect(self.load_sample_next)

        # 相机聚焦
        self.control_panel.ui.btn_focus_body.clicked.connect(self.focus_body)
        self.control_panel.ui.btn_focus_bottom.clicked.connect(self.focus_bottom)
        self.control_panel.ui.btn_focus_side.clicked.connect(self.focus_side)

        from csv_parser import CSVParser
        self.parser = CSVParser()
        self.add_log(f"加载成功 (样本量 {self.parser.length}) \n")
        self.control_panel.ui.spinBox.setMaximum(self.parser.length - 1)
        self.init()

    def focus_body(self):
        # xy, xz, yz, yx, zx, zy, iso
        # self.pl.camera_position = "xyz"
        self.m.pl.fly_to(self.m.mesh_body.center) # type: ignore

    def focus_bottom(self):
        # self.pl.camera_position = "xz"
        self.m.pl.fly_to(self.m.mesh_bottom.center) # type: ignore

    def focus_side(self):
        # self.pl.camera_position = "yz"
        self.m.pl.fly_to(self.m.mesh_side.center) # type: ignore


    def set_opacity(self, val: float):
        for mesh in [self.m.actor_body, self.m.actor_top, self.m.actor_bottom]:
            if mesh is not None:
                mesh.GetProperty().SetOpacity(val / 100)

    def init(self):
        from hot_cylinder import HotCylinder
        self.m = HotCylinder(self.pl)
        self.m.create_body(150, 80 // 2, 50, 50)
        # self.m.add_heat_spots(heat_spots)

        # 添加标记
        # self.m.add_markers(marker_points)
        # self.m.submit_markers()
        # self.m._add_arrow()

        # 加载样本
        self.load_sample()
        
    def load_sample_prev(self):
        sample_id = int(self.control_panel.ui.spinBox.value())
        if sample_id > 0:
            self.control_panel.ui.spinBox.setValue(sample_id - 1)
            self.load_sample()

    def load_sample_next(self):
        sample_id = int(self.control_panel.ui.spinBox.value())
        if sample_id < self.parser.length - 1:
            self.control_panel.ui.spinBox.setValue(sample_id + 1)
            self.load_sample()
        
    def load_sample(self):
        # 清空
        self.clean_all()
        # self.m.add_heat_spots(heat_spots)
        # 加载指定样本
        sample_id = int(self.control_panel.ui.spinBox.value())
        self.add_log(f"加载样本 {sample_id} / (样本量 {self.parser.length}) \n")

        # 顶面底面
        # self.m._add_box(0,0, 150, 30,30, 0.001, color="yellow", opacity=0.1, show_edges=True)
        # self.m._add_box(0,0, 0, 100,100, 0.001, color="yellow", opacity=0.1, show_edges=True)

        def _add_ellipse(z):
            return self.m._add_ellipse(
                0,0,z, 40, 40, 0.001, 100, 
                color="blue", opacity=0.5, show_edges=True) # type: ignore
        
        # x 给圆柱体添加顶面和底面
        self.m.actor_top = _add_ellipse(0) # type: ignore
        self.m.actor_bottom = _add_ellipse(150) # type: ignore
        
        # self.m.load_3d_file("cup.stl", 1500)

        # self.m._add_plane()

        p = self.parser.get(sample_id)
        self.m.add_rotated_rectangle(p)
        self.m.submit(opacity=1)
        # self.m.submit(opacity=0.5)

    def add_log(self, msg):
        self.control_panel.ui.txt_log.insertPlainText(msg)

    def clean_all(self):
        self.m.clean_all()



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PyVistaWindow()
    window.show()
    sys.exit(app.exec())
