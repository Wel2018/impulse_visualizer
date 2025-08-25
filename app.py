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
    # {"theta": np.pi,     "z": 15.0, "diameter_theta": np.pi / 8, "diameter_z": 0.7},
    # {"theta": np.pi,     "z": 10.0, "diameter_theta": np.pi / 4, "diameter_z": 2},
    # {"theta": np.pi,     "z": 6.0,  "diameter_theta": np.pi / 8, "diameter_z": 0.7},
    # {"theta": np.pi,     "z": 5.0,  "diameter_theta": np.pi / 8, "diameter_z": 0.7},
    # {"theta": np.pi,     "z": 4.0,  "diameter_theta": np.pi / 8, "diameter_z": 0.7},
    # {"theta": np.pi,     "z": 3.0,  "diameter_theta": np.pi / 8, "diameter_z": 0.7},
    # {"theta": np.pi,     "z": 2.0,  "diameter_theta": np.pi / 8, "diameter_z": 0.7},
    # {"theta": np.pi,     "z": 100.0,  "diameter_theta": np.pi / 8, "diameter_z": 7},
    # {"theta": np.pi,     "z": 60.0,   "diameter_theta": np.pi / 8, "diameter_z": 7},
    # {"theta": np.pi,     "z": 50.0,   "diameter_theta": np.pi / 8, "diameter_z": 7},
    {"theta": .06,     "z": 40.0,   "diameter_theta": np.pi / 8, "diameter_z": 7},
    {"theta": .06,     "z": 30.0,   "diameter_theta": np.pi / 8, "diameter_z": 7},
    {"theta": .06,     "z": 20.0,   "diameter_theta": np.pi / 8, "diameter_z": 7},
]

marker_points = [
    {"theta": np.pi / 2, "z": 150},
    {"theta": np.pi / 2, "z": 100},
    {"theta": np.pi / 2, "z": 50},
    {"theta": np.pi / 2, "z": 10},
    # {"theta": 3 * np.pi / 4, "z": 2.8},
    # {"theta": 3 * np.pi / 4, "z": 2.9},
    # {"theta": 3 * np.pi / 4, "z": 3.0},
    # {"theta": 3 * np.pi / 4, "z": 3.1},
    # {"theta": 3 * np.pi / 4, "z": 3.2},
    # {"theta": 3 * np.pi / 4, "z": 3.3},
    # {"theta": 2.9 * np.pi / 4, "z": 3.3},
    # {"theta": 2.8 * np.pi / 4, "z": 3.3},
    # {"theta": 2.7 * np.pi / 4, "z": 3.3},
    # {"theta": 2.6 * np.pi / 4, "z": 3.3},
    # {"theta": 2.5 * np.pi / 4, "z": 3.3},
    # {"theta": np.pi, "z": 3.0},
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
        self.setWindowTitle("Cylinder Impulse Visualizer (2025-08-25)")
        self.setGeometry(100, 100, 1000, 600)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        # 水平布局：左侧 PyVista，右侧控制面板
        h_layout = QHBoxLayout()
        main_widget.setLayout(h_layout)

        # # 创建 PyVista 对应的 QWidget 控件
        self.pl = QtInteractor(self) # type: ignore
        h_layout.addWidget(self.pl, stretch=3)

        # # 右侧控制面板
        # control_panel = QWidget()
        # v_layout = QVBoxLayout()
        # control_panel.setLayout(v_layout)

        # # 添加两个按钮
        # btn1 = QPushButton("加载数据")
        # btn2 = QPushButton("可视化")
        # v_layout.addWidget(btn1)
        # v_layout.addWidget(btn2)

        # # 添加文本框
        # text_box = QLineEdit()
        # text_box.setPlaceholderText("请输入文件路径")
        # text_box.setText("visualization.csv")
        # v_layout.addWidget(text_box)

        # # 添加弹性空间，使控件靠上
        # v_layout.addStretch()

        # # 将控制面板加入水平布局
        self.control_panel = ControlWidget()
        h_layout.addWidget(self.control_panel, stretch=1)

        self.control_panel.ui.opacity.valueChanged.connect(self.set_opacity)

        # 创建锥体并添加到渲染器
        # cone = pv.Cone(resolution=8)
        # self.plotter.add_mesh(cone, color="red", show_edges=True)

        # 设置背景颜色
        #self.plotter.set_background("blue")
        self.init()

    def set_opacity(self, val: float):
        mesh = self.m.actor_mesh
        mesh.GetProperty().SetOpacity(val / 100)

    def init(self):
        from hot_cylinder import HotCylinder
        self.m = HotCylinder(self.pl)
        self.m.create_body(150, 80 // 2, 50, 50)
        self.m.add_heat_spots(heat_spots)

        # 添加标记
        self.m.add_markers(marker_points)
        self.m.submit_markers()
        # self.m._add_arrow()

        # self.m._add_box(
        #     0,0, 150, 30,30, 0.001, 
        #     color="yellow", opacity=0.1, show_edges=True)
        
        # 底面
        self.m._add_box(
            0,0, 0, 100,100, 0.001, 
            color="yellow", opacity=0.1, show_edges=True)

        # 添加旋转矩形
        from csv_parser import CSVParser
        parser = CSVParser()
        # for i in range(300, parser.length):
        # for i in range(parser.length):
        i = 0
        rect = parser.get(i)
        self.m.add_rotated_rectangle(rect)
        self.m.submit(opacity=1)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PyVistaWindow()
    window.show()
    sys.exit(app.exec())
