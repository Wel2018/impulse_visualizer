import pyvista as pv
from pyvistaqt import QtInteractor
import numpy as np


class HotCylinder():
    """在 pyvista 空间中绘制圆柱体，要求可视化网格、热图、标记点、法向箭头等"""
    def __init__(self, pl: QtInteractor) -> None:
        self.pl: pv.Plotter = pl
        self.pl.set_background("#303c64") # type: ignore
        self.pl.add_camera_orientation_widget()

        self.body_grid = None
        self.label_positions = []
        self.label_texts = []
        self.el = {}
        self.el_idx = 0
    
    def create_body(self, height=4.0, radius=1.0, n_theta=100, n_z=100):
        """构建网格圆柱体，高为 height，半径为 radius，网格为 n_theta x n_z"""
        # 构建网格
        theta = np.linspace(0, 2 * np.pi, n_theta)
        z = np.linspace(0, height, n_z)
        theta_grid, z_grid = np.meshgrid(theta, z)
        x = radius * np.cos(theta_grid)
        y = radius * np.sin(theta_grid)
        points = np.stack((x, y, z_grid), axis=-1)

        # 创建结构化网格
        grid = pv.StructuredGrid()
        grid.points = points.reshape(-1, 3)
        grid.dimensions = [n_theta, n_z, 1]
        self.heat = np.zeros_like(theta_grid)  # 热力图网格

        self.body_grid = grid
        self.body_theta_grid = theta_grid
        self.body_z_grid = z_grid
        self.body_height = height
        self.body_radius = radius
        self.body_n_theta = n_theta
        self.body_n_z = n_z


    def _add_heat_spot(self, theta, z, diameter_theta=np.pi / 8, diameter_z=7):
        """添加一个热点
        theta	        热点中心点的位置（沿 θ 方向，弧度值）
        z	            热点中心点的高度位置
        diameter_theta	θ 方向上热点的直径（不是方差！是直径），用于控制热点在圆周方向上的宽度
        diameter_z	    z 方向上热点的直径（同样，不是方差），用于控制热点在竖直方向上的宽度
        """
        # {"theta": np.pi / 2, "z": 1.5, "diameter_theta": np.pi / 6, "diameter_z": 1.0},
        assert self.body_grid
        theta_grid = self.body_theta_grid
        z_grid = self.body_z_grid
        center_theta = theta
        center_z = z
        sigma_theta = diameter_theta / 2.355
        sigma_z = diameter_z / 2.355
    
        # theta = theta_grid - center_theta
        theta = np.angle(np.exp(1j * (theta_grid - center_theta)))

        # 周期高斯核：在 theta 方向上，普通的高斯分布 exp(-((θ - θ₀)² / 2σ²)) 不能正确处理圆周边界。比如：theta = 0（中心）时， theta = 2π - ε 的点其实非常接近中心，但在公式里差值大，导致它被排除。这是由于 theta 的值存在周期性（范围通常是 [-π, π] 或 [0, 2π]），而你当前的高斯函数并**没有考虑这个“角度环绕”**特性。
        self.heat += np.exp(
            - (theta**2 / (2 * sigma_theta**2) +
               (z_grid - center_z)**2 / (2 * sigma_z**2))
        )

        _heat = self.heat.copy()
        _heat = _heat / np.max(_heat)
        self.body_grid["heat"] = _heat.ravel()


    def add_heat_spots(self, heat_spots: list):
        for spot in heat_spots:
            self._add_heat_spot(
                spot["theta"], spot["z"], 
                spot["diameter_theta"], 
                spot["diameter_z"])
    

    def _add_marker(self, theta, z, msg=""):
        radius = self.body_radius
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        pos = np.array([x, y, z])
        
        # 使用红色小球表示标记
        sphere_radius = 0.03*10
        sphere = pv.Sphere(radius=sphere_radius, center=pos)
        el = self.pl.add_mesh(sphere, color="red")
        self.label_positions.append(pos)
        self.label_texts.append(msg)
        self.el[f"marker_{self.el_idx}"] = el
        self.el_idx += 1

    def _add_box(
            self, x,y,z,width,length,height,**kargs,
        ):
        box = pv.Box(bounds=[x-width/2, x+width/2, y-length/2, y+length/2, z-height/2, z+height/2])
        el = self.pl.add_mesh(box, **kargs)
        self.el[f"box_{self.el_idx}"] = el
        self.el_idx += 1

    def add_markers(self, markers: list):
        for marker in markers:
            # print(f"添加 marker：{marker}")
            self._add_marker(marker["theta"], marker["z"])


    def submit_markers(self):
        # print(self.label_positions)
        # print(self.label_texts)

        el = self.pl.add_point_labels(
            self.label_positions,
            self.label_texts,
            font_size=12,
            text_color="white",
            point_color="red",
            point_size=10,
            shape_opacity=0.5,
            always_visible=True,
        )
        self.el["markers"] = el

    def _add_arrow(self, theta, z):
        # 热斑中心点（圆柱体表面）
        arrow_len = 0.1  # 箭头长度
        radius = self.body_radius

        end = np.array([
            radius * np.cos(theta),
            radius * np.sin(theta), z])

        # 法向量单位方向（圆柱侧面）
        normal = np.array([np.cos(theta), np.sin(theta), 0.0])

        # 箭头起点（在表面外部）
        # start = end + normal * arrow_len
        start = end

        # 箭头方向
        # direction = end - start  # 就是 -normal * arrow_len
        direction = normal * arrow_len  # 或 direction = end + normal*arrow_len - start

        # 创建箭头
        arrow = pv.Arrow(
            start=start,
            direction=direction,
            tip_length=0.1,  # 尖尖的长度
            tip_radius=0.03,
            shaft_radius=0.01  # 箭头身体直径
        )
        self.pl.add_mesh(arrow, color="white")


    def submit(self, **kargs):
        self.actor_mesh = self.pl.add_mesh(
            self.body_grid, # type: ignore
            scalars="heat",
            cmap="rainbow",
            show_edges=True,
            clim=[0, 1],
            specular=0.1,
            smooth_shading=False,
            ambient=1.0,
            diffuse=0.0,
            # opacity=0.9
            **kargs
        )
        self.el["body"] = self.actor_mesh

    # 旋转矩形 -------------------------------------------------

    @staticmethod
    def create_rotated_rectangle(x, y, z, width, length, angle_deg):
        # 定义局部平面内的矩形4个点（XY 平面上）
        w, l = width / 2, length / 2
        corners = np.array([
            [-w, -l, 0],
            [ w, -l, 0],
            [ w,  l, 0],
            [-w,  l, 0],
        ])

        # 构造绕 X 轴的旋转矩阵
        angle_rad = np.radians(angle_deg)
        rot_x = np.array([
            [1, 0, 0],
            [0, np.cos(angle_rad), -np.sin(angle_rad)],
            [0, np.sin(angle_rad),  np.cos(angle_rad)]
        ])

        # 绕 Z 轴旋转 90 度（正方向）
        angle_rad_z = np.radians(90)
        rot_z = np.array([
            [np.cos(angle_rad_z), -np.sin(angle_rad_z), 0],
            [np.sin(angle_rad_z),  np.cos(angle_rad_z), 0],
            [0, 0, 1]
        ])

        # 先绕 X，再绕 Z（注意矩阵乘法顺序：先应用 X，再 Z → rot_z @ rot_x）
        # rotation_matrix = rot_z @ rot_x

        # 旋转 + 平移
        rotated = (rot_x @ corners.T).T + np.array([x, y, z])
        return rotated, rot_x

    def add_rotated_rectangle(self, rect: dict):
        scale = 1
        rect_center = rect['center']
        rect_impulse = rect['impulse']
        points, rot_x = self.create_rotated_rectangle(
            rect_center["x"],
            rect_center["y"],
            rect_center["z"],
            rect_center["width"],
            rect_center["length"],
            rect_center["angle"],
        )

        # 使用 PolyData + 面连接
        faces = [4, 0, 1, 2, 3]  # 4个点构成一个面
        mesh = pv.PolyData(points, faces)
        act = self.pl.add_mesh(mesh, color="yellow", opacity=1, show_edges=True)
        # self.pl.remove_actor()

        # 法向量：原始平面的法向量 (0,0,1) 经过旋转矩阵
        normal_local = np.array([0, 0, 1])
        normal_world = rot_x @ normal_local

        # print(normal_world)

        arrow_center = np.array([
            rect_center["x"], 
            rect_center["y"], 
            rect_center["z"]])
        
        # 在中心点绘制箭头
        self.pl.add_arrows(
        arrow_center, 
        normal_world, 
        mag=1.0, 
        color="white")


        # 前三个是中心点的坐标
        # 4-6为矩形的长、宽、以及旋转绕x轴旋转的角度
        # 7-21每三个为法向量的x，y，z的方向，如果xyz都为0则不存在这个冲击
        # 一共有300多组数据，前面的都是两个冲击的，后面有81个4个冲击的
        for impluse in rect_impulse:
            vec = normal_world.copy()
            # vec = normal_local.copy()
            # normal_local = np.array([0, 0, 1])
            # normal_world = rot_x @ normal_local
            vec[0] += impluse["x"] * scale
            vec[1] += impluse["y"] * scale
            vec[2] += impluse["z"] * scale
            self.pl.add_arrows(
                arrow_center,
                vec, 
                # normal_world, 
                mag=2.0, 
                color="#00ff33")
        
        # 将旋转矩形中心点对应的坐标转换为圆柱体坐标，并在外侧绘制一个热点
        x,y = arrow_center[0], arrow_center[1]
        theta = self.point_to_cylinder_coordinates(x, y)
        msg = f"x={x:.2f} y={y:.2f} \n theta={theta:.2f}"
        self._add_marker(theta, arrow_center[2], msg)
        self.submit_markers()
        self._add_heat_spot(theta, arrow_center[2])


    @staticmethod
    def point_to_cylinder_coordinates(x, y):
        # 要从圆柱体的原点 (0, 0, 0) 出发，计算空间中某一点 (x, y, z) 相对于该圆柱的：
        # theta（角度）：该点在 XY 平面上相对于 X 轴的极角（单位为弧度或角度）
        # 高度差：即 Z 坐标上的差值（直接就是 z）

        import numpy as np
        # theta：从 x 轴逆时针转到该点的角度（单位：弧度）
        theta_rad = np.arctan2(y, x)  # 范围 [-π, π]
        theta_deg = np.degrees(theta_rad)  # 可选，角度表示

        # 高度差
        # height = z

        return theta_rad

    def clean_all(self):
        self.pl.clear()
        # self.pl.clear_actors()
        # 清空元素
        for k in self.el.keys():
            # print(k)
            self.pl.remove_actor(self.el[k])
        
        # 清空文本标记
        self.label_positions.clear()
        self.label_texts.clear()

        # 清空热力图
        self.heat = np.zeros_like(self.body_theta_grid)  # 热力图网格
