import csv


class CSVParser:
    """
    ```
    获取第0行数据
    record = parser.get(0)
    print(record)

    输出类似：
    {
        'center': {'x': ..., 'y': ..., 'z': ...},
        'rectangle': {'width': ..., 'length': ..., 'angle_x_deg': ...},
        'normals': [
            {'x': ..., 'y': ..., 'z': ...},
            {'x': ..., 'y': ..., 'z': ...},
            ...
        ]
    }
    ```
    """
    def __init__(self, filepath='visualization.csv'):
        self.data = []
        self.length = 0
        self._load_file(filepath)

    def _load_file(self, filepath):
        with open(filepath, 'r', newline='') as f:
            reader = csv.reader(f, delimiter=',')
            # 读取前6行字段说明（总共21个字段）
            header_lines = [next(reader) for _ in range(6)]
            self.field_names = sum(header_lines, [])  # 扁平化字段列表

            # 读取数据部分
            for row in reader:
                float_row = [float(val.strip()) for val in row if val.strip()]
                self.data.append(float_row)
                self.length += 1

    def get(self, index):
        """
        X_POS	Y_POS	Z_POS	
        width	length	angle	
        I1_x	I1_y	I1_z
        I2_x	I2_y	I2_z
        I3_x	I3_y	I3_z
        I4_x	I4_y	I4_z
        """
        row = self.data[index]
        
        result = {
            'center': {
                'x': row[0],
                'y': row[1],
                'z': row[2],
                'width': row[3],
                'length': row[4],
                'angle': row[5],
            },
            'impulse': []
        }

        for i in range(6, len(row), 3):
            nx, ny, nz = row[i:i+3]
            if nx == 0 and ny == 0 and nz == 0:
                continue  # 忽略无效冲击
            result['impulse'].append({'x': nx, 'y': ny, 'z': nz})

        return result
