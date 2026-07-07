# LiDAR Point Cloud Visualization Project

基于 Flask + Open3D + Three.js 的 LiDAR 点云可视化 Web 应用。

## 功能特性

- **点云加载**：支持 PCD、PLY、BIN、XYZ、PTS 格式
- **点云处理**：统计滤波去噪、体素下采样、RANSAC 地面分割
- **3D 可视化**：高度着色、强度着色、分类着色
- **视角控制**：Orbit 轨道模式、FPS 第一人称模式
- **帧序列播放**：多帧点云序列播放、暂停、跳转

## 技术栈

- **后端**：Flask 3.0、Open3D 0.18、NumPy、SciPy
- **前端**：Three.js、HTML5、CSS3、JavaScript

## 快速开始

### 环境要求

- Python 3.8+
- pip

### 安装依赖

```bash
pip install -r requirements.txt
```

### 启动服务

```bash
python app.py
```

### 访问应用

打开浏览器访问 http://localhost:5000

## API 接口

| 接口 | 方法 | 描述 |
|------|------|------|
| `/api/files` | GET | 获取点云文件列表 |
| `/api/upload` | POST | 上传点云文件 |
| `/api/process` | POST | 处理点云（滤波、下采样、分割） |
| `/api/frames` | POST | 获取帧序列列表 |

## 项目结构

```
task_script/
├── app.py              # Flask 后端服务
├── requirements.txt    # Python 依赖
├── data/               # 点云数据文件
│   ├── *.pcd
│   ├── *.ply
│   └── *.bin
└── static/
    └── index.html      # 前端页面
```

## 数据格式

支持以下点云格式：
- **PCD**：Point Cloud Data 格式
- **PLY**：Polygon File Format
- **BIN**：二进制格式（N×4 或 N×3 float32）
- **XYZ**：ASCII 文本格式
- **PTS**：ASCII 文本格式

## 许可证

MIT License
