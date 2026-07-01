"""
Generate sample point cloud data simulating an autonomous driving scene.
Creates .pcd files with ground plane, vehicles, pedestrians, and vegetation.
"""
import os
import numpy as np
import open3d as o3d


def generate_ground_plane(width=100, length=100, density=0.5, noise=0.02):
    n_points = int(width * length * density)
    x = np.random.uniform(-width / 2, width / 2, n_points)
    y = np.random.uniform(-length / 2, length / 2, n_points)
    z = np.random.normal(0, noise, n_points)
    return np.column_stack([x, y, z])


def generate_road(width=8, length=80, density=2.0):
    n_points = int(width * length * density)
    x = np.random.uniform(-width / 2, width / 2, n_points)
    y = np.random.uniform(-length / 2, length / 2, n_points)
    z = np.random.normal(-0.05, 0.01, n_points)
    return np.column_stack([x, y, z])


def generate_vehicle(center, size=(4.5, 2.0, 1.5), n_points=800):
    points = []
    cx, cy, cz = center
    lx, ly, lz = size

    for _ in range(n_points // 6):
        points.append([cx + np.random.uniform(-lx/2, lx/2), cy + np.random.uniform(-ly/2, ly/2), cz])
        points.append([cx + np.random.uniform(-lx/2, lx/2), cy + np.random.uniform(-ly/2, ly/2), cz + lz])

    for _ in range(n_points // 6):
        points.append([cx - lx/2, cy + np.random.uniform(-ly/2, ly/2), cz + np.random.uniform(0, lz)])
        points.append([cx + lx/2, cy + np.random.uniform(-ly/2, ly/2), cz + np.random.uniform(0, lz)])

    for _ in range(n_points // 6):
        points.append([cx + np.random.uniform(-lx/2, lx/2), cy - ly/2, cz + np.random.uniform(0, lz)])
        points.append([cx + np.random.uniform(-lx/2, lx/2), cy + ly/2, cz + np.random.uniform(0, lz)])

    return np.array(points)


def generate_pedestrian(center, n_points=200):
    cx, cy, cz = center
    body_points = []
    for _ in range(n_points):
        angle = np.random.uniform(0, 2 * np.pi)
        r = np.random.uniform(0, 0.3)
        h = np.random.uniform(0, 1.7)
        body_points.append([cx + r * np.cos(angle), cy + r * np.sin(angle), cz + h])
    return np.array(body_points)


def generate_tree(center, n_points=500):
    cx, cy, cz = center
    points = []

    for _ in range(n_points // 5):
        r = np.random.uniform(0, 0.15)
        angle = np.random.uniform(0, 2 * np.pi)
        h = np.random.uniform(0, 3.0)
        points.append([cx + r * np.cos(angle), cy + r * np.sin(angle), cz + h])

    for _ in range(4 * n_points // 5):
        r = np.random.uniform(0, 2.0) * np.random.uniform(0.3, 1.0)
        angle = np.random.uniform(0, 2 * np.pi)
        h = np.random.uniform(2.5, 5.5)
        points.append([cx + r * np.cos(angle), cy + r * np.sin(angle), cz + h])

    return np.array(points)


def generate_building(center, size=(10, 8, 12), n_points=2000):
    cx, cy, cz = center
    lx, ly, lz = size
    points = []

    for _ in range(n_points // 4):
        points.append([cx - lx/2, cy + np.random.uniform(-ly/2, ly/2), cz + np.random.uniform(0, lz)])
        points.append([cx + lx/2, cy + np.random.uniform(-ly/2, ly/2), cz + np.random.uniform(0, lz)])
        points.append([cx + np.random.uniform(-lx/2, lx/2), cy - ly/2, cz + np.random.uniform(0, lz)])
        points.append([cx + np.random.uniform(-lx/2, lx/2), cy + ly/2, cz + np.random.uniform(0, lz)])

    return np.array(points)


def add_noise_points(bounds, n_points=200):
    x = np.random.uniform(bounds[0], bounds[1], n_points)
    y = np.random.uniform(bounds[2], bounds[3], n_points)
    z = np.random.uniform(bounds[4], bounds[5], n_points)
    return np.column_stack([x, y, z])


def generate_scene(frame_idx=0):
    np.random.seed(42 + frame_idx)

    all_points = []

    ground = generate_ground_plane(100, 100, density=0.3)
    all_points.append(ground)

    road = generate_road(8, 80, density=1.5)
    all_points.append(road)

    vehicle_positions = [
        (3, 10 + frame_idx * 0.5, 0),
        (-3, -15, 0),
        (3, 30, 0),
        (-3, -35 + frame_idx * 0.3, 0),
    ]
    for pos in vehicle_positions:
        all_points.append(generate_vehicle(pos))

    pedestrian_positions = [
        (6, 5 + frame_idx * 0.1, 0),
        (-5, 20, 0),
        (7, -10, 0),
    ]
    for pos in pedestrian_positions:
        all_points.append(generate_pedestrian(pos))

    tree_positions = [
        (15, 10, 0), (15, -10, 0), (15, 30, 0),
        (-15, 5, 0), (-15, -20, 0), (-15, 25, 0),
        (20, 0, 0), (-20, 15, 0),
    ]
    for pos in tree_positions:
        all_points.append(generate_tree(pos))

    building_positions = [
        (30, 0, 0), (-30, 10, 0), (30, 30, 0), (-30, -20, 0),
    ]
    for pos in building_positions:
        all_points.append(generate_building(pos, size=(8 + np.random.uniform(-2, 2), 6 + np.random.uniform(-1, 1), 8 + np.random.uniform(0, 8))))

    noise = add_noise_points((-50, 50, -50, 50, -1, 15), n_points=300)
    all_points.append(noise)

    points = np.vstack(all_points)

    intensity = np.zeros(len(points))
    idx = 0
    idx += len(ground)
    intensity[idx:idx+len(road)] = 0.8
    idx += len(road)
    for pos in vehicle_positions:
        v = generate_vehicle(pos)
        intensity[idx:idx+len(v)] = 0.6
        idx += len(v)

    return points, intensity


def save_point_cloud(points, intensity, filepath):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    intensity_norm = (intensity - intensity.min()) / (intensity.max() - intensity.min() + 1e-8)
    colors = np.column_stack([intensity_norm, intensity_norm, intensity_norm])
    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.io.write_point_cloud(filepath, pcd)
    print(f"Saved: {filepath} ({len(points)} points)")


def main():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    os.makedirs(data_dir, exist_ok=True)

    n_frames = 5
    print(f"Generating {n_frames} frames of sample LiDAR data...")

    for i in range(n_frames):
        points, intensity = generate_scene(frame_idx=i)
        filepath = os.path.join(data_dir, f'frame_{i:04d}.pcd')
        save_point_cloud(points, intensity, filepath)

    print(f"\nDone! Generated {n_frames} frames in: {data_dir}")
    print("Run 'python app.py' to start the visualization server.")


if __name__ == '__main__':
    main()
