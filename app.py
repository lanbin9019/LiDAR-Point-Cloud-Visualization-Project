import os
import json
import glob
import hashlib
import numpy as np
import open3d as o3d
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

app = Flask(__name__, static_folder='static')
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
ALLOWED_EXTENSIONS = {'.pcd', '.ply', '.bin', '.xyz', '.pts'}

file_id_map = {}


def get_file_id(filepath):
    file_hash = hashlib.md5(filepath.encode()).hexdigest()[:16]
    file_id_map[file_hash] = filepath
    return file_hash


def get_filepath_from_id(file_id):
    if file_id not in file_id_map:
        return None
    filepath = file_id_map[file_id]
    if not os.path.abspath(filepath).startswith(os.path.abspath(DATA_DIR)):
        return None
    if not os.path.exists(filepath):
        return None
    return filepath


def load_point_cloud(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.bin':
        raw_data = np.fromfile(filepath, dtype=np.float32)
        if len(raw_data) % 4 == 0:
            points = raw_data.reshape(-1, 4)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points[:, :3])
            intensity = points[:, 3]
        elif len(raw_data) % 3 == 0:
            points = raw_data.reshape(-1, 3)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            intensity = np.zeros(len(pcd.points))
        else:
            raise ValueError("Invalid .bin file format")
    elif ext in ('.pcd', '.ply', '.xyz', '.pts'):
        pcd = o3d.io.read_point_cloud(filepath)
        intensity = None
        if pcd.has_colors():
            colors = np.asarray(pcd.colors)
            intensity = colors[:, 0]
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    if intensity is None:
        intensity = np.zeros(len(pcd.points))

    if len(intensity) > 0 and intensity.max() > 1.0:
        intensity = (intensity - intensity.min()) / (intensity.max() - intensity.min() + 1e-8)

    return pcd, intensity


def statistical_filter(pcd, nb_neighbors=20, std_ratio=2.0):
    filtered, ind = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio
    )
    return filtered, ind


def voxel_downsample(pcd, voxel_size=0.1):
    downsampled = pcd.voxel_down_sample(voxel_size=voxel_size)
    return downsampled


def segment_ground(pcd, distance_threshold=0.3, ransac_n=3, num_iterations=1000):
    points = np.asarray(pcd.points)
    if len(points) < ransac_n:
        return np.zeros(len(points), dtype=int)

    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations
    )

    labels = np.zeros(len(points), dtype=int)
    labels[inliers] = 1
    return labels


def process_point_cloud(filepath, voxel_size=0.1, stat_nb=20, stat_ratio=2.0,
                        ground_dist=0.3, ransac_n=3, ransac_iter=1000):
    pcd, intensity = load_point_cloud(filepath)

    filtered_pcd, filter_ind = statistical_filter(pcd, stat_nb, stat_ratio)
    intensity = intensity[filter_ind]

    downsampled_pcd = voxel_downsample(filtered_pcd, voxel_size)

    ds_points = np.asarray(downsampled_pcd.points)
    orig_points = np.asarray(filtered_pcd.points)

    if len(ds_points) < len(intensity):
        from scipy.spatial import cKDTree
        try:
            tree = cKDTree(orig_points)
            _, idx = tree.query(ds_points, k=1)
            intensity = intensity[idx]
        except Exception:
            intensity = np.zeros(len(ds_points))

    ground_labels = segment_ground(
        downsampled_pcd, ground_dist, ransac_n, ransac_iter
    )

    points = np.asarray(downsampled_pcd.points)

    if len(intensity) != len(points):
        intensity = np.zeros(len(points))

    return {
        'points': points.tolist(),
        'intensity': intensity.tolist(),
        'ground_labels': ground_labels.tolist(),
        'num_points': len(points),
        'bounds': {
            'min': points.min(axis=0).tolist(),
            'max': points.max(axis=0).tolist()
        }
    }


@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)


@app.route('/api/files', methods=['GET'])
def list_files():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    extensions = ('*.pcd', '*.ply', '*.bin', '*.xyz', '*.pts')
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(DATA_DIR, ext)))
        files.extend(glob.glob(os.path.join(DATA_DIR, '**', ext), recursive=True))

    file_list = []
    seen = set()
    for f in files:
        rel = os.path.relpath(f, DATA_DIR)
        if rel not in seen:
            seen.add(rel)
            file_id = get_file_id(f)
            file_list.append({
                'name': rel,
                'file_id': file_id,
                'size': os.path.getsize(f)
            })

    return jsonify({'files': file_list})


@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({'error': f'Unsupported format: {ext}. Supported: {", ".join(ALLOWED_EXTENSIONS)}'}), 400

    filename = secure_filename(file.filename)
    if not filename:
        filename = 'uploaded' + ext

    os.makedirs(DATA_DIR, exist_ok=True)
    filepath = os.path.join(DATA_DIR, filename)

    counter = 1
    base, extension = os.path.splitext(filepath)
    while os.path.exists(filepath):
        filepath = f"{base}_{counter}{extension}"
        counter += 1

    file.save(filepath)

    try:
        pcd = o3d.io.read_point_cloud(filepath)
        if len(pcd.points) == 0 and ext == '.bin':
            points = np.fromfile(filepath, dtype=np.float32).reshape(-1, 4)
            if len(points) == 0:
                os.remove(filepath)
                return jsonify({'error': 'File contains no valid point cloud data'}), 400
        elif len(pcd.points) == 0:
            os.remove(filepath)
            return jsonify({'error': 'File contains no valid point cloud data'}), 400
    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': f'Invalid point cloud file: {str(e)}'}), 400

    file_id = get_file_id(filepath)
    return jsonify({
        'success': True,
        'file_id': file_id,
        'filename': os.path.basename(filepath)
    })


@app.route('/api/process', methods=['POST'])
def process_file():
    data = request.get_json()
    file_id = data.get('file_id', '')
    filepath = get_filepath_from_id(file_id)

    if filepath is None:
        return jsonify({'error': 'Invalid file ID'}), 404

    params = {
        'voxel_size': data.get('voxel_size', 0.1),
        'stat_nb': data.get('stat_nb', 20),
        'stat_ratio': data.get('stat_ratio', 2.0),
        'ground_dist': data.get('ground_dist', 0.3),
        'ransac_n': data.get('ransac_n', 3),
        'ransac_iter': data.get('ransac_iter', 1000),
    }

    try:
        result = process_point_cloud(filepath, **params)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/frames', methods=['POST'])
def get_frames():
    data = request.get_json()
    directory = data.get('directory', '')

    if not os.path.isabs(directory):
        directory = os.path.join(DATA_DIR, directory)

    if not os.path.isdir(directory):
        return jsonify({'error': 'Directory not found'}), 404

    extensions = ('*.pcd', '*.ply', '*.bin')
    frames = []
    for ext in extensions:
        frames.extend(glob.glob(os.path.join(directory, ext)))

    frames.sort()
    frame_list = [{'name': os.path.basename(f), 'path': f} for f in frames]
    return jsonify({'frames': frame_list})


if __name__ == '__main__':
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Data directory: {DATA_DIR}")
    print(f"Server starting at http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
