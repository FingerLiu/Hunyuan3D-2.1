#!/usr/bin/env python3
"""
SEM 数据预处理脚本
将 SEM 采集的数据转换为 Hunyuan3D-2.1 训练所需的格式

输入:
- camera_parameters.json: SEM 相机参数（包含 rotation 矩阵）
- *.stl: STL 格式的 3D 模型
- *.png: SEM 图像

输出:
- transforms.json: 训练所需的相机参数格式
- mesh.ply: PLY 格式的 mesh
- xxx_sdf.npz: SDF 采样数据
- xxx_surface.npz: 表面采样数据
"""

import os
import sys
import json
import argparse
import shutil
import numpy as np
import trimesh

# 添加 watertight 工具路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TOOLS_DIR = os.path.join(os.path.dirname(os.path.dirname(SCRIPT_DIR)), 'tools')
sys.path.insert(0, os.path.join(TOOLS_DIR, 'watertight'))

try:
    import igl
    from watertight_and_sample import Watertight, SampleMesh, normalize_to_unit_box
    HAS_IGL = True
except ImportError:
    HAS_IGL = False
    print("[WARNING] igl 未安装，无法生成 SDF 数据。请安装: pip install libigl")


def parse_matrix_string(matrix_str: str) -> np.ndarray:
    """解析字符串格式的矩阵为 numpy 数组"""
    # 使用 eval 解析矩阵字符串 (例如 "[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]")
    matrix = eval(matrix_str)
    return np.array(matrix)


def parse_vector_string(vec_str: str) -> np.ndarray:
    """解析字符串格式的向量为 numpy 数组"""
    vec = eval(vec_str)
    return np.array(vec)


def rotation_matrix_to_euler_angles(R: np.ndarray) -> tuple:
    """
    从旋转矩阵提取方位角(azimuth)和仰角(elevation)
    假设相机朝向原点
    
    Returns:
        azimuth: 方位角 (弧度)
        elevation: 仰角 (弧度)
    """
    # 从旋转矩阵提取相机方向
    # 假设相机的前向方向是 -Z 轴
    # 旋转矩阵的第三列是相机的 Z 轴方向
    
    # 计算相机位置（假设相机看向原点）
    # 对于 SEM 数据，rotation 矩阵描述的是视图旋转
    
    # 从旋转矩阵计算欧拉角
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    
    singular = sy < 1e-6
    
    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])  # roll
        y = np.arctan2(-R[2, 0], sy)       # pitch (elevation)
        z = np.arctan2(R[1, 0], R[0, 0])   # yaw (azimuth)
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0
    
    return z, y  # azimuth, elevation


def build_transform_matrix(R: np.ndarray, t: np.ndarray, cam_dis: float = 1.5) -> list:
    """
    构建 4x4 变换矩阵
    
    Args:
        R: 3x3 旋转矩阵
        t: 3D 平移向量
        cam_dis: 相机距离
    
    Returns:
        4x4 变换矩阵 (list 格式)
    """
    # 构建 4x4 变换矩阵
    transform = np.eye(4)
    transform[:3, :3] = R
    
    # 如果 translation 为 0，从旋转矩阵计算相机位置
    if np.allclose(t, 0):
        # 相机位置在 Z 轴方向，距离为 cam_dis
        # 然后用旋转矩阵变换
        cam_pos = R.T @ np.array([0, 0, cam_dis])
        transform[:3, 3] = cam_pos
    else:
        transform[:3, 3] = t
    
    return transform.tolist()


def convert_camera_params(
    input_json: str,
    output_json: str,
    cam_dis: float = 1.5,
    camera_angle_x: float = 0.8,  # ~45° FOV
    proj_type: int = 0,  # 0 = perspective, 1 = orthographic
    scale: float = 1.0,
    offset: list = None
):
    """
    将 camera_parameters.json 转换为 transforms.json 格式
    
    Args:
        input_json: 输入的 camera_parameters.json 路径
        output_json: 输出的 transforms.json 路径
        cam_dis: 相机距离 (默认 1.5)
        camera_angle_x: 相机 FOV (弧度)
        proj_type: 投影类型
        scale: 缩放因子
        offset: 偏移量
    """
    with open(input_json, 'r') as f:
        data = json.load(f)
    
    if offset is None:
        offset = [0.0, 0.0, 0.0]
    
    output_data = {
        "aabb": [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        "scale": scale,
        "offset": offset,
        "frames": []
    }
    
    # 获取输入目录，用于修正文件路径
    input_dir = os.path.dirname(input_json)
    
    for i, frame in enumerate(data.get("frames", [])):
        # 解析旋转矩阵
        R = parse_matrix_string(frame["rotation"])
        t = parse_vector_string(frame["translation"])
        
        # 计算 azimuth 和 elevation
        azimuth, elevation = rotation_matrix_to_euler_angles(R)
        
        # 构建变换矩阵
        transform_matrix = build_transform_matrix(R, t, cam_dis)
        
        # 获取文件名
        file_path = frame["file_path"]
        # 修正文件路径：移除前导空格和 ./
        file_path = file_path.strip()
        if file_path.startswith("./"):
            file_path = file_path[2:]
        if file_path.startswith("/"):
            file_path = os.path.basename(file_path)
        
        # 创建输出帧数据
        frame_data = {
            "file_path": file_path,
            "camera_angle_x": camera_angle_x,
            "proj_type": proj_type,
            "azimuth": azimuth,
            "elevation": elevation,
            "cam_dis": cam_dis,
            "transform_matrix": transform_matrix
        }
        
        output_data["frames"].append(frame_data)
    
    # 保存输出
    with open(output_json, 'w') as f:
        json.dump(output_data, f, indent=4)
    
    print(f"[INFO] 已转换 {len(output_data['frames'])} 个相机视图")
    print(f"[INFO] 保存到: {output_json}")
    
    return output_data


def stl_to_ply(input_stl: str, output_ply: str, normalize: bool = True):
    """
    将 STL 转换为 PLY 格式
    
    Args:
        input_stl: 输入 STL 文件路径
        output_ply: 输出 PLY 文件路径
        normalize: 是否归一化到单位立方体
    """
    print(f"[INFO] 加载 STL: {input_stl}")
    mesh = trimesh.load(input_stl)
    
    if normalize:
        # 归一化到 [-0.5, 0.5]^3
        vertices = mesh.vertices
        center = (vertices.max(axis=0) + vertices.min(axis=0)) / 2
        scale = (vertices.max(axis=0) - vertices.min(axis=0)).max()
        
        mesh.vertices = (vertices - center) / scale
        print(f"[INFO] 归一化: center={center}, scale={scale}")
    
    mesh.export(output_ply)
    print(f"[INFO] 已保存 PLY: {output_ply}")
    
    return mesh, scale if normalize else 1.0, center.tolist() if normalize else [0, 0, 0]


def generate_sdf_and_surface(input_mesh: str, output_prefix: str):
    """
    从 mesh 生成 SDF 和 surface 采样数据
    
    Args:
        input_mesh: 输入 mesh 文件路径 (支持 PLY, OBJ 等)
        output_prefix: 输出文件前缀
    """
    if not HAS_IGL:
        print("[ERROR] 需要安装 igl 库才能生成 SDF 数据")
        print("[ERROR] 请运行: pip install libigl")
        return None, None
    
    print(f"[INFO] 加载 mesh: {input_mesh}")
    
    # 读取 mesh
    V, F = igl.read_triangle_mesh(input_mesh)
    print(f"[INFO] 顶点数: {V.shape[0]}, 面数: {F.shape[0]}")
    
    # 归一化
    V = normalize_to_unit_box(V)
    
    # 水密化处理
    print("[INFO] 正在进行水密化处理...")
    mc_verts, mc_faces = Watertight(V, F)
    print(f"[INFO] 水密化后: 顶点数 {mc_verts.shape[0]}, 面数 {mc_faces.shape[0]}")
    
    # 采样
    print("[INFO] 正在采样表面和 SDF...")
    surface_data, sdf_data = SampleMesh(mc_verts, mc_faces)
    
    # 保存
    parent_folder = os.path.dirname(output_prefix)
    if parent_folder:
        os.makedirs(parent_folder, exist_ok=True)
    
    surface_path = f'{output_prefix}_surface.npz'
    sdf_path = f'{output_prefix}_sdf.npz'
    
    np.savez(surface_path, **surface_data)
    print(f"[INFO] 已保存 surface 数据: {surface_path}")
    
    np.savez(sdf_path, **sdf_data)
    print(f"[INFO] 已保存 SDF 数据: {sdf_path}")
    
    return surface_data, sdf_data


def preprocess_sem_data(
    input_dir: str,
    output_dir: str,
    stl_file: str = None,
    camera_json: str = None,
    cam_dis: float = 1.5,
    camera_angle_x: float = 0.8,
    sample_name: str = None
):
    """
    完整的 SEM 数据预处理流程
    
    Args:
        input_dir: 输入目录（包含图片和 camera_parameters.json）
        output_dir: 输出目录
        stl_file: STL 文件路径
        camera_json: 相机参数 JSON 文件路径
        cam_dis: 相机距离
        camera_angle_x: 相机 FOV
        sample_name: 样本名称（用于输出命名）
    """
    # 确定输入文件
    if camera_json is None:
        camera_json = os.path.join(input_dir, 'camera_parameters.json')
    
    if not os.path.exists(camera_json):
        raise FileNotFoundError(f"找不到相机参数文件: {camera_json}")
    
    # 确定样本名称
    if sample_name is None:
        sample_name = os.path.basename(os.path.normpath(input_dir))
    
    # 创建输出目录结构
    render_cond_dir = os.path.join(output_dir, 'render_cond')
    geo_data_dir = os.path.join(output_dir, 'geo_data')
    os.makedirs(render_cond_dir, exist_ok=True)
    os.makedirs(geo_data_dir, exist_ok=True)
    
    print("=" * 60)
    print(f"SEM 数据预处理")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print("=" * 60)
    
    # Step 1: 复制图片到 render_cond
    print("\n[Step 1] 复制图片...")
    png_files = [f for f in os.listdir(input_dir) if f.endswith('.png')]
    for png_file in sorted(png_files):
        src = os.path.join(input_dir, png_file)
        dst = os.path.join(render_cond_dir, png_file)
        shutil.copy2(src, dst)
    print(f"[INFO] 已复制 {len(png_files)} 张图片")
    
    # Step 2: 转换 STL 到 PLY
    mesh_ply_path = os.path.join(render_cond_dir, 'mesh.ply')
    scale = 1.0
    offset = [0.0, 0.0, 0.0]
    
    if stl_file and os.path.exists(stl_file):
        print(f"\n[Step 2] 转换 STL 到 PLY...")
        _, scale, offset = stl_to_ply(stl_file, mesh_ply_path, normalize=True)
    else:
        print(f"\n[Step 2] 未提供 STL 文件，跳过 mesh 转换")
    
    # Step 3: 转换相机参数
    print(f"\n[Step 3] 转换相机参数...")
    transforms_json_path = os.path.join(render_cond_dir, 'transforms.json')
    convert_camera_params(
        camera_json,
        transforms_json_path,
        cam_dis=cam_dis,
        camera_angle_x=camera_angle_x,
        scale=scale,
        offset=offset
    )
    
    # Step 4: 生成 SDF 和 surface 数据
    if os.path.exists(mesh_ply_path):
        print(f"\n[Step 4] 生成 SDF 和 surface 数据...")
        output_prefix = os.path.join(geo_data_dir, sample_name)
        generate_sdf_and_surface(mesh_ply_path, output_prefix)
    else:
        print(f"\n[Step 4] 没有 mesh 文件，跳过 SDF 生成")
    
    print("\n" + "=" * 60)
    print("预处理完成！")
    print(f"输出目录结构:")
    print(f"  {output_dir}/")
    print(f"  ├── render_cond/")
    print(f"  │   ├── *.png (图片)")
    print(f"  │   ├── mesh.ply")
    print(f"  │   └── transforms.json")
    print(f"  └── geo_data/")
    print(f"      ├── {sample_name}_sdf.npz")
    print(f"      └── {sample_name}_surface.npz")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='SEM 数据预处理脚本')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='输入目录（包含 PNG 图片和 camera_parameters.json）')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='输出目录')
    parser.add_argument('--stl', type=str, default=None,
                       help='STL 文件路径')
    parser.add_argument('--camera_json', type=str, default=None,
                       help='相机参数 JSON 文件路径（默认为 input_dir/camera_parameters.json）')
    parser.add_argument('--cam_dis', type=float, default=1.5,
                       help='相机距离（默认 1.5）')
    parser.add_argument('--camera_angle_x', type=float, default=0.8,
                       help='相机 FOV（弧度，默认 0.8 ≈ 45°）')
    parser.add_argument('--name', type=str, default=None,
                       help='样本名称（用于输出命名）')
    
    # 支持单独的转换操作
    parser.add_argument('--only_transforms', action='store_true',
                       help='仅转换相机参数')
    parser.add_argument('--only_stl', action='store_true',
                       help='仅转换 STL 到 PLY')
    parser.add_argument('--only_sdf', action='store_true',
                       help='仅生成 SDF 数据')
    
    args = parser.parse_args()
    
    if args.only_transforms:
        # 仅转换相机参数
        input_json = args.camera_json or os.path.join(args.input_dir, 'camera_parameters.json')
        output_json = os.path.join(args.output_dir, 'transforms.json')
        os.makedirs(args.output_dir, exist_ok=True)
        convert_camera_params(input_json, output_json, args.cam_dis, args.camera_angle_x)
    elif args.only_stl:
        # 仅转换 STL
        if not args.stl:
            print("[ERROR] 需要提供 --stl 参数")
            return
        output_ply = os.path.join(args.output_dir, 'mesh.ply')
        os.makedirs(args.output_dir, exist_ok=True)
        stl_to_ply(args.stl, output_ply)
    elif args.only_sdf:
        # 仅生成 SDF
        mesh_path = args.stl or os.path.join(args.input_dir, 'mesh.ply')
        name = args.name or 'sample'
        output_prefix = os.path.join(args.output_dir, name)
        os.makedirs(args.output_dir, exist_ok=True)
        generate_sdf_and_surface(mesh_path, output_prefix)
    else:
        # 完整预处理流程
        preprocess_sem_data(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            stl_file=args.stl,
            camera_json=args.camera_json,
            cam_dis=args.cam_dis,
            camera_angle_x=args.camera_angle_x,
            sample_name=args.name
        )


if __name__ == '__main__':
    main()
