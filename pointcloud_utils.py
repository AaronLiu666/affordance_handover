import open3d as o3d
import numpy as np

def visualize_multiple_point_clouds_with_colors(filenames, colors):
    """
    使用 open3d 同时可视化多个保存在 .npy 文件里的点云数据，并为每个点云设置不同的颜色

    :param filenames: .npy 文件名列表
    :param colors: 颜色列表，每个颜色为一个 (3,) 的列表，表示 RGB 值
    """
    if len(filenames) != len(colors):
        raise ValueError("The number of filenames must match the number of colors.")
    
    point_clouds = []
    
    for filename, color in zip(filenames, colors):
        # 从 .npy 文件中加载点云数据
        points = np.load(filename)
        
        # 创建 open3d 点云对象
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        
        # 设置点云颜色
        point_cloud.paint_uniform_color(color)
        
        # 添加点云到列表
        point_clouds.append(point_cloud)
    
    # 可视化多个点云
    o3d.visualization.draw_geometries(point_clouds)


