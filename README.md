##简介
occupancygrid_generator 是一个 ROS 节点，用于将输入的 3D 点云（如 Livox LiDAR 数据）转换为 2D 占据栅格地图。其核心思想是：对点云聚类 → 对每个聚类计算 2D 凸包 → 将凸包区域投影填充为占据栅格，适用于人形障碍物/物体检测后的环境建模与导航预处理。
##功能特性
- 实时接收 sensor_msgs/PointCloud2 点云数据
- 支持多级点云预处理：
    - 距离半径过滤（radius_filter）
    - Z 轴高度裁剪（z_min, z_max）
    - 体素下采样（voxel_size）
    - 统计离群点剔除
- DBSCAN 聚类（支持大聚类二次分割）
- 使用 scipy.spatial.ConvexHull + skimage.draw.polygon 生成凸包区域并填充栅格
- 降级策略：当点数 < 3 或凸包计算失败时，退化为点投影
- 发布：
    - /livox_occupancy_grid：nav_msgs/OccupancyGrid
    - /cluster_markers：visualization_msgs/MarkerArray（聚类边界框 + 文本标签）
    - /filtered_points：处理后的点云（调试用）
