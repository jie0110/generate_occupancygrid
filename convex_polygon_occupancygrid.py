#!/usr/bin/env python3
# 凸包占据栅格生成

import rospy
import numpy as np
import open3d as o3d
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import Pose, PoseStamped
from scipy.spatial import ConvexHull
from skimage.draw import polygon


class PointCloudToOccupancyGrid:
    def __init__(self):
        rospy.init_node("occupancygrid_generator")
         # 点云预处理参数
        self.z_min = rospy.get_param('~z_min', 0.0)
        self.z_max = rospy.get_param('~z_max', 1.8)
        self.radius_filter = rospy.get_param('~radius_filter',20.0)
        self.voxel_size = rospy.get_param('~voxel_size', 0.1)

        # DBSCAN聚类参数
        self.eps = rospy.get_param('~eps', 0.3)             
        self.min_points = rospy.get_param('~min_points', 3)
        self.max_points = rospy.get_param('~max_points', 200)
        self.min_cluster_size = rospy.get_param('~min_cluster_size', 10)
        # 占据栅格参数
        self.resolution = rospy.get_param('~resolution', 0.05)  # 每个网格的物理尺寸 米/格
        self.width = rospy.get_param('~width', 10.0)            # 米
        self.height = rospy.get_param('~height', 10.0)          # 米
        self.origin_x = rospy.get_param('~origin_x', -5.0)     # 地图原点 X
        self.origin_y = rospy.get_param('~origin_y', -5.0)     # 地图原点 Y
       
        self.grid_width = int(self.width / self.resolution)   
        self.grid_height = int(self.height / self.resolution)

        # 初始化占据栅格，-1: unknown, 0: free, 100: occupied
        self.occupancy_grid = np.full((self.grid_height, self.grid_width), -1, dtype=np.int8)

        self.pc_sub = rospy.Subscriber('/cloud_registered_body', PointCloud2, self.pointcloud_callback)
        self.grid_pub = rospy.Publisher('/livox_occupancy_grid', OccupancyGrid, queue_size=1)
        self.maeker_pub = rospy.Publisher('/cluster_markers', MarkerArray, queue_size=1)
        self.filtered_pc_publisher = rospy.Publisher('/filtered_points', PointCloud2, queue_size=1)
        rospy.loginfo("PointCloud to OccupancyGrid node started.")
        
        # tf坐标转换参数
        self.source_frame = rospy.get_param('~source_frame','camera_init')
        self.target_frame = rospy.get_param('~target_frame','body')
        print(self.source_frame, self.target_frame)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        rospy.loginfo(f"TF listening from {self.source_frame} to {self.target_frame}")
        try:
            self.tf_buffer.lookup_transform(self.target_frame, self.source_frame, rospy.Time(0), rospy.Duration(10.0))
            rospy.loginfo("TF between source and target frame is available.")
        except Exception as e:
            rospy.logwarn(f"TF lookup failed during initialization: {e}")


    def pointcloud_callback(self, msg):
        self.occupancy_grid.fill(-1)
        point_list = list(pc2.read_points(msg, skip_nans=True, field_names=("x","y","z")))
        if len(point_list)==0:
            rospy.logwarn("Received empty pointcloud")
            return
        points = np.array(point_list)
        clusters = self.process_pointcloud(points)

        if clusters:
            self.publish_marker(clusters, msg.header)
            rospy.loginfo(f"Detected {len(clusters)} clusters")
        # 用聚类凸包区域填充占据栅格
        for cluster in clusters:
            cluster_points = cluster['points']
            if len(cluster_points) < 3:
                # 点数太少无法构成凸包，直接投影点
                for pt in cluster_points:
                    grid_x = int((pt[0] - self.origin_x) / self.resolution)
                    grid_y = int((pt[1] - self.origin_y) / self.resolution)
                    if 0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height:
                        self.occupancy_grid[grid_y, grid_x] = 100
                continue
            # 只取x, y用于凸包
            points_2d = cluster_points[:, :2]
            try:
                hull = ConvexHull(points_2d)
                hull_pts = points_2d[hull.vertices]
                # 转为栅格索引
                grid_xs = ((hull_pts[:, 0] - self.origin_x) / self.resolution).astype(int)
                grid_ys = ((hull_pts[:, 1] - self.origin_y) / self.resolution).astype(int)
                # 用skimage画多边形
                rr, cc = polygon(grid_ys, grid_xs, self.occupancy_grid.shape)
                self.occupancy_grid[rr, cc] = 100
            except Exception as e:
                rospy.logwarn(f"凸包计算失败: {e}")
                # 失败时退化为点投影
                for pt in cluster_points:
                    grid_x = int((pt[0] - self.origin_x) / self.resolution)
                    grid_y = int((pt[1] - self.origin_y) / self.resolution)
                    if 0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height:
                        self.occupancy_grid[grid_y, grid_x] = 100
        self.publish_grid(msg.header)

    def compute_bounding_box(self, points):
        """计算点云的边界框"""
        min_bound = np.min(points, axis=0)
        max_bound = np.max(points, axis=0)
        return {
            'min': min_bound,
            'max': max_bound,
            'size': max_bound - min_bound
        }

    def process_pointcloud(self, points):
        """使用Open3D处理点云并进行聚类"""
        try:
            # 创建Open3D点云对象
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            # 1. 距离过滤（移除过远的点）
            center = np.array([0.0, 0.0, 0.0])
            distances = np.linalg.norm(np.asarray(pcd.points) - center, axis=1)
            pcd = pcd.select_by_index(np.where(distances < self.radius_filter)[0])
            # 2. Z轴范围过滤
            points_np = np.asarray(pcd.points)
            z_mask = (points_np[:, 2] > self.z_min) & (points_np[:, 2] < self.z_max)
            pcd = pcd.select_by_index(np.where(z_mask)[0])
            # 3. 体素网格化
            if self.voxel_size > 0.0:
                pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)
            # 4. 移除离群点
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            # 发布过滤后的点云
            self.publish_filtered_pointcloud(pcd)
            # 5. DBSCAN聚类
            labels = np.array(pcd.cluster_dbscan(eps=self.eps, min_points=self.min_points))
            # 处理聚类结果
            clusters = []
            max_label = labels.max()
            for i in range(max_label + 1):
                cluster_indices = np.where(labels == i)[0]
                # 过滤小聚类
                if len(cluster_indices) < self.min_cluster_size:
                    continue  
                # 大聚类进行二次分割
                if len(cluster_indices) > self.max_points:
                    # 根据法线方向进行聚类
                    # normals = np.asarray(pcd.normals)[cluster_indices]
                    
                    sub_labels = np.array(pcd.select_by_index(cluster_indices).cluster_dbscan(eps=self.eps, min_points=self.min_points))
                    sub_max_label = sub_labels.max()
                    for j in range(sub_max_label + 1):
                        sub_cluster_indices = cluster_indices[np.where(sub_labels == j)[0]]
                        if len(sub_cluster_indices) < self.min_cluster_size:
                            continue
                        sub_cluster_points = np.asarray(pcd.points)[sub_cluster_indices]
                        clusters.append({
                            'points': sub_cluster_points,
                            'center': np.mean(sub_cluster_points, axis=0),
                            'size': len(sub_cluster_indices),
                            'bbox': self.compute_bounding_box(sub_cluster_points)
                        })
                    continue
                cluster_points = np.asarray(pcd.points)[cluster_indices]
                clusters.append({
                    'points': cluster_points,
                    'center': np.mean(cluster_points, axis=0),
                    'size': len(cluster_indices),
                    'bbox': self.compute_bounding_box(cluster_points)
                })
            return clusters
            
        except Exception as e:
            rospy.logerr(f"Open3D处理错误: {str(e)}")
            return []


    def publish_grid(self, header):
        "发布占据栅格地图"
        grid_msg = OccupancyGrid()
        grid_msg.header = header
        grid_msg.header.frame_id = "camera_init"  
        grid_msg.info.resolution = self.resolution
        grid_msg.info.width = self.grid_width
        grid_msg.info.height = self.grid_height

        # 原点在body系下
        origin_pose = PoseStamped()
        origin_pose.header.frame_id = "body"
        origin_pose.header.stamp = rospy.Time(0)
        origin_pose.pose.position.x = self.origin_x
        origin_pose.pose.position.y = self.origin_y
        origin_pose.pose.position.z = 0.0
        origin_pose.pose.orientation.w = 1.0

        try:
            # 获取变换矩阵
            trans = self.tf_buffer.lookup_transform(self.source_frame, self.target_frame, rospy.Time(0), rospy.Duration(0.2))
            # 转换原点
            map_origin = tf2_geometry_msgs.do_transform_pose(origin_pose, trans)
            grid_msg.info.origin = map_origin.pose
        
        except Exception as e:
            rospy.logwarn(f"TF变换失败，使用body系原点: {e}")
            grid_msg.info.origin.position.x = self.origin_x
            grid_msg.info.origin.position.y = self.origin_y
            grid_msg.info.origin.position.z = 0.0
            grid_msg.info.origin.orientation.w = 1.0

        grid_msg.data = self.occupancy_grid.flatten().tolist()
        self.grid_pub.publish(grid_msg)

    def publish_filtered_pointcloud(self, pcd):
        """发布过滤后的点云"""
        try:
            points = np.asarray(pcd.points)
            if len(points) == 0:
                return
                
            header = Header()
            header.stamp = rospy.Time.now()
            header.frame_id = "body"  
            
            # 创建PointCloud2消息
            fields = [
                pc2.PointField('x', 0, pc2.PointField.FLOAT32, 1),
                pc2.PointField('y', 4, pc2.PointField.FLOAT32, 1),
                pc2.PointField('z', 8, pc2.PointField.FLOAT32, 1),
            ]
            
            pc_msg = pc2.create_cloud(header, fields, points)
            self.filtered_pc_publisher.publish(pc_msg)
            
        except Exception as e:
            rospy.logerr(f"发布过滤点云错误: {str(e)}")

    def publish_marker(self, clusters, header):
        """发布聚类结果为MarkerArray"""
        try:
            marker_array = MarkerArray()
            
            for i, cluster in enumerate(clusters):
                # 创建边界框标记
                marker = Marker()
                marker.header = header
                marker.header.stamp = rospy.Time.now()
                marker.ns = "clusters"
                marker.id = i
                marker.type = Marker.CUBE
                marker.action = Marker.ADD
                
                # 设置边界框位置和大小
                center = cluster['center']
                size = cluster['bbox']['size']
                
                marker.pose.position.x = center[0]
                marker.pose.position.y = center[1]
                marker.pose.position.z = center[2]
                marker.pose.orientation.w = 1.0
                
                marker.scale.x = max(size[0], 0.1)
                marker.scale.y = max(size[1], 0.1)
                marker.scale.z = max(size[2], 0.1)
                
                # 设置颜色（根据聚类ID）
                colors = [
                    (1.0, 0.0, 0.0),  # 红
                    (0.0, 1.0, 0.0),  # 绿
                    (0.0, 0.0, 1.0),  # 蓝
                    (1.0, 1.0, 0.0),  # 黄
                    (1.0, 0.0, 1.0),  # 紫
                    (0.0, 1.0, 1.0),  # 青
                ]
                color = colors[i % len(colors)]
                marker.color.r = color[0]
                marker.color.g = color[1]
                marker.color.b = color[2]
                marker.color.a = 0.5
                
                marker.lifetime = rospy.Duration(0.5)
                marker_array.markers.append(marker)
                
                # 创建文本标记显示聚类信息
                text_marker = Marker()
                text_marker.header = header
                text_marker.header.stamp = rospy.Time.now()
                text_marker.ns = "cluster_text"
                text_marker.id = i + 1000
                text_marker.type = Marker.TEXT_VIEW_FACING
                text_marker.action = Marker.ADD
                
                text_marker.pose.position.x = center[0]
                text_marker.pose.position.y = center[1]
                text_marker.pose.position.z = center[2] + size[2]/2 + 0.5
                text_marker.pose.orientation.w = 1.0
                
                text_marker.scale.z = 0.5
                text_marker.color.r = 1.0
                text_marker.color.g = 1.0
                text_marker.color.b = 1.0
                text_marker.color.a = 1.0
                
                text_marker.text = f"Cluster {i}\nPoints: {cluster['size']}"
                text_marker.lifetime = rospy.Duration(0.5)
                marker_array.markers.append(text_marker)
            
            # 清除旧的标记
            if len(clusters) < getattr(self, 'last_cluster_count', 0):
                for i in range(len(clusters), self.last_cluster_count):
                    delete_marker = Marker()
                    delete_marker.header = header
                    delete_marker.ns = "clusters"
                    delete_marker.id = i
                    delete_marker.action = Marker.DELETE
                    marker_array.markers.append(delete_marker)
                    
                    delete_text = Marker()
                    delete_text.header = header
                    delete_text.ns = "cluster_text"
                    delete_text.id = i + 1000
                    delete_text.action = Marker.DELETE
                    marker_array.markers.append(delete_text)
            
            self.last_cluster_count = len(clusters)
            self.maeker_pub.publish(marker_array)
            
        except Exception as e:
            rospy.logerr(f"发布聚类标记错误: {str(e)}")


if __name__ == '__main__':
    node = PointCloudToOccupancyGrid()
    rospy.spin()