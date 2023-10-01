import glob
import os
import re
import numpy as np
import open3d as o3d
import pandas as pd

class PointCloudProcessor:
    """
    Class for processing and analyzing point clouds with angle Z.
    """
    def __init__(self, directory, gps_file, point_step=48, frames=75):
        """
        Initializes an instance of the class.
        
        :param directory: Path to the directory containing point cloud files.
        :param gps_file: Path to the GPS data file.
        :param point_step: The number of bytes used for storing each point.
        :param frames: The number of frames for analysis.
        """
        self.directory = directory
        self.gps_file = gps_file
        self.point_step = point_step
        self.frames = frames

    @staticmethod
    def read_point_cloud_from_file(file_path, point_step):
        """
        Loads a point cloud from a file.
        
        :param file_path: Path to the file.
        :param point_step: The number of bytes used for storing each point.
        :return: Point cloud.
        """
        with open(file_path, 'rb') as f:
            binary_data = f.read()

        points_np = np.frombuffer(binary_data, dtype=np.uint8)
        reshaped_points = points_np.reshape(-1, point_step)

        x = np.frombuffer(reshaped_points[:, 0:4].tobytes(), dtype=np.float32)
        y = np.frombuffer(reshaped_points[:, 4:8].tobytes(), dtype=np.float32)
        z = np.frombuffer(reshaped_points[:, 8:12].tobytes(), dtype=np.float32)
        
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(np.array([x, y, z]).T)
        
        # Statistical outlier removal
        point_cloud, _ = point_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        
        return point_cloud

    @staticmethod
    def nearest_neighbors_kdtree(kdtree, query_points):
        """
        Finds the nearest points for the specified query points using k-d tree.
        
        :param kdtree: k-d tree.
        :param query_points: Query points.
        :return: Indices of the nearest points.
        """
        nearest_indices = []
        for point in query_points:
            _, idx, _ = kdtree.search_knn_vector_3d(point, 1)
            nearest_indices.append(idx[0])
        return np.array(nearest_indices)

    def process_point_clouds(self):
        """
        Main method for processing and analyzing point clouds.
        """

        file_names = glob.glob(os.path.join(self.directory, '*.bin'))
        file_names = sorted(file_names, key=lambda x: int(re.search(r'rowid_(\d+)', x).group(1)))
        
        df = pd.read_csv(self.gps_file)

        # Main loop for processing chunks of point cloud files
        for i in range(0, len(file_names), self.frames):
            chunk_files = file_names[i:i+self.frames]
            latitude_list = []
            longitude_list = []
            altitude_list = []

            combined_pcd = self.read_point_cloud_from_file(file_names[0], self.point_step)
            
            for file_name in chunk_files[1:]:
                next_pcd = self.read_point_cloud_from_file(file_name, self.point_step)
                
                # GPS Data Lookup
                df['points_file_path'] = df['points_file_path'].str.split('/').str[-1]
                df['geo_file_path'] = df['geo_file_path'].str.split('/').str[-1]
                file_name_without_extension = os.path.splitext(os.path.basename(file_name))[0]
                new_file_name = f"{file_name_without_extension}.json"

                latitude = df[df['points_file_path']==new_file_name]['latitude']
                longitude = df[df['points_file_path']==new_file_name]['longitude']
                altitude = df[df['points_file_path']==new_file_name]['altitude']
                latitude_list.append(float(latitude))
                longitude_list.append(float(longitude))
                altitude_list.append(float(altitude))

                # Apply ICP algorithm to combine point clouds
                threshold = 1.0  # maximum distance for point "matching"
                trans_init = np.asarray([[1, 0, 0, 0],  # identity transformation
                                        [0, 1, 0, 0],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]])
                
                reg_p2p = o3d.pipelines.registration.registration_icp(
                    next_pcd, combined_pcd, threshold, trans_init,
                    o3d.pipelines.registration.TransformationEstimationPointToPoint())
                
                next_pcd.transform(reg_p2p.transformation)
                
                combined_pcd += next_pcd

            point_cloud = combined_pcd

            average_latitude = latitude_list[-1]
            average_longitude = longitude_list[-1]
            average_altitude = altitude_list[-1]

            # Using RANSAC to find the plane
            plane_model, inliers = point_cloud.segment_plane(distance_threshold=0.1,
                                                            ransac_n=50,
                                                            num_iterations=4000)

            # Extract normal vector of the plane
            a, b, c, d = plane_model

            plane_normal = np.array([a, b, c])

            # Normalize the normal vector
            plane_normal = plane_normal / np.linalg.norm(plane_normal)

            # Find the rotation vector and angle between the plane's normal and the Z-axis
            axis = np.cross(plane_normal, [0, 0, 1])
            axis = axis / np.linalg.norm(axis)
            angle = np.arccos(np.dot(plane_normal, [0, 0, 1]))

            # Create the rotation matrix
            R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)

            # Apply the rotation to the point cloud
            point_cloud.rotate(R, center=(0, 0, 0))

            points_cloud = np.asarray(point_cloud.points)

            z_coordinates = points_cloud[:, 2]  # Extract z-coordinates

            # Determine the minimum and maximum Z values for bucket boundaries
            z_min = np.min(z_coordinates)
            z_max = np.max(z_coordinates)

            # Create 40 buckets
            num_buckets = 40
            buckets = np.linspace(z_min, z_max, num_buckets + 1)

            # Divide points into buckets
            bucket_indices = np.digitize(z_coordinates, buckets)

            # Count the number of points in each bucket
            counts = np.bincount(bucket_indices)

            co= 0 # Initialize the counter for the bucket with the most points

            # Print the number of points and boundaries for each bucket
            for i in range(1, len(buckets)): 
                count = counts[i] if i < len(counts) else 0  # На случай, если нет точек в последнем бакете
                if count > co:
                    co = count
                    lower_bound = buckets[i - 1]
                    upper_bound = buckets[i]

            new_point_cloud = o3d.geometry.PointCloud()

            # Filter the points that are within the bucket with the most points
            filtered_points_array = points_cloud[points_cloud[:, 2] <= upper_bound]

            new_point_cloud.points = o3d.utility.Vector3dVector(filtered_points_array)

            points_cloud = np.asarray(new_point_cloud.points)

            # Clustering
            # Sampling
            sample_indices = np.random.choice(points_cloud.shape[0], size=int(points_cloud.shape[0]*0.1), replace=False)
            sampled_points = points_cloud[sample_indices]

            # Clustering of the sample
            sample_point_cloud = o3d.geometry.PointCloud()
            sample_point_cloud.points = o3d.utility.Vector3dVector(sampled_points)
            labels = np.array(sample_point_cloud.cluster_dbscan(eps=0.1, min_points=20, print_progress=True))

            # Assigning labels to the remaining points
            kdtree = o3d.geometry.KDTreeFlann(sample_point_cloud)
            labels_all = labels[self.nearest_neighbors_kdtree(kdtree, points_cloud)]

            # Shift all labels by 1 to get rid of negative values
            adjusted_labels = labels_all + 1

            # Counting the number of points in each cluster
            cluster_counts = np.bincount(adjusted_labels)

            # Find the index of the cluster with the most points
            largest_cluster_idx = np.argmax(cluster_counts[1:]) + 1

            # Filtering points to only keep points from the largest cluster
            largest_cluster_points = points_cloud[adjusted_labels == largest_cluster_idx]

            z_coords = largest_cluster_points[:, 2]

            mean_z = np.mean(z_coords)
            std_dev = np.std(z_coords)
            threshold = mean_z - 4 * std_dev  # 4 sigma

            # Select points that deviate by 4 sigma from the mean
            outliers = [point for point in largest_cluster_points if point[2] < threshold]

            # Creating a point cloud for the outliers
            outliers_cloud = o3d.geometry.PointCloud()
            outliers_cloud.points = o3d.utility.Vector3dVector(outliers)

            # Applying DBSCAN to identify clusters among the outliers
            with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
                labels = np.array(outliers_cloud.cluster_dbscan(eps=0.06, min_points=3, print_progress=True))
                if np.any(labels != -1):
                    print(average_latitude, average_longitude, average_altitude)
        

if __name__ == "__main__":

    processor = PointCloudProcessor(
        directory='path_to_points',
        gps_file='path_to_gps.csv'
    )

    processor.process_point_clouds()