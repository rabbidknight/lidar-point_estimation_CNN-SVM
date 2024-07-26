import struct
import numpy as np
import rosbag
from scipy.spatial.transform import Rotation as R
from transform_the_data import quaternion_to_euler, create_transformation_matrix



def extract_and_transform_data(bag_file, seq_offset):
    """Extract data from a ROS bag and apply transformations based on LiDAR poses."""
    bag = rosbag.Bag(bag_file)
    point_cloud_data = {}
    lidar_transform_data = {}

    # Define byte format for a single point, considering only x, y, z coordinates (each 4 bytes)
    point_format = '<fff'  # little-endian, three floats (x, y, z)
    point_step = 16  # Each point occupies 16 bytes in the data array

    # Extraction step
    for topic, msg, t in bag.read_messages():
        if topic == "/lidar_localizer/aligned_cloud":
            num_points = msg.width  # Total number of points per message
            points = []
            # Decode point data, skipping the irrelevant bytes
            for i in range(num_points):
                offset = i * point_step  # Calculate offset for each point
                point_data = msg.data[offset:offset + 12]  # Extract only the first 12 bytes (x, y, z)
                point = struct.unpack(point_format, point_data)
                points.append(point)
            point_cloud_data[msg.header.seq] = np.array(points)  # Convert list of tuples to numpy array
        elif topic == "/lidar_localizer/lidar_pose":
            position = msg.pose.pose.position
            orientation = msg.pose.pose.orientation
            lidar_transform_data[msg.header.seq] = [
                position.x, position.y, position.z,
                orientation.x, orientation.y, orientation.z, orientation.w
            ]

    bag.close()
    

    # Sync and reshape
    synced_point_clouds = []
    synced_poses = []
    for seq, cloud in point_cloud_data.items():
        corresponding_pose_seq = seq + seq_offset
        if corresponding_pose_seq in lidar_transform_data:
            pose = lidar_transform_data[corresponding_pose_seq]
            synced_point_clouds.append(cloud)
            synced_poses.append(pose)

        else:
            print(f"Skipping point cloud {seq} because corresponding pose is missing.")
    print(len(synced_point_clouds[0]))
    index10=-1

    # Transformation
    transformed_point_clouds = []
    index1=-1
    total_len = 0
    for cloud, pose in zip(synced_point_clouds, synced_poses):
            index1+=1
            print('Current batch:', index1)
            # Ensure cloud length is divisible by 3
            needed_padding = (-len(cloud) % 3)
            if needed_padding:
                cloud = np.pad(cloud, (0, needed_padding), 'constant', constant_values=0)
            reshaped_cloud = cloud.reshape(-1, 3)
            # Get rotation matrix and translation vector from the lidar pose
            if not np.isnan(pose).any():
                euler_angles = quaternion_to_euler([pose[6], pose[3], pose[4], pose[5]])  # w x y z 
                transformation_matrix_lidar = create_transformation_matrix(*euler_angles, pose[0], pose[1], pose[2] ) # Roll, pitch, yaw, tx, ty, tz
                index2=-1
                for point in reshaped_cloud:
                    index2+=1
                    a = np.array([[point[0]], [point[1]], [point[2]], [1]]) # Convert point-cloud xyz to column vector
                    b = np.dot(transformation_matrix_lidar, a) # Apply the transformation
                    transformed_point_clouds.append(b) # Append the transformed point cloud
                    #print('Transform done for index', index1, 'and point', index2)
                    #print('Transformed point:', transformed_point_clouds[-1])
            else:
                raise ValueError("Missing pose data for transformation.")
            #print(total_len)       
    print('ALL DONE')
    return transformed_point_clouds, synced_poses