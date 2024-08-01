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
    last_timestamp = None  # Track the last timestamp
    num_points = [] # Track the current number of points in the point cloud

    # Extraction step
    for topic, msg, t in bag.read_messages():
        if topic == "/lidar_localizer/aligned_cloud":
            #if last_timestamp != msg.header.stamp:
                #last_timestamp = msg.header.stamp
                current_num_points = msg.width
                num_points.append(current_num_points) #dont mind this, just for tracking the number of points
                points = []
                for i in range(current_num_points):
                    offset = i * point_step
                    if offset + 12 <= len(msg.data):
                        point_data = msg.data[offset:offset + 12]
                        point = struct.unpack(point_format, point_data)
                        points.append(point)
                    else:
                        print("Data buffer does not have enough bytes for a full point extraction.")
                point_cloud_data[msg.header.seq] = np.array(points)  # Convert list of tuples to numpy array
        elif topic == "/lidar_localizer/lidar_pose":
            position = msg.pose.pose.position
            orientation = msg.pose.pose.orientation
            lidar_transform_data[msg.header.seq] = [
                position.x, position.y, position.z,
                orientation.x, orientation.y, orientation.z, orientation.w
            ]

    bag.close()
    print("Point clouds extracted:", len(point_cloud_data))
    print("Lidar poses extracted:", len(lidar_transform_data))


    # Sync and reshape
    synced_point_clouds = []
    synced_point_clouds2 = []   #This is just for catching the total 1D length, 
    #to be used in the training. Dont mind this.
    synced_poses = []
    synced_poses2 = [] #This is just for catching the total 1D length,
    #to be used in the training. Dont mind this.

    for seq, cloud in point_cloud_data.items():
        corresponding_pose_seq = seq + seq_offset
        if corresponding_pose_seq in lidar_transform_data:                
            pose = lidar_transform_data[corresponding_pose_seq]
            if len(cloud) % 3 != 0:
                excess = len(cloud) % 3
                cloud = cloud[:-excess]  # Remove the last few points to make the length divisible by 3
            for point in cloud:
                if ((point[1]<-75) and (point[1]>-90)): #thresholding the y values
                    pose_array_for_training = np.array([pose])  # Clone the pose for each point in the cloud
                    synced_poses2.extend(pose_array_for_training) # Flatten the list of poses, for training
                    synced_point_clouds.append(point)  
                    synced_point_clouds2.extend(point) # Flatten the list of points, for training
                    synced_poses.append(pose)
        else:
            print(f"Skipping point cloud {seq} because corresponding pose is missing.")
            print("also ending the program because go fix it")
            exit()
    '''
    print("length of synced_point_clouds", len(synced_point_clouds))
    print("length of synced_point_clouds2", len(synced_point_clouds2))
    print("length of synced_poses", len(synced_poses))
    print("length of synced_poses2", len(synced_poses2))
    '''

    # Transformation 
    print('Transforming point clouds...')
    transformed_point_clouds = []
    transformed_poses = []

    index1=-1
    for cloud, pose in zip(synced_point_clouds, synced_poses):
            index1+=1
            #print('Current batch transforming:', index1)

            #original_cloud_size = len(cloud)  # Number of points in the original cloud
            #transformed_points_count = 0  # This will count how many points are actually processed

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
                    #transformed_points_count += 1

            else:
                raise ValueError("Missing pose data for transformation.")
            #print(f"Processed cloud with {original_cloud_size} points, transformed into {transformed_points_count} points")
    print('Transformation of point clouds done')
    print('Transforming poses...')
    for pose in synced_poses2:
        # Transform poses with themselves to set new offset
        euler_angles = quaternion_to_euler([pose[6], pose[3], pose[4], pose[5]])  # w x y z 
        transformation_matrix_lidar = create_transformation_matrix(*euler_angles, pose[0], pose[1], pose[2] ) # Roll, pitch, yaw, tx, ty, tz
        pose_vec = np.array([[pose[0]], [pose[1]], [pose[2]], [1]])
        transformed_poses.append(np.dot(transformation_matrix_lidar, pose_vec))
        

    print('ALL DONE, now onto training!')

    return transformed_point_clouds, transformed_poses, num_points