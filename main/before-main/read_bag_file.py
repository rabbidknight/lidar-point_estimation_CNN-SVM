import rosbag

# Function to write data from two specific topics in a ROS bag to a file
def write_selected_topics_to_file(bag_file, topics, output_file):
    bag = rosbag.Bag(bag_file)
    with open(output_file, 'w') as file:
        for topic, msg, t in bag.read_messages(topics=topics):
            file.write(f"Time: {t}\n")
            file.write(f"Topic: {topic}\n")
            file.write(f"Data: {msg}\n")
    bag.close()

# Example usage
bag_file = 'Issue_ID_4_2024_06_13_07_47_15.bag'  # Replace with your ROS bag file path
output_file = 'output.txt'  # Output file name
topics = ['/lidar_localizer/lidar_pose', '/lidar_localizer/aligned_cloud']  # List of topics to filter
write_selected_topics_to_file(bag_file, topics, output_file)
