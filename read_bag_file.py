import rosbag

# Function to print all data from a ROS bag
def print_all_rosbag_data(bag_file):
    bag = rosbag.Bag(bag_file)
    for topic, msg, t in bag.read_messages():
        print("Time:", t)
        print("Topic:", topic)
        print("Data:", msg)
    bag.close()

# Example usage
bag_file = 'Issue_ID_4_2024_06_13_07_47_15.bag'  # Replace with your ROS bag file path
print_all_rosbag_data(bag_file)
