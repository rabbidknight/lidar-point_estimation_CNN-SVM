import rosbag

# Function to print all data from a ROS bag to a text file
def print_all_rosbag_data_to_file(bag_file, output_file):
    bag = rosbag.Bag(bag_file)
    with open(output_file, 'w') as file:
        for topic, msg, t in bag.read_messages():
            file.write(f"Time: {t}\n")
            file.write(f"Topic: {topic}\n")
            file.write(f"Data: {msg}\n")
    bag.close()

# Example usage
bag_file = 'Issue_ID_4_2024_06_13_07_47_15.bag'  # Replace with your ROS bag file path
output_file = 'output.txt'  # Output file name
print_all_rosbag_data_to_file(bag_file, output_file)
