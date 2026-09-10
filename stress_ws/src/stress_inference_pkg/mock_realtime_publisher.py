#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, String
import pickle
import os
import sys

# Sampling frequency of EDA and BVP signals coming from Emotibit sensor
BVP_FS = 25
EDA_FS = 15
RATIO = int(BVP_FS / EDA_FS) 

class MockRealtimePublisher(Node):
    def __init__(self):
        super().__init__('mock_realtime_publisher')
        
        self.get_logger().info(f"Loading data for")
        self.bvp_data, self.eda_data = self.load_realtime_data()
        
        if self.bvp_data is None:
            self.get_logger().error("Failed to load data. Node stopping.")
            sys.exit(1)

        self.get_logger().info(f"Data Loaded. BVP samples: {len(self.bvp_data)}, EDA samples: {len(self.eda_data)}")

        # Publishers
        self.pub_bvp = self.create_publisher(Float32, 'bvp_raw', 10)
        self.pub_eda = self.create_publisher(Float32, 'eda_raw', 10)
        self.pub_truth = self.create_publisher(String, 'ground_truth_label', 10)

        self.timer = self.create_timer(1.0 / BVP_FS, self.stream_callback)
        
        self.idx_bvp = 0
        self.idx_eda = 0
        self.running = True
        
        self.get_logger().info("STREAMING STARTED")
        self.get_logger().info("Run the inference node in a separate terminal")

    def load_realtime_data(self):
        """Loads pickle and aligns labels."""
        # Edit according to the file created by pickle_creator.py
        subject_path = '/stress_ws/combined_data.pkl' 

        if not os.path.exists(subject_path):
            self.get_logger().error(f"File not found: {subject_path}")
            return None, None, None

        try:
            with open(subject_path, 'rb') as file:
                data = pickle.load(file, encoding='latin1')
            
            bvp = data['bvp_raw'].flatten()
            eda = data['eda_raw'].flatten()
            
            return bvp, eda 

        except Exception as e:
            self.get_logger().error(f"Error reading pickle: {e}")
            return None, None, None

    def stream_callback(self):
        if not self.running: return

        if self.idx_bvp >= len(self.bvp_data) or self.idx_eda >= len(self.eda_data):
            self.get_logger().info("End of the dataset")
            self.running = False
            return

        # Publish BVP (Every tick)
        msg_bvp = Float32()
        msg_bvp.data = float(self.bvp_data[self.idx_bvp])
        self.pub_bvp.publish(msg_bvp)

        # Publish EDA (Achieve same frequency as BVP)
        if self.idx_bvp % RATIO == 0:
            if self.idx_eda < len(self.eda_data):
                msg_eda = Float32()
                msg_eda.data = float(self.eda_data[self.idx_eda])
                self.pub_eda.publish(msg_eda)
                self.idx_eda += 1

        
def main(args=None):
    rclpy.init(args=args)
    node = MockRealtimePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
