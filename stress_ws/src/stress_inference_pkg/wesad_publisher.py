#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, String, Int32  
import pickle
import numpy as np
import os
import sys

WESAD_ROOT_DIR = "/Dataset/WESAD/" 
TARGET_SUBJECT = "S3" # Change to the subject as desired

# Sampling frequency of WESAD dataset
BVP_FS = 64
EDA_FS = 4
RATIO = int(BVP_FS / EDA_FS) 

class MockWESADPublisher(Node):
    def __init__(self):
        super().__init__('mock_wesad_publisher')
        
        self.get_logger().info(f"Loading data for {TARGET_SUBJECT}...")
        self.bvp_data, self.eda_data, self.labels = self.load_wesad_data()
        
        if self.bvp_data is None:
            self.get_logger().error("Failed to load data. Node stopping.")
            sys.exit(1)

        self.get_logger().info(f"Data Loaded. BVP samples: {len(self.bvp_data)}, EDA samples: {len(self.eda_data)}")

        START_TIME_SEC = 0
        self.idx_bvp = START_TIME_SEC * BVP_FS
        self.idx_eda = START_TIME_SEC * EDA_FS
        
        if self.idx_bvp >= len(self.bvp_data):
            self.get_logger().error(f"Start time {START_TIME_SEC}s is beyond data length!")
            sys.exit(1)

        self.get_logger().info(f"Data Loaded. Starting from {START_TIME_SEC}s (Idx BVP: {self.idx_bvp})")
        
        # Publishers
        self.pub_bvp = self.create_publisher(Float32, 'bvp_raw', 10)
        self.pub_eda = self.create_publisher(Float32, 'eda_raw', 10)
        self.pub_truth = self.create_publisher(String, 'ground_truth_label', 10)
        self.pub_gt_int = self.create_publisher(Int32, 'ground_truth', 10)

        self.timer = self.create_timer(1.0 / BVP_FS, self.stream_callback)
        self.running = True
        
        self.get_logger().info("STREAMING STARTED")

    def load_wesad_data(self):
        subject_path = os.path.join(WESAD_ROOT_DIR, TARGET_SUBJECT, f"{TARGET_SUBJECT}.pkl")
        if not os.path.exists(subject_path):
            self.get_logger().error(f"File not found: {subject_path}")
            return None, None, None

        try:
            with open(subject_path, 'rb') as file:
                data = pickle.load(file, encoding='latin1')
            bvp = data['signal']['wrist']['BVP'].flatten()
            eda = data['signal']['wrist']['EDA'].flatten()
            original_labels = data['label']
            x_old = np.linspace(0, len(bvp), len(original_labels))
            x_new = np.arange(len(bvp))
            labels_resampled = np.interp(x_new, x_old, original_labels).round().astype(int)
            return bvp, eda, labels_resampled
        except Exception as e:
            self.get_logger().error(f"Error reading pickle: {e}")
            return None, None, None

    def stream_callback(self):
        if not self.running: return

        if self.idx_bvp >= len(self.bvp_data):
            self.get_logger().info("End of the dataset")
            self.running = False
            return

        # Only publish the Label 1 - Baseline; 2 - Stress for analysis
        current_label = self.labels[self.idx_bvp]
        
        if current_label not in [1, 2]:
            remaining_labels = self.labels[self.idx_bvp:]
            valid_offsets = np.where((remaining_labels == 1) | (remaining_labels == 2))[0]
            
            if len(valid_offsets) > 0:
                jump_amount = valid_offsets[0]
                self.idx_bvp += jump_amount
                self.idx_eda = int(self.idx_bvp / RATIO)
                self.get_logger().info(f"Jumping to next valid segment at {self.idx_bvp/64:.1f}s...")
                # Re-check label after jump
                current_label = self.labels[self.idx_bvp]
            else:
                self.get_logger().info("No more Baseline or Stress data found.")
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

        # Publish Ground Truth (String and Int)
        if self.idx_bvp % 64 == 0:
            lbl_name = "Baseline" if current_label == 1 else "Stress"
            
            msg_truth = String()
            msg_truth.data = lbl_name
            self.pub_truth.publish(msg_truth)

            gt_msg = Int32()
            gt_msg.data = 0 if current_label == 1 else 1
            self.pub_gt_int.publish(gt_msg)
            
            if self.idx_bvp % 640 == 0:
                self.get_logger().info(f"Streaming Truth: {lbl_name} (Time: {self.idx_bvp/64:.1f}s)")

        # Standard Increment
        self.idx_bvp += 1

def main(args=None):
    rclpy.init(args=args)
    node = MockWESADPublisher()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()