#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from pylsl import resolve_streams, StreamInlet
import threading
import numpy as np

EDA_STREAM_TYPE = 'EDA'
BVP_STREAM_TYPE = 'PPGGreen' # Three different green, red, IR; 
# PPGGreen is used because it gives better look at the heart rate

class EmotiBitNode(Node):
    def __init__(self):
        super().__init__('emotibit_publisher')
        
        # Can be used to record the ros bags for future use
        self.pub_bvp = self.create_publisher(Float32, 'bvp_raw', 10)
        self.pub_eda = self.create_publisher(Float32, 'eda_raw', 10)
        
        self.running = True
        
        self.thread_eda = threading.Thread(target=self.stream_eda)
        self.thread_bvp = threading.Thread(target=self.stream_bvp)
        
        self.thread_eda.start()
        self.thread_bvp.start()
    
        self.get_logger().info("Searching for EmotiBit LSL streams...")

    def stream_eda(self):
        streams = resolve_streams(wait_time=5.0)
        inlet_eda = next((StreamInlet(s) for s in streams if s.type() == EDA_STREAM_TYPE), None)
        
        if not inlet_eda:
            self.get_logger().error("EDA Stream not found!")
            return
            
        self.get_logger().info("Connected to EmotiBit EDA.")
        
        while self.running and rclpy.ok():
            sample, timestamp = inlet_eda.pull_sample(timeout=1.0)
            if sample:
                msg = Float32()
                msg.data = float(sample[0])
                self.pub_eda.publish(msg)

    def stream_bvp(self):
        streams = resolve_streams(wait_time=5.0)
        inlet_bvp = next((StreamInlet(s) for s in streams if s.type() == BVP_STREAM_TYPE), None)
        
        if not inlet_bvp:
            self.get_logger().error("PPGGreen Stream not found!")
            return
            
        self.get_logger().info("Connected to EmotiBit PPGGreen.")
        
        while self.running and rclpy.ok():
            sample, timestamp = inlet_bvp.pull_sample(timeout=1.0)
            if sample:
                val = sample[0]
                if not np.isnan(val): 
                    msg = Float32()
                    msg.data = float(val)
                    self.pub_bvp.publish(msg)

    def destroy_node(self):
        self.running = False
        self.thread_eda.join()
        self.thread_bvp.join()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = EmotiBitNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()