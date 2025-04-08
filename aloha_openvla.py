import os
import sys

# Monkey patch the import system to prevent Flash Attention from being imported
# This needs to happen before ANY other imports
flash_attn_original = sys.modules.get('flash_attn', None)
sys.modules['flash_attn'] = None  # Block the module entirely

# Now do all other imports
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import torch
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import numpy as np
import cv2
from threading import Lock

class AlohaOpenVLANode(Node):
    def __init__(self):
        super().__init__('aloha_openvla')
        self.latest_image = None
        self.image_lock = Lock()
        
        # Subscribe to the robot camera topic
        self.subscription = self.create_subscription(
            CompressedImage,
            '/camera_high/camera/color/image_rect_raw/compressed',
            self.image_callback,
            10)
        
        # Load Processor & VLA
        self.processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
        self.vla = AutoModelForVision2Seq.from_pretrained(
            "openvla/openvla-7b", 
            torch_dtype=torch.bfloat16, 
            low_cpu_mem_usage=True, 
            trust_remote_code=True
        ).to("cuda:0")
        
        self.get_logger().info("Waiting for camera image...")
    
    def image_callback(self, msg):
        try:
            # Convert compressed image to PIL image
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            pil_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
            
            with self.image_lock:
                self.latest_image = pil_image
        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")
    
    def run_prediction(self, prompt="pick up the black marker"):
        # Wait for image
        while self.latest_image is None and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
        
        with self.image_lock:
            image = self.latest_image
        
        if image is None:
            self.get_logger().error("Failed to get camera image")
            return
        
        # Predict Action (7-DoF; un-normalize for BridgeData V2)
        inputs = self.processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
        action = self.vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
        
        # Output result
        self.get_logger().info(f"Action: {action}")
        return action

def main(args=None):
    rclpy.init(args=args)
    node = AlohaOpenVLANode()
    
    try:
        # Main loop - run continuously
        node.get_logger().info("Starting continuous prediction mode...")
        
        while rclpy.ok():
            # Run prediction and log time
            node.get_logger().info(f"Making prediction at {node.get_clock().now().seconds_nanoseconds()[0]} seconds")
            action = node.run_prediction()
            
            # Process any pending callbacks
            for _ in range(5):  # Process multiple callbacks to ensure we're keeping up
                rclpy.spin_once(node, timeout_sec=0.1)
            
            # Add a sleep to control prediction rate (adjust as needed)
            node.get_logger().info("Sleeping before next prediction...")
            import time
            time.sleep(0.5)  # Sleep for 0.5 seconds between predictions
            
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down...")
    finally:
        node.get_logger().info("Shutting down node...")
        rclpy.shutdown()

if __name__ == '__main__':
    main()