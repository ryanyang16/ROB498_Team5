import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
import cv2
from cv_bridge import CvBridge
import yaml
import os

class IMX219Node(Node):
    def __init__(self):
        super().__init__('imx219_camera')
        
        # Publish both the raw image and the calibration info
        self.image_pub = self.create_publisher(Image, '/camera/bottom/image_raw', 10)
        self.info_pub = self.create_publisher(CameraInfo, '/camera/bottom/camera_info', 10)
        
        self.timer = self.create_timer(0.033, self.timer_callback) # ~30fps
        self.bridge = CvBridge()
        
        # Load the calibration data
        yaml_path = os.path.expanduser('~/ros2_ws/calibrationdata/ost.yaml')
        self.camera_info_msg = self.parse_calibration_yaml(yaml_path)

        # GStreamer pipeline
        pipeline = (
            "nvarguscamerasrc ! "
            "video/x-raw(memory:NVMM), width=640, height=480, format=NV12, framerate=30/1 ! "
            "nvvidconv ! video/x-raw, format=BGRx ! "
            "videoconvert ! video/x-raw, format=BGR ! appsink"
        )
        self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        self.get_logger().info("IMX219 Camera Node Started (Calibrated)")

    def parse_calibration_yaml(self, yaml_file):
        try:
            with open(yaml_file, "r") as file_handle:
                calib_data = yaml.safe_load(file_handle)

            cam_info = CameraInfo()
            cam_info.width = calib_data["image_width"]
            cam_info.height = calib_data["image_height"]
            cam_info.k = calib_data["camera_matrix"]["data"]
            cam_info.d = calib_data["distortion_coefficients"]["data"]
            cam_info.r = calib_data["rectification_matrix"]["data"]
            cam_info.p = calib_data["projection_matrix"]["data"]
            cam_info.distortion_model = calib_data["camera_name"]
            
            # The frame_id must match between the image and the info!
            cam_info.header.frame_id = "camera_bottom_optical_frame" 
            return cam_info
            
        except Exception as e:
            self.get_logger().error(f"Failed to load calibration YAML: {e}")
            return CameraInfo() # Return empty if it fails so the node doesn't crash

    def timer_callback(self):
        ret, frame = self.cap.read()
        if ret:
            # Prepare Image message
            img_msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
            img_msg.header.frame_id = "camera_bottom_optical_frame"
            img_msg.header.stamp = self.get_clock().now().to_msg()
            
            # Synchronize the timestamp of the CameraInfo message
            self.camera_info_msg.header.stamp = img_msg.header.stamp
            
            # Publish both
            self.image_pub.publish(img_msg)
            self.info_pub.publish(self.camera_info_msg)

def main(args=None):
    rclpy.init(args=args)
    node = IMX219Node()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()