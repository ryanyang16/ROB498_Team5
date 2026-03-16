import rclpy
import numpy as np
from rclpy.node import Node
from std_srvs.srv import Trigger
from geometry_msgs.msg import PoseStamped
from rclpy.qos import QoSProfile, ReliabilityPolicy
from mavros_msgs.srv import CommandBool, SetMode

class CommNode(Node):
    def __init__(self):
        super().__init__('rob498_drone_5')

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=10)
        
        # State Machine
        self.current_state = "INIT"
        self.current_wp_index = 0
        self.waiting_at_wp = False
        self.wp_arrival_time = 0.0
        
        # Target Setpoints
        self.target_x = 0.0
        self.target_y = 0.0
        self.target_z = 0.0
        self.target_orientation_w = 1.0
        self.target_orientation_x = 0.0
        self.target_orientation_y = 0.0
        self.target_orientation_z = 0.0
        
        self.current_pose = PoseStamped()
        self.got_initial_pose = False

        # MAVROS communication
        self.vision_sub = self.create_subscription(PoseStamped, '/mavros/local_position/pose', self.pose_callback, qos)
        self.setpoint_pub = self.create_publisher(PoseStamped, '/mavros/setpoint_position/local', 10)
        self.arm_client = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.mode_client = self.create_client(SetMode, '/mavros/set_mode')

        # Services
        self.srv_launch = self.create_service(Trigger, 'rob498_drone_5/comm/launch', self.callback_launch)
        self.srv_land   = self.create_service(Trigger, 'rob498_drone_5/comm/land', self.callback_land)
        self.srv_abort  = self.create_service(Trigger, 'rob498_drone_5/comm/abort', self.callback_abort)
        # We will repurpose the calibrate service to trigger your photo flight path
        self.srv_calib  = self.create_service(Trigger, 'rob498_drone_5/comm/calibrate', self.callback_start_photos)

        # --- Photo Collection Waypoints ---
        self.photo_waypoints = [
            # Bottom Layer (Z = 0.5)
            [-2.0, -2.0, 0.5], 
            [2.0, -2.0, 0.5],  
            [0.0, 0.0, 0.5],   
            [2.0, 2.0, 0.5],   
            [-2.0, 2.0, 0.5],  

            # Mid-Air Face Centers
            [-2.0, 0.0, 1.0],  
            [0.0, -2.0, 1.0],  
            [0.0, 0.0, 1.25],  
            [0.0, 2.0, 1.5],   
            [2.0, 0.0, 1.5],   

            # Top Layer (Z = 2.0)
            [-2.0, -2.0, 2.0], 
            [2.0, -2.0, 2.0],  
            [0.0, 0.0, 2.0],   
            [2.0, 2.0, 2.0],   
            [-2.0, 2.0, 2.0]   
        ]

        # The main code (runs at 50Hz)
        self.timer = self.create_timer(0.02, self.main_loop)
        self.get_logger().info("Drone 5 Photo Node Initialized and Waiting.")

    def pose_callback(self, msg):
        self.current_pose = msg
        if not self.got_initial_pose:
            self.target_x = msg.pose.position.x
            self.target_y = msg.pose.position.y
            self.target_z = msg.pose.position.z
            self.target_orientation_x = msg.pose.orientation.x
            self.target_orientation_y = msg.pose.orientation.y
            self.target_orientation_z = msg.pose.orientation.z
            self.target_orientation_w = msg.pose.orientation.w
            self.got_initial_pose = True

    # Services
    def callback_launch(self, request, response):
        self.get_logger().info('Requested: LAUNCH')
        self.current_state = "LAUNCH"
        response.success = True
        return response

    def callback_start_photos(self, request, response):
        self.get_logger().info('Requested: PHOTO FLIGHT (Starting Sequence)')
        self.current_state = "PHOTO_FLIGHT"
        self.current_wp_index = 0
        self.waiting_at_wp = False
        response.success = True
        return response

    def callback_land(self, request, response):
        self.get_logger().info('Requested: LAND')
        self.current_state = "LAND"
        response.success = True
        return response

    def callback_abort(self, request, response):
        self.get_logger().fatal('Requested: ABORT')
        self.current_state = "ABORT"
        response.success = True
        return response

    def main_loop(self):
        if not self.got_initial_pose:
            return 

        if self.current_state == "INIT":
            self.target_x = self.current_pose.pose.position.x
            self.target_y = self.current_pose.pose.position.y
            self.target_z = self.current_pose.pose.position.z
            self.target_orientation_w = self.current_pose.pose.orientation.w
            self.target_orientation_x = self.current_pose.pose.orientation.x
            self.target_orientation_y = self.current_pose.pose.orientation.y
            self.target_orientation_z = self.current_pose.pose.orientation.z

        elif self.current_state == "LAUNCH":
            self.target_z = 0.5 

        elif self.current_state == "PHOTO_FLIGHT":
            if self.current_wp_index < len(self.photo_waypoints):
                target_wp = self.photo_waypoints[self.current_wp_index]
                
                # Update target
                self.target_x = target_wp[0]
                self.target_y = target_wp[1]
                self.target_z = target_wp[2]
                
                # Calculate Distance
                current_pos = np.array([
                    self.current_pose.pose.position.x,
                    self.current_pose.pose.position.y,
                    self.current_pose.pose.position.z
                ])
                distance = np.linalg.norm(current_pos - target_wp)

                # Wait/Next Logic
                if not self.waiting_at_wp:
                    if distance < 0.15: # 15cm threshold for arrival
                        self.get_logger().info(f"Reached Waypoint {self.current_wp_index + 1}/{len(self.photo_waypoints)}. Hovering for 3 seconds...")
                        self.waiting_at_wp = True
                        self.wp_arrival_time = self.get_clock().now().nanoseconds / 1e9
                else:
                    current_time = self.get_clock().now().nanoseconds / 1e9
                    time_elapsed = current_time - self.wp_arrival_time
                    
                    if time_elapsed >= 3.0:
                        self.waiting_at_wp = False
                        self.current_wp_index += 1
                        
                        if self.current_wp_index < len(self.photo_waypoints):
                            self.get_logger().info(f"Moving to Waypoint {self.current_wp_index + 1}...")
                        else:
                            self.get_logger().info("Photo collection route complete! Hovering at final position.")
            else:
                pass # Hold final position once array is complete

        elif self.current_state == "LAND":
            self.target_z = -0.1

        elif self.current_state == "ABORT":
            self.target_z = 0.0
            arm_req = CommandBool.Request()
            arm_req.value = False 
            self.arm_client.call_async(arm_req)

        # ALWAYS PUBLISH SETPOINTS (50Hz)
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        
        msg.pose.position.x = float(self.target_x)
        msg.pose.position.y = float(self.target_y)
        msg.pose.position.z = float(self.target_z)
        
        msg.pose.orientation.w = self.target_orientation_w
        msg.pose.orientation.x = self.target_orientation_x
        msg.pose.orientation.y = self.target_orientation_y
        msg.pose.orientation.z = self.target_orientation_z

        self.setpoint_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = CommNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()