import rclpy
import numpy as np
from rclpy.node import Node
from std_srvs.srv import Trigger
from geometry_msgs.msg import PoseStamped, PoseArray
from rclpy.qos import QoSProfile, ReliabilityPolicy
from mavros_msgs.srv import CommandBool, SetMode
from scipy.spatial.transform import Rotation as R

class CommNode(Node):
    def __init__(self):
        super().__init__('rob498_drone_5')

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=10)
        
        # State Machine Variables
        self.current_state = "INIT"
        self.waypoints = np.empty((0, 3))
        self.transformed_waypoints = np.empty((0, 3))
        self.waypoints_received = False
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

        # --- KABSCH CALIBRATION VARIABLES ---
        self.vicon_points = []
        self.rs_points = []
        self.calib_rotation = None
        self.calib_translation = None
        self.current_vicon_pose = PoseStamped()

        # MAVROS communication
        self.vision_sub = self.create_subscription(PoseStamped, '/mavros/local_position/pose', self.pose_callback, qos)
        self.setpoint_pub = self.create_publisher(PoseStamped, '/mavros/setpoint_position/local', 10)
        self.arm_client = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.mode_client = self.create_client(SetMode, '/mavros/set_mode')

        # Vicon Ground Truth
        self.vicon_sub = self.create_subscription(PoseStamped, '/vicon/ROB498_Drone/ROB498_Drone', self.vicon_callback, 10)

        # TA Grading Communication
        self.srv_launch = self.create_service(Trigger, 'rob498_drone_5/comm/launch', self.callback_launch)
        self.srv_test   = self.create_service(Trigger, 'rob498_drone_5/comm/test', self.callback_test)
        self.srv_land   = self.create_service(Trigger, 'rob498_drone_5/comm/land', self.callback_land)
        self.srv_abort  = self.create_service(Trigger, 'rob498_drone_5/comm/abort', self.callback_abort)
        self.sub_waypoints = self.create_subscription(PoseArray, 'rob498_drone_5/comm/waypoints', self.callback_waypoints, 10)

        # Custom Calibration Services
        self.srv_record = self.create_service(Trigger, 'rob498_drone_5/comm/record_pt', self.callback_record_pt)
        self.srv_calibrate = self.create_service(Trigger, 'rob498_drone_5/comm/calibrate', self.callback_calibrate)

        # Main Control Loop (50Hz)
        self.timer = self.create_timer(0.02, self.main_loop)
        self.get_logger().info("Drone 5 Node Initialized and Waiting.")


    # ==========================================
    # SENSOR CALLBACKS
    # ==========================================
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

    def vicon_callback(self, msg):
        # Continually update our known Vicon position
        self.current_vicon_pose = msg

    def callback_waypoints(self, msg):
        if self.waypoints_received:
            return
        self.get_logger().info('Waypoints received from TA')
        self.waypoints_received = True
        for pose in msg.poses:
            pos = np.array([pose.position.x, pose.position.y, pose.position.z])
            self.waypoints = np.vstack((self.waypoints, pos))


    # ==========================================
    # KABSCH CALIBRATION SERVICES
    # ==========================================
    def callback_record_pt(self, request, response):
        """Records the current X,Y,Z from both Vicon and RealSense"""
        if not self.got_initial_pose:
            self.get_logger().error("No RealSense pose data yet! Cannot record.")
            response.success = False
            return response

        # Grab current coordinates
        v_pt = [self.current_vicon_pose.pose.position.x, 
                self.current_vicon_pose.pose.position.y, 
                self.current_vicon_pose.pose.position.z]
                
        r_pt = [self.current_pose.pose.position.x, 
                self.current_pose.pose.position.y, 
                self.current_pose.pose.position.z]

        self.vicon_points.append(v_pt)
        self.rs_points.append(r_pt)

        self.get_logger().info(f"Recorded Point Pair {len(self.vicon_points)}")
        self.get_logger().info(f"Vicon: {v_pt}; \nRealsense: {r_pt}")
        response.success = True
        return response

    def callback_calibrate(self, request, response):
        """Calculates the Kabsch transform from the recorded points"""
        if len(self.vicon_points) < 3:
            self.get_logger().error("Need at least 3 points for 3D calibration!")
            response.success = False
            return response

        v_pts = np.array(self.vicon_points)
        r_pts = np.array(self.rs_points)

        # 1. Find centroids
        centroid_v = np.mean(v_pts, axis=0)
        centroid_r = np.mean(r_pts, axis=0)

        # 2. Center the point clouds
        v_centered = v_pts - centroid_v
        r_centered = r_pts - centroid_r

        # 3. Kabsch Algorithm: Find optimal rotation
        rot, rmsd = R.align_vectors(r_centered, v_centered)
        self.calib_rotation = rot

        # 4. Calculate translation offset
        self.calib_translation = centroid_r - rot.apply(centroid_v)

        self.get_logger().info("=========================================")
        self.get_logger().info("         CALIBRATION SUCCESSFUL          ")
        self.get_logger().info(f" RMS Error: {rmsd:.4f} meters")
        self.get_logger().info(f" Translation: {self.calib_translation}")
        self.get_logger().info(f" Rotation: {self.calib_rotation}")

        self.get_logger().info("=========================================")
        
        response.success = True
        return response


    # ==========================================
    # TA SERVICE CALLBACKS
    # ==========================================
    def callback_launch(self, request, response):
        self.get_logger().info('TA Requested: LAUNCH')
        
        # --- APPLY THE CALIBRATION TO THE WAYPOINTS ---
        if self.waypoints_received:
            if self.calib_rotation is not None and self.calib_translation is not None:
                self.get_logger().info("Applying Kabsch Calibration to Waypoints...")
                self.transformed_waypoints = np.empty((0, 3))
                
                for wp in self.waypoints:
                    # Math: P_local = R * P_vicon + t
                    local_wp = self.calib_rotation.apply(wp) + self.calib_translation
                    self.transformed_waypoints = np.vstack((self.transformed_waypoints, local_wp))
            else:
                self.get_logger().error("CRITICAL: No calibration found! Using raw Vicon waypoints.")
                self.transformed_waypoints = self.waypoints
        
        self.current_state = "LAUNCH"
        response.success = True
        return response

    def callback_test(self, request, response):
        self.get_logger().info('TA Requested: TEST (Vicon Off)')
        self.current_state = "TEST"
        response.success = True
        return response

    def callback_land(self, request, response):
        self.get_logger().info('TA Requested: LAND')
        self.current_state = "LAND"
        response.success = True
        return response

    def callback_abort(self, request, response):
        self.get_logger().fatal('TA Requested: ABORT')
        self.current_state = "ABORT"
        response.success = True
        return response

    
    # ==========================================
    # MAIN LOOP (50Hz)
    # ==========================================
    def main_loop(self):
        if not self.got_initial_pose:
            return 

        if self.current_state == "INIT":
            # Track current position to prevent snapping on takeoff
            self.target_x = self.current_pose.pose.position.x
            self.target_y = self.current_pose.pose.position.y
            self.target_z = self.current_pose.pose.position.z
            self.target_orientation_w = self.current_pose.pose.orientation.w
            self.target_orientation_x = self.current_pose.pose.orientation.x
            self.target_orientation_y = self.current_pose.pose.orientation.y
            self.target_orientation_z = self.current_pose.pose.orientation.z

        elif self.current_state == "LAUNCH":
            # Command hover at 0.5m
            self.target_z = 0.5 

        elif self.current_state == "TEST":
            # Use TRANSFORMED waypoints to navigate
            if self.waypoints_received and self.current_wp_index < len(self.transformed_waypoints):
                target_wp = self.transformed_waypoints[self.current_wp_index]
                
                self.target_x = target_wp[0]
                self.target_y = target_wp[1]
                self.target_z = target_wp[2]
                
                current_pos = np.array([
                    self.current_pose.pose.position.x,
                    self.current_pose.pose.position.y,
                    self.current_pose.pose.position.z
                ])
                
                distance = np.linalg.norm(current_pos - target_wp)
                
                if not self.waiting_at_wp:
                    # Check if arrived (within 10cm threshold)
                    if distance < 0.1:
                        self.get_logger().info(f"Reached Waypoint {self.current_wp_index + 1}/{len(self.transformed_waypoints)}. Pausing for 3 seconds...")
                        self.waiting_at_wp = True
                        self.wp_arrival_time = self.get_clock().now().nanoseconds / 1e9
                else:
                    current_time = self.get_clock().now().nanoseconds / 1e9
                    time_elapsed = current_time - self.wp_arrival_time
                    
                    if time_elapsed >= 3.0:
                        self.waiting_at_wp = False
                        self.current_wp_index += 1
                        
                        if self.current_wp_index < len(self.transformed_waypoints):
                            self.get_logger().info(f"Proceeding to Waypoint {self.current_wp_index + 1}...")
                        else:
                            self.get_logger().info("Course complete! Holding final position.")
            
            elif self.current_wp_index >= len(self.transformed_waypoints):
                 pass

        elif self.current_state == "LAND":
            self.target_z = -0.1

        elif self.current_state == "ABORT":
            self.target_z = 0.0
            arm_req = CommandBool.Request()
            arm_req.value = False 
            self.arm_client.call_async(arm_req)

        # ALWAYS PUBLISH SETPOINTS
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
