#!/usr/bin/env python3
import sys
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from geometry_msgs.msg import Pose, Point, PoseStamped, Quaternion
from visualization_msgs.msg import Marker
# from tf2_ros import TransformListener, Buffer, TransformException
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_geometry_msgs import do_transform_pose
from copy import deepcopy
import numpy as np
import cv2

from ur5_draw.image_to_svg import image_to_lines
from ur_draw_cmake.srv import DrawStroke

IN_TO_M = 0.0254

class Draw(Node):
    PEN_HEIGHT = .05
    def __init__(self):
        super().__init__('draw_img')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.client = self.create_client(DrawStroke, 'moveit_draw_stroke')
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')
        self.request = DrawStroke.Request()

        self.image_to_world_t = self.get_img_transform('image_frame', 'world')
        self.get_logger().info('Transform successfully retrieved and cached.')

        # rviz marker for image viz
        self.img_viz_pub = self.create_publisher(Marker, 'image_viz', 10)
        self.img_viz = Marker()
        self.img_viz.header.frame_id = 'world'
        self.img_viz.type = Marker.LINE_STRIP
        self.img_viz.scale.x = 0.002
        self.img_viz.color.a = 1.0
        self.img_viz.color.r = 1.0
        self.img_viz.points = []

        # image settings
        scale_factor = 1
        self.page_length_m  =  15 * IN_TO_M * scale_factor
        self.page_width_m   =   11 * IN_TO_M * scale_factor

        # home pose in image frame
        self.home_pose = Pose(position=Point(y=-0.10, z=0.3),orientation=Quaternion(w=.707,y=.707))

    def go_home(self):
        home_world_pose = do_transform_pose(self.home_pose, self.image_to_world_t)
        self.get_logger().info("Returning to home position")
        future = self.send_traj_request([home_world_pose])
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            self.get_logger().info(f"Move completed {'' if future.result().success else 'un'}successfully. {future.result().message}")
        else:
            self.get_logger().error("Service call failed.")

    def get_img_transform(self, source_frame, target_frame):
        wait_duration_sec = 5.0
        start_time = self.get_clock().now()
        transform_available = False

        while rclpy.ok() and (self.get_clock().now() - start_time).nanoseconds / 1e9 < wait_duration_sec:
            if self.tf_buffer.can_transform(target_frame, source_frame, rclpy.time.Time()):
                self.get_logger().info('Transform available!')
                transform_available = True
                break

            # Spin the node once to allow the TransformListener to process incoming TF messages
            rclpy.spin_once(self, timeout_sec=0.1)

        if not transform_available:
             self.get_logger().error(f"Failed to find transform between {source_frame} and {target_frame} within {wait_duration_sec}s.")
             # Exit the node setup gracefully or raise an error
             sys.exit(1)

        # Now that we know the transform exists, we can perform the lookup (with a short timeout)
        try:
            return self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=0.1) # Short timeout since we just confirmed it exists
            )

        except TransformException as ex:
             # Should not happen if can_transform passed, but good practice to catch it
             self.get_logger().error(f"Transform lookup failed after waiting: {ex}")
             sys.exit(1)

    def send_traj_request(self, poses):
        self.request.target_poses = poses
        self.get_logger().info("Sending request to move to the target pose...")
        return self.client.call_async(self.request)

    def draw_stroke_traj(self, points,img_length,img_width):
        pose_list = []
        for point in points:
            x, y = point
            img_pose = Pose()
            img_pose.position.x = x/img_length * self.page_length_m
            img_pose.position.y = y/img_width * self.page_width_m
            img_pose.position.z = self.PEN_HEIGHT

            img_pose.orientation.w = 0.707
            img_pose.orientation.x = 0.0
            img_pose.orientation.y = 0.707
            img_pose.orientation.z = 0.0

            tool_pose = deepcopy(img_pose)
            tool_pose.position.z = 0.0
            world_pose = do_transform_pose(img_pose, self.image_to_world_t)
            pose_list.append(world_pose)
            self.img_viz.points.append(do_transform_pose(tool_pose,self.image_to_world_t).position)
            self.img_viz.header.stamp = self.get_clock().now().to_msg()
            self.img_viz_pub.publish(self.img_viz)

        self.get_logger().info("Sending Trajectory to service")
        future = self.send_traj_request(pose_list)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            self.get_logger().info(f"Move completed {'' if future.result().success else 'un'}successfully. {future.result().message}")
        else:
            self.get_logger().error("Service call failed.")

    def draw_strokes(self, strokes,img_length,img_width):
        self.get_logger().info(f"Starting Drawing of {len(strokes)} strokes")
        for i,stroke in enumerate(strokes):
            self.get_logger().info(f"Starting stroke {i} of length {len(stroke)}")
            self.draw_stroke_traj(stroke,img_length,img_width)
        self.get_logger().info("All strokes drawn.")
        self.go_home()

def main(args=None):
    rclpy.init(args=args)
    # test_img = cv2.imread('/HRI_Adversarial_Robot_Project/ros2_ws/test.jpg')
    test_img = cv2.imread('/home/studioadmin/HRI_Adversarial_Robot_Project/ros2_ws/test.jpg')
    height, width, channels = test_img.shape
    strokes = image_to_lines(test_img)

    print(*[f"{i}::{len(s)}\n" for i,s in enumerate(strokes)])

    draw_node = Draw()
    draw_node.go_home()

    try:
        draw_node.draw_strokes(strokes,img_length=height,img_width=width)
    except KeyboardInterrupt:
        pass
    finally:
        draw_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()