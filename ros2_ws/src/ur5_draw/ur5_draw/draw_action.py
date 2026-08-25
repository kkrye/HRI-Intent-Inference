#!/usr/bin/env python3
import sys
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import Pose, Point, Quaternion
from visualization_msgs.msg import Marker
# from tf2_ros import TransformListener, Buffer, TransformException
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_geometry_msgs import do_transform_pose
from copy import deepcopy
import cv2
import time

from ur_draw_cmake.srv import DrawStroke
from ur_draw_cmake.action import DrawStrokes

IN_TO_M = 0.0254

class DrawActionServer(Node):
    PEN_HEIGHT = .05
    def __init__(self):
        super().__init__('draw_action_server')
        # Use reentrant callback group to allow multiple callbacks simultaneously
        self.callback_group = ReentrantCallbackGroup()

        # tf setup
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.image_to_world_t = self.get_img_transform('image_frame', 'world')
        self.get_logger().info('Transform successfully retrieved and cached.')

        # service client setup
        self.client = self.create_client(DrawStroke, 'moveit_draw_stroke',callback_group=self.callback_group)
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')
        self.request = DrawStroke.Request()

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
        scale_factor = .9
        self.page_length_m  =  11 * IN_TO_M * scale_factor
        self.page_width_m   = 8.5 * IN_TO_M * scale_factor

        # home pose in image frame
        self.home_pose = Pose(position=Point(y=-0.10, z=0.3),orientation=Quaternion(w=.707,y=.707))
        
        # Action server
        self._action_server = ActionServer(
            self,
            DrawStrokes,
            'draw_strokes',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=self.callback_group
        )
        
        self.get_logger().info('Draw Action Server initialized and ready.')
    
    def goal_callback(self, goal_request):
        """Accept or reject a client request to begin an action."""
        self.get_logger().info('Received goal request')
        return GoalResponse.ACCEPT
    
    def cancel_callback(self, goal_handle):
        """Accept or reject a client request to cancel an action."""
        self.get_logger().info('Received cancel request')
        return CancelResponse.ACCEPT

    def execute_callback(self, goal_handle):
        """Execute the draw strokes action."""
        self.get_logger().info('Executing goal...')
        
        # Set image dimensions from goal
        img_width = goal_handle.request.img_width
        img_length = goal_handle.request.img_length
        self.get_logger().info(f'Image dimensions: {img_width} x {img_length}')
        
        # Get strokes directly from goal
        strokes = self.convert_strokes_from_msg(goal_handle.request.strokes)
        total_strokes = len(strokes)
        
        # Feedback message
        feedback_msg = DrawStrokes.Feedback()
        feedback_msg.total_strokes = total_strokes
        
        # Execute drawing
        try:
            for i, stroke in enumerate(strokes):
                # Check if goal has been canceled
                if goal_handle.is_cancel_requested:
                    goal_handle.canceled()
                    result = DrawStrokes.Result()
                    result.success = False
                    result.message = 'Goal canceled by client'
                    result.strokes_completed = i
                    self.get_logger().info('Goal canceled')
                    return result
                
                # Update feedback
                feedback_msg.current_stroke = i + 1
                feedback_msg.percent_complete = ((i + 1) / total_strokes) * 100.0
                feedback_msg.status_message = f'Drawing stroke {i + 1} of {total_strokes}'
                goal_handle.publish_feedback(feedback_msg)
                
                self.get_logger().info(f"Starting stroke {i + 1} of {total_strokes}, length {len(stroke)}")
                
                # Draw the stroke
                success = self.draw_stroke_traj(stroke,img_width,img_length)
                if not success:
                    goal_handle.abort()
                    result = DrawStrokes.Result()
                    result.success = False
                    result.message = f'Failed to draw stroke {i + 1}'
                    result.strokes_completed = i
                    return result
            
            # Return to home
            self.get_logger().info("All strokes drawn, returning home")
            self.go_home()
            
            # Goal succeeded
            goal_handle.succeed()
            result = DrawStrokes.Result()
            result.success = True
            result.message = f'Successfully drew {total_strokes} strokes'
            result.strokes_completed = total_strokes
            
            self.get_logger().info('Goal succeeded')
            return result
        
        except Exception as e:
            self.get_logger().error(f'Exception during execution: {str(e)}')
            goal_handle.abort()
            result = DrawStrokes.Result()
            result.success = False
            result.message = f'Exception: {str(e)}'
            result.strokes_completed = 0
            return result
    
    def convert_strokes_from_msg(self, stroke_msgs):
        """Convert ROS message strokes to list of (x, y) tuples."""
        strokes = []
        for stroke_msg in stroke_msgs:
            stroke = [(pt.x, pt.y) for pt in stroke_msg.points]
            strokes.append(stroke)
        return strokes
    
    def go_home(self):
        home_world_pose = do_transform_pose(self.home_pose, self.image_to_world_t)
        self.get_logger().info("Returning to home position")
        future = self.send_traj_request([home_world_pose])

        # Wait for service response (MultiThreadedExecutor handles spinning)
        while not future.done():
            time.sleep(0.01)

        if future.result() is not None:
            self.get_logger().info(f"Move completed {'' if future.result().success else 'un'}successfully. {future.result().message}")
            return future.result().success
        else:
            self.get_logger().error("Service call failed.")
            return False

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

    def draw_stroke_traj(self, points,width,length):
        pose_list = []
        for point in points:
            x, y = point
            img_pose = Pose()
            img_pose.position.x = x/length * self.page_length_m
            img_pose.position.y = y/width * self.page_width_m
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
        
        # Wait for service response (MultiThreadedExecutor handles spinning)
        while not future.done():
            time.sleep(0.01)

        if future.result() is not None:
            self.get_logger().info(f"Move completed {'' if future.result().success else 'un'}successfully. {future.result().message}")
            return future.result().success
        else:
            self.get_logger().error("Service call failed.")
            return False

def main(args=None):
    rclpy.init(args=args)
    
    draw_action_server = DrawActionServer()
    
    # Use MultiThreadedExecutor for concurrent callback execution
    executor = MultiThreadedExecutor()
    executor.add_node(draw_action_server)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        draw_action_server.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()