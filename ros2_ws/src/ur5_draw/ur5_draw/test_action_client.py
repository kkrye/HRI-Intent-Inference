#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import Point
import cv2

from ur_draw_cmake.action import DrawStrokes
from ur_draw_cmake.msg import Stroke
from ur5_draw.image_to_svg import image_to_lines


class DrawActionClient(Node):
    def __init__(self):
        super().__init__('draw_action_client')

        self._action_client = ActionClient(
            self,
            DrawStrokes,
            'draw_strokes'
        )

        self.get_logger().info('Waiting for action server...')
        self._action_client.wait_for_server()
        self.get_logger().info('Action server available!')

    def send_goal(self, strokes, img_length,img_width):
        """Load image, convert to strokes, and send goal to action server."""


        self.get_logger().info(f'Image dimensions: {img_width} x {img_length}')

        self.get_logger().info(f'Generated {len(strokes)} strokes')

        # Create goal message
        goal_msg = DrawStrokes.Goal()
        goal_msg.img_width = int(img_width)
        goal_msg.img_length = int(img_length)

        # Convert strokes to ROS messages
        for stroke_points in strokes:
            stroke_msg = Stroke()
            for x, y in stroke_points:
                stroke_msg.points.append(Point(x=x, y=y, z=0.0))
            goal_msg.strokes.append(stroke_msg)

        self.get_logger().info('Sending goal to action server...')
        self.action_completed = False

        # Send goal with feedback callback
        send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )

        send_goal_future.add_done_callback(self.goal_response_callback)

        return send_goal_future

    def goal_response_callback(self, future):
        """Called when the action server accepts/rejects the goal."""
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().error('Goal rejected by action server')
            return

        self.get_logger().info('Goal accepted by action server')

        # Get result
        get_result_future = goal_handle.get_result_async()
        get_result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        """Called when feedback is received from the action server."""
        feedback = feedback_msg.feedback
        self.get_logger().info(
            f'Feedback: Stroke {feedback.current_stroke}/{feedback.total_strokes} '
            f'({feedback.percent_complete:.1f}%) - {feedback.status_message}'
        )

    def get_result_callback(self, future):
        """Called when the action completes."""
        result = future.result().result

        if result.success:
            self.get_logger().info(
                f'SUCCESS! Completed {result.strokes_completed} strokes. '
                f'Message: {result.message}'
            )
        else:
            self.get_logger().error(
                f'FAILED! Completed {result.strokes_completed} strokes. '
                f'Message: {result.message}'
            )

        self.action_completed = True


def main(args=None):
    rclpy.init(args=args)

        # Get image path from command line argument
    import sys
    user_args = rclpy.utilities.remove_ros_args(sys.argv)

    # Get image path from command line argument
    if len(user_args) < 2:
        print("Usage: ros2 run your_package draw_action_client.py <image_path>")
        print("Example: ros2 run your_package draw_action_client.py /path/to/image.jpg")
        # Path to your test image
        image_path = '/home/studioadmin/HRI_Adversarial_Robot_Project/ros2_ws/test.jpg'
    else:
        image_path = user_args[1]

    # Load and process image
    print(f'Loading image from: {image_path}')
    img = cv2.imread(image_path)

    if img is None:
        print(f'Failed to load image: {image_path}')
        exit(1)


    height, width, channels = img.shape
    # Convert image to strokes
    print('Converting image to strokes...')
    strokes = image_to_lines(img)

    action_client = DrawActionClient()

    # Send goal
    future = action_client.send_goal(strokes, img_length=height,img_width=width)

    if future is None:
        action_client.get_logger().error('Failed to send goal')
        action_client.destroy_node()
        rclpy.shutdown()
        return

    # Spin until the action completes
    try:
        rclpy.spin(action_client)
    except KeyboardInterrupt:
        action_client.get_logger().info('Keyboard interrupt, canceling goal...')
        # Optionally cancel the goal here if needed
    finally:
        action_client.destroy_node()


if __name__ == '__main__':
    main()