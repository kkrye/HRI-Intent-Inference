#include <memory>
#include <thread>
#include <cmath>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit_msgs/msg/orientation_constraint.hpp>
#include <moveit_msgs/msg/constraints.hpp>
#include <moveit_msgs/msg/collision_object.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>

#include "ur_draw_cmake/srv/draw_stroke.hpp"

class DrawStrokeServer : public rclcpp::Node
{
public:
    DrawStrokeServer()
        : Node("moveit_draw_stroke_server",
               rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true)),
          planning_group_("ur_manipulator"),
          raised_pen_height_(0.02),
          distance_threshold_(0.8),
          eef_step_(0.001),
          arm_bounds_(0.4),
          table_height_(-0.0025),
          cartesian_fraction_threshold_(0.65)
    {
        // Declare and get parameters
        this->declare_parameter("planning_group", planning_group_);
        this->declare_parameter("raised_pen_height", raised_pen_height_);
        this->declare_parameter("distance_threshold", distance_threshold_);
        this->declare_parameter("eef_step", eef_step_);
        this->declare_parameter("arm_bounds", arm_bounds_);
        this->declare_parameter("table_height", table_height_);
        this->declare_parameter("cartesian_fraction_threshold", cartesian_fraction_threshold_);
        
        this->get_parameter("planning_group", planning_group_);
        this->get_parameter("raised_pen_height", raised_pen_height_);
        this->get_parameter("distance_threshold", distance_threshold_);
        this->get_parameter("eef_step", eef_step_);
        this->get_parameter("arm_bounds", arm_bounds_);
        this->get_parameter("table_height", table_height_);
        this->get_parameter("cartesian_fraction_threshold", cartesian_fraction_threshold_);

        // Create service
        service_ = this->create_service<ur_draw_cmake::srv::DrawStroke>(
            "moveit_draw_stroke",
            std::bind(&DrawStrokeServer::handle_draw_stroke_request,
                     this,
                     std::placeholders::_1,
                     std::placeholders::_2));

        RCLCPP_INFO(this->get_logger(), "DrawStroke Service Server initialized");
        RCLCPP_INFO(this->get_logger(), "  Planning group: %s", planning_group_.c_str());
        RCLCPP_INFO(this->get_logger(), "  Raised pen height: %.3f m", raised_pen_height_);
        RCLCPP_INFO(this->get_logger(), "  Distance threshold: %.3f m", distance_threshold_);
        RCLCPP_INFO(this->get_logger(), "Service ready at: /moveit_draw_stroke");
        first_move_ = true;
    }

    /**
     * @brief Add collision objects to the planning scene
     */
    void add_collision_objects()
    {
        // Wait a bit for planning scene to be ready
        rclcpp::sleep_for(std::chrono::seconds(1));

        moveit::planning_interface::PlanningSceneInterface planning_scene;

        // Create table collision object
        moveit_msgs::msg::CollisionObject table;
        // table.header.frame_id = "world";
        table.header.frame_id = "table_frame";
        table.header.stamp = this->now();
        table.id = "table";
        table.operation = moveit_msgs::msg::CollisionObject::ADD;

        // Define table as a box
        shape_msgs::msg::SolidPrimitive primitive;
        primitive.type = primitive.BOX;
        primitive.dimensions.resize(3);
        primitive.dimensions[0] = 2.0;  // X length (2m)
        primitive.dimensions[1] = 2.0 * arm_bounds_;  // Y width (2m)
        primitive.dimensions[2] = 0.05; // Z height (5cm)

        // Position the table
        geometry_msgs::msg::Pose table_pose;
        table_pose.orientation.w = 1.0;
        table_pose.position.x = 0.0;
        table_pose.position.y = 0.0;
        table_pose.position.z = table_height_; // Top surface at Z=0

        table.primitives.push_back(primitive);
        table.primitive_poses.push_back(table_pose);

        // Define table as a box
        shape_msgs::msg::SolidPrimitive primitive_wall_1;
        primitive_wall_1.type = primitive_wall_1.BOX;
        primitive_wall_1.dimensions.resize(3);
        primitive_wall_1.dimensions[0] = 2.0;
        primitive_wall_1.dimensions[1] = .005;
        primitive_wall_1.dimensions[2] = 2.0;

        // Position the table
        geometry_msgs::msg::Pose table_pose_wall_1;
        table_pose_wall_1.orientation.w = 1.0;
        table_pose_wall_1.position.x = 0.0;
        table_pose_wall_1.position.y = -1*arm_bounds_;
        table_pose_wall_1.position.z = 0.0;

        table.primitives.push_back(primitive_wall_1);
        table.primitive_poses.push_back(table_pose_wall_1);

        // Define table as a box
        shape_msgs::msg::SolidPrimitive primitive_wall_2;
        primitive_wall_2.type = primitive_wall_2.BOX;
        primitive_wall_2.dimensions.resize(3);
        primitive_wall_2.dimensions[0] = 2.0;
        primitive_wall_2.dimensions[1] = .005;
        primitive_wall_2.dimensions[2] = 2.0;

        // Position the table
        geometry_msgs::msg::Pose table_pose_wall_2;
        table_pose_wall_2.orientation.w = 1.0;
        table_pose_wall_2.position.x = 0.0;
        table_pose_wall_2.position.y = arm_bounds_;
        table_pose_wall_2.position.z = 0.0;

        table.primitives.push_back(primitive_wall_2);
        table.primitive_poses.push_back(table_pose_wall_2);

        // Add to scene
        std::vector<moveit_msgs::msg::CollisionObject> collision_objects;
        collision_objects.push_back(table);

        bool success = planning_scene.applyCollisionObjects(collision_objects);

        if (success) {
            RCLCPP_INFO(this->get_logger(),
                       "Successfully added 'table' collision object to planning scene");

            // Verify the object was added
            auto known_objects = planning_scene.getKnownObjectNames();
            RCLCPP_INFO(this->get_logger(), "Known objects in scene: %zu", known_objects.size());
            for (const auto& obj : known_objects) {
                RCLCPP_INFO(this->get_logger(), "  - %s", obj.c_str());
            }
        } else {
            RCLCPP_WARN(this->get_logger(),
                       "Failed to add 'table' collision object to planning scene");
        }
    }

private:
    // Member variables
    rclcpp::Service<ur_draw_cmake::srv::DrawStroke>::SharedPtr service_;
    std::string planning_group_;
    double raised_pen_height_;
    double distance_threshold_;
    double eef_step_;
    double arm_bounds_;
    double table_height_;
    double cartesian_fraction_threshold_;
    bool first_move_;

    /**
     * @brief Calculate Euclidean distance between two poses
     */
    double calculate_distance(const geometry_msgs::msg::Pose& pose1,
                             const geometry_msgs::msg::Pose& pose2) const
    {
        double dx = pose2.position.x - pose1.position.x;
        double dy = pose2.position.y - pose1.position.y;
        double dz = pose2.position.z - pose1.position.z;
        return std::sqrt(dx*dx + dy*dy + dz*dz);
    }

    /**
     * @brief Get current pose of the end effector in world frame
     */
    geometry_msgs::msg::Pose get_current_pose(
        moveit::planning_interface::MoveGroupInterface& move_group) const
    {
        geometry_msgs::msg::PoseStamped current_pose_stamped =
            move_group.getCurrentPose("tool0");

        RCLCPP_DEBUG(this->get_logger(), "Current pose: [%.3f, %.3f, %.3f]",
                    current_pose_stamped.pose.position.x,
                    current_pose_stamped.pose.position.y,
                    current_pose_stamped.pose.position.z);

        return current_pose_stamped.pose;
    }

    /**
     * @brief Create a raised version of a pose (pen up)
     */
    geometry_msgs::msg::Pose create_raised_pose(const geometry_msgs::msg::Pose& pose) const
    {
        auto raised_pose = pose;
        raised_pose.position.z += raised_pen_height_;
        return raised_pose;
    }

    /**
     * @brief Move to a target pose using standard planning
     */
    bool move_to_pose(moveit::planning_interface::MoveGroupInterface& move_group,
                     const geometry_msgs::msg::Pose& target_pose,
                     const std::string& error_prefix)
    {
        move_group.setPoseTarget(target_pose, "tool0");

        moveit::planning_interface::MoveGroupInterface::Plan plan;
        bool success = static_cast<bool>(move_group.plan(plan));

        if (!success) {
            RCLCPP_ERROR(this->get_logger(), "%s: Planning failed", error_prefix.c_str());
            return false;
        }

        if (move_group.execute(plan) != moveit::core::MoveItErrorCode::SUCCESS) {
            RCLCPP_ERROR(this->get_logger(), "%s: Execution failed", error_prefix.c_str());
            return false;
        }

        return true;
    }

    /**
     * @brief Setup orientation constraints for Cartesian path
     */
    void setup_orientation_constraints(
        moveit::planning_interface::MoveGroupInterface& move_group,
        const geometry_msgs::msg::Pose& reference_pose)
    {
        moveit_msgs::msg::OrientationConstraint ocm;
        ocm.link_name = "tool0";
        ocm.header.frame_id = move_group.getPlanningFrame();
        ocm.orientation = reference_pose.orientation;
        ocm.absolute_x_axis_tolerance = 0.01;
        ocm.absolute_y_axis_tolerance = 0.01;
        ocm.absolute_z_axis_tolerance = M_PI;
        ocm.weight = 1.0;

        moveit_msgs::msg::Constraints path_constraints;
        path_constraints.orientation_constraints.push_back(ocm);
        move_group.setPathConstraints(path_constraints);
    }

    /**
     * @brief Execute Cartesian path through waypoints
     */
    bool execute_cartesian_path(
        moveit::planning_interface::MoveGroupInterface& move_group,
        const std::vector<geometry_msgs::msg::Pose>& waypoints,bool run_fast,
        std::string& message)
    {
        RCLCPP_INFO(this->get_logger(),
                   "Computing Cartesian path with %zu waypoints...", waypoints.size());

        moveit_msgs::msg::RobotTrajectory trajectory;
        const double jump_threshold = 0.0; // Prevent joint jumps

        double eef_step_final;
        if (run_fast){
            eef_step_final = eef_step_ * 20;
        }
        else{
            eef_step_final = eef_step_;
        }

        double fraction = move_group.computeCartesianPath(
            waypoints,
            eef_step_final,
            jump_threshold,
            trajectory,
            true // avoid_collisions
        );

        move_group.clearPathConstraints();

        RCLCPP_INFO(this->get_logger(),
                   "Cartesian path computed: %.2f%% achieved", fraction * 100.0);

        if (fraction < cartesian_fraction_threshold_) {
            message = "Failed to compute at least " +
                     std::to_string(static_cast<int>(cartesian_fraction_threshold_ * 100)) +
                     "% of the Cartesian path. Achieved: " +
                     std::to_string(static_cast<int>(fraction * 100)) + "%";
            RCLCPP_ERROR(this->get_logger(), "%s", message.c_str());
            return false;
        }

        // Execute the trajectory
        moveit::planning_interface::MoveGroupInterface::Plan cartesian_plan;
        cartesian_plan.trajectory_ = trajectory;

        if (move_group.execute(cartesian_plan) != moveit::core::MoveItErrorCode::SUCCESS) {
            message = "Cartesian path execution failed";
            RCLCPP_ERROR(this->get_logger(), "%s", message.c_str());
            return false;
        }

        message = "Path execution complete and successful";
        RCLCPP_INFO(this->get_logger(), "%s", message.c_str());
        return true;
    }

    /**
     * @brief Main service callback handler
     */
    void handle_draw_stroke_request(
        const std::shared_ptr<ur_draw_cmake::srv::DrawStroke::Request> request,
        std::shared_ptr<ur_draw_cmake::srv::DrawStroke::Response> response)
    {
        RCLCPP_INFO(this->get_logger(),
                   "Received draw stroke request with %zu poses",
                   request->target_poses.size());

        // Validate request
        if (request->target_poses.empty()) {
            RCLCPP_WARN(this->get_logger(), "Received empty pose list");
            response->success = true;
            response->message = "Received empty pose list. No movement executed.";
            return;
        }

        // Initialize MoveGroupInterface
        auto move_group = std::make_shared<moveit::planning_interface::MoveGroupInterface>(
            shared_from_this(), planning_group_);
        move_group->setPoseReferenceFrame("world");
        move_group->setNumPlanningAttempts(20);

        // Get current and target poses
        geometry_msgs::msg::Pose current_pose = get_current_pose(*move_group);
        const auto& first_pose = request->target_poses[0];
        auto raised_first_pose = create_raised_pose(first_pose);

        // Calculate distance to determine approach strategy
        double distance = calculate_distance(current_pose, raised_first_pose);
        RCLCPP_INFO(this->get_logger(),
                   "Distance to first target: %.4f m (threshold: %.4f m)",
                   distance, distance_threshold_);

        bool used_standard_planning = false;

        // Phase 1: Approach start position if far away
        if ((distance > distance_threshold_) || first_move_) {
            RCLCPP_INFO(this->get_logger(),
                       "Phase 1: Using standard planning to approach start pose");

            if (!move_to_pose(*move_group, raised_first_pose,
                            "Failed to reach raised initial pose")) {
                response->success = false;
                response->message = "Failed to reach the raised initial pose with standard planning";
                return;
            }
            first_move_ = false;

            used_standard_planning = true;
            RCLCPP_INFO(this->get_logger(), "Successfully reached raised start position");
        } else {
            RCLCPP_INFO(this->get_logger(),
                       "Phase 1: Distance within threshold. Using Cartesian path only");
        }

        // Phase 2: Execute drawing path
        RCLCPP_INFO(this->get_logger(), "Phase 2: Executing drawing path");

        if (!used_standard_planning) {
            std::vector<geometry_msgs::msg::Pose> prior_waypoints;
            std::string prior_message;
            // prior_waypoints.push_back(current_pose);
            prior_waypoints.push_back(raised_first_pose);
            setup_orientation_constraints(*move_group, raised_first_pose);
            bool sub_success = execute_cartesian_path(*move_group, prior_waypoints,true ,prior_message);
            RCLCPP_INFO(this->get_logger(), "Submove message: %s", prior_message.c_str());
            if (!sub_success){
                RCLCPP_WARN(this->get_logger(), "Cartesian first move failed trying direct move to pose");
                if (!move_to_pose(*move_group, raised_first_pose,
                                "Failed to reach raised initial pose")) {
                    response->success = false;
                    response->message = "Failed to reach the raised initial pose with standard planning";
                    return;
                }
            }
        }
        // Setup orientation constraints
        setup_orientation_constraints(*move_group, first_pose);

        // Build waypoint list
        std::vector<geometry_msgs::msg::Pose> waypoints;

        // Add all target poses
        for (const auto& pose : request->target_poses) {
            waypoints.push_back(pose);
        }

        // Raise pen at the end
        auto final_raised_pose = create_raised_pose(request->target_poses.back());
        waypoints.push_back(final_raised_pose);

        // Execute the Cartesian path
        std::string message;
        response->success = execute_cartesian_path(*move_group, waypoints, false ,message);
        response->message = message;
    }
};

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);

    auto node = std::make_shared<DrawStrokeServer>();

    // Create executor and spin in a separate thread
    rclcpp::executors::SingleThreadedExecutor executor;
    executor.add_node(node);
    std::thread spinner = std::thread([&executor]() { executor.spin(); });

    RCLCPP_INFO(node->get_logger(), "Executor spinning in background thread");

    // Wait for planning scene to be ready, then add collision objects
    rclcpp::sleep_for(std::chrono::seconds(2));
    node->add_collision_objects();

    // Wait for the spinner thread to finish (will run until shutdown)
    spinner.join();

    rclcpp::shutdown();
    return 0;
}