# Adversarial Human–Robot Co-Painting System

An end-to-end human–robot interaction platform for studying how an AI-guided robot responds to a person's drawing with collaborative, adversarial, or visually divergent additions.

The system captures a physical canvas, generates a constrained response with Google Gemini, converts the generated marks into drawable strokes, and commands a UR5 robot through ROS 2 and MoveIt 2.

## Overview

The project supports turn-based co-painting between a person and a robot:

1. A camera observes the drawing surface.
2. Four ArUco markers locate the canvas and correct its perspective.
3. The current drawing is sent to Gemini with a selected interaction condition.
4. Gemini preserves the original image and proposes a small set of new line-based strokes.
5. Image-processing code isolates the generated strokes and converts them into paths.
6. A ROS 2 action server maps the paths onto the physical page.
7. MoveIt 2 plans and executes the corresponding UR5 trajectories.

The repository also contains tools for generating controlled experimental stimuli, administering pairwise-preference surveys, and analyzing participant responses.

## Key Features

- End-to-end integration of computer vision, generative AI, and robot motion planning
- Real-time canvas capture with OpenCV
- ArUco-marker detection and perspective correction
- Prompt-controlled visual and semantic variation
- Collaborative, adversarial, antagonistic, and custom interaction conditions
- Separation of human and AI-generated marks by color
- Raster-to-vector conversion for generated line art
- ROS 2 action interface with progress feedback and cancellation
- MoveIt 2 trajectory planning for a UR5 manipulator
- Support for real-hardware and simulated-robot operation
- Browser-based pairwise-preference survey built with NiceGUI
- Survey analysis using Krippendorff's alpha and Bradley–Terry ranking
- Reproducible development environment using Docker and VS Code Dev Containers

## System Pipeline

```text
Human drawing
      ↓
Camera capture and ArUco detection
      ↓
Perspective-corrected canvas image
      ↓
Gemini image generation
      ↓
Generated-stroke isolation
      ↓
Raster-to-vector path conversion
      ↓
ROS 2 DrawStrokes action
      ↓
MoveIt 2 trajectory planning
      ↓
UR5 drawing execution
```

## Interaction Conditions

Generated responses can be controlled along two axes:

| Axis | Levels |
|---|---|
| Visual similarity | Similar, neutral, different |
| Semantic similarity | Similar, neutral, different |

The generation tools can evaluate combinations of these levels to create stimuli for controlled HRI experiments.

Predefined conditions include:

- **Collaborative:** visually and semantically similar
- **Adversarial:** visually related but semantically different
- **Antagonistic:** visually and semantically different
- **Custom:** any supported combination of visual and semantic similarity

## Repository Structure

```text
image-generation/
  robot_co_painter.py       End-to-end camera, Gemini, and ROS integration
  co_painter.py             Turn-based AI co-painting without robot control
  co_painter_memory.py      Multi-turn co-painting with model context
  paint_with_gemini.py      Gemini request and image-processing utilities
  image_to_svg.py           Raster-to-vector stroke conversion
  config.py                 Canvas and prompting configuration
  brute_force_prompting/    Experimental stimulus generation

ros2_ws/src/ur5_draw/
  draw_action.py            ROS 2 DrawStrokes action server
  draw_node.py              Drawing client and robot-control logic
  test_action_client.py     Image-to-action test client
  launch/ur5_draw.launch.py UR driver, MoveIt, frames, and drawing nodes

ros2_ws/src/ur_draw_cmake/
  action/DrawStrokes.action Custom multi-stroke action definition
  srv/DrawStroke.srv        Motion-planning service definition
  src/moveit_service.cpp    MoveIt trajectory-planning service

survey/
  survey_webapp.py          Browser-based pairwise-comparison study
  analysis_survey.py        Agreement and preference analysis

tests/                      Camera, robot-control, and model experiments

HRI_Project_Paper           Unpublished preliminary findings
```

## Requirements

### AI and Image Processing

- Python 3
- OpenCV
- NumPy
- Pillow
- Google Generative AI SDK
- svgwrite
- potracer
- scikit-image

### Robotics

- ROS 2
- MoveIt 2
- Universal Robots ROS 2 Driver
- UR simulation packages for simulated operation
- A supported Universal Robots manipulator, such as the UR5, for hardware operation

### Survey and Analysis

- NiceGUI
- pandas
- PyTorch
- Matplotlib

## Security Setup

Load the Gemini API key from an environment variable instead:

```python
import os

API_KEY = os.environ["GEMINI_API_KEY"]
```

Set the variable before starting the co-painter:

```bash
export GEMINI_API_KEY="your-api-key"
```

## Installation

### 1. Install the Python Dependencies

```bash
cd image-generation
python3 -m pip install -r requirements.txt
```

Install the additional survey dependencies:

```bash
python3 -m pip install nicegui pandas torch matplotlib
```

### 2. Build the ROS 2 Workspace

From the repository root:

```bash
cd ros2_ws
colcon build --symlink-install
source install/setup.bash
```

The necessary ROS 2, MoveIt 2, and Universal Robots packages must already be installed. The included Dev Container files can be used to reproduce the intended development environment.

## Running the Robot System

### Simulation

```bash
cd ros2_ws
source install/setup.bash
ros2 launch ur5_draw ur5_draw.launch.py enable_sim:=true
```

### Real UR5 Hardware

```bash
cd ros2_ws
source install/setup.bash
ros2 launch ur5_draw ur5_draw.launch.py \
  enable_sim:=false \
  robot_ip:=<ROBOT_IP> \
  ur_type:=ur5
```

Before commanding the physical robot, verify the configured transforms, page dimensions, pen height, workspace clearance, safety limits, and emergency-stop access. Start at reduced speed and supervise all motion.

### Send a Test Drawing

After launching the robot stack, send an image through the drawing action client:

```bash
ros2 run ur5_draw test <IMAGE_PATH>
```

### Run the End-to-End Co-Painter

From the `image-generation/` directory, with the ROS workspace sourced:

```bash
python3 robot_co_painter.py --action-server
```

Use a virtual drawing window instead of a camera:

```bash
python3 robot_co_painter.py --virtual-camera --action-server
```

### Controls

| Key | Action |
|---|---|
| `N` | Capture the canvas and begin the robot's next turn |
| `C` | Clear the working canvas |
| `S` | Save the current drawing |
| `Q` or `Esc` | Quit |

Camera mode expects four visible ArUco markers corresponding to the canvas corners. A robot turn will not begin until all four markers are detected.

## Running the Survey

Start the survey application:

```bash
cd survey
python3 survey_webapp.py
```

Open a participant-specific address in a browser:

```text
http://localhost:8080/<participant-id>
```

Each participant should use a unique identifier so their responses are stored separately.

Run the analysis tools with:

```bash
python3 analysis_survey.py
```

The analysis code includes pairwise-preference ranking with a Bradley–Terry model and inter-rater agreement measurement using Krippendorff's alpha.

## Technologies

- Python
- C++
- ROS 2
- MoveIt 2
- Universal Robots UR5
- OpenCV and ArUco markers
- Google Gemini
- NiceGUI
- PyTorch
- Docker and VS Code Dev Containers

## Research Context

This project explores how the degree of visual and semantic alignment in a robot's contribution affects the human experience of creative collaboration.

It combines a functional robotic co-painting system with controlled stimulus generation, participant data collection, and quantitative preference analysis.

## Acknowledgments

Developed as a team human–robot interaction research project. Third-party robot drivers, motion-planning packages, generative-model APIs, and supporting libraries remain subject to their respective licenses and terms.
