# Use the official ROS 2 Jazzy desktop image as the base
FROM osrf/ros:jazzy-desktop

# Set environment variables for non-interactive install
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary packages: UR Driver, OpenCV, venv, and build tools
RUN apt-get update && apt-get install -y \
    ros-jazzy-ur-robot-driver \
    python3-opencv \
    python3-pip \
    python3-venv \
    fish \
    tmux \
    net-tools \
    iputils-ping \
    dos2unix \
    git \
    && rm -rf /var/lib/apt/lists/*

# --- Workspace and Venv Setup ---
WORKDIR /ros2_ws

# Create a directory for the virtual environment
ENV VENV_DIR=/opt/venv
RUN python3 -m venv $VENV_DIR

# Activate venv and install Python dependencies
# Note: We use $VENV_DIR/bin/pip to ensure we use the venv's pip
RUN $VENV_DIR/bin/pip install --no-cache-dir \
    numpy
    # Add other Python packages here as needed (e.g., tensorflow, torch, etc.)


RUN fish -c "curl -sL https://git.io/fisher | source && fisher install jorgebucaran/fisher"
# RUN fish -c "fisher install edc/bass"
# RUN fish -c "fisher install kpbaks/ros2.fish"


# Copy custom entrypoint script
COPY ../entrypoint.sh /usr/local/bin/entrypoint.sh
RUN dos2unix /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
# ENTRYPOINT ["../ros_entrypoint.sh"]

# Default command: keeps the container running and ready for interactive use
CMD ["bash"]