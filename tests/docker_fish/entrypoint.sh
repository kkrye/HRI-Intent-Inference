#!/bin/bash

# --- Venv Activation ---
VENV_DIR="/opt/venv"
if [ -f "$VENV_DIR/bin/activate" ]; then
    echo "Activating Python virtual environment..."
    # Venv activation MUST be done here, as it's a bash script
    source "$VENV_DIR/bin/activate"
fi

# The shell the user requested is passed as $1
REQUESTED_SHELL=$(basename "$1")

# --- Shell Execution ---

if [ "$REQUESTED_SHELL" = "fish" ]; then
    echo "Launching Fish shell with pre-configured ROS 2 environment..."
    # The 'ros2.fish' plugin (installed during build) handles sourcing on Fish startup.
    fish -c "fisher install edc/bass"
    fish -c "fisher install kpbaks/ros2.fish"
    exec "$@"

elif [ "$REQUESTED_SHELL" = "bash" ]; then
    echo "Launching Bash shell with sourced ROS 2 environment..."

    # Manually source ROS 2 for the bash session
    if [ -f "/opt/ros/jazzy/setup.bash" ]; then
        source /opt/ros/jazzy/setup.bash
    fi

    # Execute the command passed to the container for bash
    exec "$@"
else
    echo "Warning: Unknown shell or command requested. Proceeding..."
    exec "$@"
fi