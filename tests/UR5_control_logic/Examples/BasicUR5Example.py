# Author: Angelica Bonilla (@monacuca on Github, @abfominaya elsewhere)
# Simple python program for UR5 control over TCP/IP. 
import socket
import time

# Note: There is an actual python interface that calculates IK and poses through RoboDK
# and an RDTE protocol for TCP/IP control of the robot. This is in case you want to directly
# write/read instructions over TCP using UR5Script. 

# ----- URScript parsing helpers -----
# MoveL: Moves directly/linearly using inverse kinematics.
# Note: Since we are just giving it a pose, specifics don't matter.
# Note: MoveL is easier to control the specific path, but harder to work with.
# Use MoveJ whenever possible.
def moveL(x, y, z, rx, ry, rz, a, v, s):
    # Format pose for moveL command. 
    pose = ("p[" + str(x) + ", " \
            + str(y) + ", " \
            + str(z) + ", " \
            + str(rx) + ", " \
            + str(ry) + ", " \
            + str(rz) + "]")
    
    # Format message for moveL command: 
    message = ("movel(" + pose + ", a=" + str(a) + ", v=" + str(v) + ")" + "\n")
    # Send message (i.e. pose) through socket and encode
    print(message)
    s.send((message).encode('utf8'))

# MoveJ: Directly controls joint rotation of the UR5.
# Note: You might notice that the movement from one point to the other is not linear
# when using this.
# Use this whenever posible. 
def moveJ(x, y, z, rx, ry, rz, a, v, s):
    # Format pose for moveJ command. 
    pose = ("p[" + str(x) + ", " \
            + str(y) + ", " \
            + str(z) + ", " \
            + str(rx) + ", " \
            + str(ry) + ", " \
            + str(rz) + "]")
    
    # Format message for moveJ command: 
    message = ("movej(" + pose + ", a=" + str(a) + ", v=" + str(v) + ")"+ "\n")
    # Send message (i.e. pose) through socket and encode
    print(message)
    s.send((message).encode('utf8'))

# ----- Machine Set Up -----
# Robot IP Address.
# Note that to control the robot, it must have the corresponding DNS connection!
HOST = "169.254.154.223" # This is the IP address of the Studio's robot.

# UR5 Robot port. Probably don't change this.
PORT = 30002

# Set up socket. 
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST, PORT))

s.send(("set_digital_out(0,True)" + "\n").encode('utf8'))
s.send(("set_digital_out(0,False)" + "\n").encode('utf8'))


# ----- Acceleration and Velocity specs for machine -----
a=1.39626
v=1.04719

# Commands for the actual robot written in URScript.
# Basic movement commands: 

moveJ(-0.5, -0.5, 0.5, -1.0, 4.5, -1, a, v, s)
time.sleep(3)
moveJ(0.3, 0.3, -0.115, 0, 3, 0, a, v, s)

data = s.recv(1024)

s.close()

print ("Received", repr(data))