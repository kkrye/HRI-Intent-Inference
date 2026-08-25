# Author: Angelica Bonilla (@monacuca on Github, @abfominaya elsewhere)
# Simple python program for UR5 control over TCP/IP. 
import socket
import time

# Note: There is an actual python interface that calculates IK and poses through RoboDK
# and an RDTE protocol for TCP/IP control of the robot. This is in case you want to directly
# write/read instructions over TCP using UR5Script. 

# ----- URScript parsing helpers -----
# MoveL: Moves using unverse kinematics
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

# MoveJ: Directly controls joint rotation of the UR5
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
HOST = "169.254.154.223"

# UR5 R/obot port. DO NOT CHANGE THIS. 
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
# Basic movement commands, i.e. Rectangle:

# Home base:
moveJ(-0.1, -0.1, 0.2, 0, 1.75, 0, a, v, s)
time.sleep(4)

# Rectangle:
moveJ(-0.5, -0.1, 0.1, 0, 1.75, 0, a, v, s)
time.sleep(2)
moveJ(-0.2, -0.1, 0.1, 0, 1.75, 0, a, v, s)
time.sleep(2)
moveJ(-0.2, -0.4, 0.1, 0, 1.75, 0, a, v, s)
time.sleep(2)
moveJ(-0.5, -0.4, 0.1, 0, 1.75, 0, a, v, s)
time.sleep(2)
moveJ(-0.5, -0.1, 0.1, 0, 1.75, 0, a, v, s)
time.sleep(4)

# Home base:
moveJ(-0.5, -0.1, 0.2, 0, 1.75, 0, a, v, s)
time.sleep(4)

# Rectangle:
moveJ(-0.5, -0.1, 0.1, 0, 1.75, 0, a, v, s)
time.sleep(2)
moveJ(-0.2, -0.1, 0.1, 0, 1.75, 0, a, v, s)
time.sleep(2)
moveJ(-0.2, -0.4, 0.1, 0, 1.75, 0, a, v, s)
time.sleep(2)
moveJ(-0.5, -0.4, 0.1, 0, 1.75, 0, a, v, s)
time.sleep(2)
moveJ(-0.5, -0.1, 0.1, 0, 1.75, 0, a, v, s)
time.sleep(2)
data = s.recv(1024)

s.close()

print ("Received", repr(data))