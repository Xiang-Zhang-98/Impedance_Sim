import socket
import numpy as np
import struct
import time


class robot_controller:
    def __init__(self):
        self.UDP_IP_IN = (
            "192.168.1.200"  # Ubuntu IP, should be the same as Matlab shows
        )
        self.UDP_PORT_IN = (
            57832  # Ubuntu receive port, should be the same as Matlab shows
        )
        self.UDP_IP_OUT = (
            "192.168.1.90"  # Target PC IP, should be the same as Matlab shows
        )
        self.UDP_PORT_OUT = 3828  # Robot 1 receive Port

        # self.s_in = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # self.s_in.bind((self.UDP_IP_IN, self.UDP_PORT_IN))
        # Receive TCP position (3*), TCP Rotation Matrix (9*), TCP Velcoity (6*), Force Torque (6*)
        self.unpacker = struct.Struct("12d 6d 6d")

        self.robot_pose, self.robot_vel, self.TCP_wrench = None, None, None

        
    def receive(self):
        self.s_in = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.s_in.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.s_in.bind((self.UDP_IP_IN, self.UDP_PORT_IN))
        data, _ = self.s_in.recvfrom(1024)
        unpacked_data = np.array(self.unpacker.unpack(data))
        self.robot_pose, self.robot_vel, self.TCP_wrench = (
            unpacked_data[0:12],
            unpacked_data[12:18],
            unpacked_data[18:24]
        )
        self.s_in.close()
        

    def send(self, udp_cmd):
        '''
        UDP command 1~6 TCP desired Position Rotation
        UDP desired vel 7~12 
        UDP Kp 13~18
        UDP Kd 19~24
        UDP Mass 25~27
        UDP Interial 28~30
        '''
        self.s_out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp_cmd = udp_cmd.astype("d").tostring()
        self.s_out.sendto(udp_cmd, (self.UDP_IP_OUT, self.UDP_PORT_OUT))
        self.s_out.close()

if __name__ == "__main__":
    controller = robot_controller()
    controller.send(2*np.ones(24))
    print(1)