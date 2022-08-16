import os
import numpy as np
import matplotlib.pyplot as plt
joint_orders_kinect = [[0,1], [1,2], [2,3], [2,4], [2,8], [4,5], [5,6], [6,7], [8,9], [9,10], [10,11], 
                 [0,12], [0,16], [12,13], [13,14], [14,15], [16,17], [17,18], [18,19]]
joint_orders_mp = [[0, 1], [1, 2], [2, 3], [3, 7], [0, 4], [4, 5], [5, 6], [6, 8], [9, 10], [11, 12], [12, 14], [14, 16],
                 [16, 22], [16, 18], [16, 20], [18, 20], [11, 13], [13, 15], [15, 21], [15, 17], [15, 19], [17, 19], [12, 24],
                 [24, 26], [26, 28], [28, 30], [28, 32], [30, 32], [11, 23], [23, 25], [25, 27], [27, 29], [27, 31], [29, 31],
                 [23, 24]]

def kinect2mp_spec_joint(mp, joint1, joint2):
    kinect = np.zeros(3, dtype=np.float32)
    kinect[0] = (mp[joint1][0] + mp[joint2][0]) / 2
    kinect[1] = (mp[joint1][1] + mp[joint2][1]) / 2
    kinect[2] = (mp[joint1][2] + mp[joint2][2]) / 2
    return kinect

def mp2kinect(mp):
    kinect2mp_list = [[3,0], [4,11], [5,13], [6,15], [8,12], [9,14], [10,16], [12,23],
    [13,25], [14,27], [15,31], [16,24], [17,26], [18,28], [19,32]]
    kinect = np.zeros((20,3), dtype=np.float32)
    for jointID in kinect2mp_list:
        kinect[jointID[0]] = mp[jointID[1]]
    kinect[0] = kinect2mp_spec_joint(mp, 23, 24)
    kinect[2] = kinect2mp_spec_joint(mp, 11, 12)
    kinect[1] = kinect2mp_spec_joint(kinect, 0, 2)
    kinect[11] = kinect2mp_spec_joint(mp, 18, 20)
    kinect[7] = kinect2mp_spec_joint(mp, 17, 19)
    return kinect
    return kinect
def visualize(skeleton_data, joint_orders):
    x = []
    y = []
    plt.axis('equal')
    for coord in skeleton_data:
        coord = list(map(float, coord))
        x.append(coord[0])
        y.append(coord[1])
    plt.scatter(x, y, color = "green")
    for joint_order in joint_orders:
        x_coord = [x[joint_order[0]], x[joint_order[1]]]
        y_coord = [y[joint_order[0]], y[joint_order[1]]]
        plt.plot(x_coord, y_coord, color=plt.cm.gray(0))
