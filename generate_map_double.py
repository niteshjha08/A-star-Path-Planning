#!/usr/bin/python3
import numpy as np
import cv2


# Define a line dividing map array in two parts, color one in white, other in black
def half_plane(map_arr,pt1,pt2,right_side_color=True):
    # equation: y = mx + c
    map_arr = np.zeros((502,802,3),dtype='uint8')
    m = (pt2[1]-pt1[1])/(pt2[0]-pt1[0]+1e-6)
    c = pt1[1] - m * pt1[0]
    x= np.arange((map_arr.shape[1]))
    y= np.arange((map_arr.shape[0]))
    xx, yy = np.meshgrid(x,y)
    z = yy - m*xx - c
    if right_side_color:
        map_arr[z>0] = [255,255,255]
    else:
        map_arr[z<=0] = [255,255,255]
    return map_arr

def define_circle():
    map_arr = np.zeros((502,802,3),dtype='uint8')
    center = [601,133]
    radius = 80
    x= np.arange((map_arr.shape[1]))
    y= np.arange((map_arr.shape[0]))
    xx, yy = np.meshgrid(x,y)
    z = (xx-center[0])**2 + (yy-center[1])**2 - radius**2
    map_arr[z>0] = [255,255,255]

    return map_arr

def define_hexagon(): 
    map_arr = np.zeros((502,802,3),dtype='uint8')
    hexagon = np.array([[471,341],[471,261],[401,221],[331,261],[331,341],[401,381]])
    side1 = half_plane(map_arr,hexagon[0],hexagon[1])
    side2 = cv2.bitwise_or(half_plane(map_arr,hexagon[2],hexagon[1],False),side1)
    side3 = cv2.bitwise_or(half_plane(map_arr,hexagon[2],hexagon[3],False),side2)
    side4 = cv2.bitwise_or(half_plane(map_arr,hexagon[3],hexagon[4]),side3)
    side5 = cv2.bitwise_or(half_plane(map_arr,hexagon[5],hexagon[4]),side4)
    hex = cv2.bitwise_or(half_plane(map_arr,hexagon[0],hexagon[5]),side5)
    return hex

def define_concave_shape():
    shape = np.array([[73,151],[231,81],[161,141],[211,301]])
    map_arr = np.zeros((502,802,3),dtype='uint8')
    line1 = half_plane(map_arr,shape[0],shape[1])
    line2 =half_plane(map_arr,shape[1],shape[2])
    line3 =half_plane(map_arr,shape[2],shape[3],False)
    line10=cv2.bitwise_and(line2,line3)
    line10 = cv2.bitwise_not(line10)
    line11=cv2.bitwise_and(line1,line10)
    line4 = half_plane(map_arr,shape[0],shape[3],False)
    line12=cv2.bitwise_and(line4,line11)
    line12=cv2.bitwise_not(line12)
    return line12

def define_obstacle_space():
    hex = define_hexagon()
    circle = define_circle()
    concave_shape = define_concave_shape()

    map_arr = cv2.bitwise_and(hex,circle)
    map_arr = cv2.bitwise_and(map_arr,concave_shape)
    map_arr[0,:,:] = [0,0,0]
    map_arr[501,:,:] = [0,0,0]

    map_arr[:,0,:] = [0,0,0]
    map_arr[:,801,:] = [0,0,0]

    return map_arr

def main():
    map_arr = define_obstacle_space()

    cv2.imshow('map',map_arr)
    cv2.waitKey()


if __name__=="__main__":
    main()
