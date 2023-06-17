import numpy as np
import numpy.linalg as la
import cv2
import math
from part1_functions import *    


# Nicholas Novak
# ENPM 673 Spring 2023

# File contains:
# Further reference functions for use in in rectification (part 2)

def x_value_of_line(input, y):
    # Known method for finding x value from a y avlue and line equation
    output = -(input[1]*y + input[2])/input[0]
    return output

def find_epilines(pointset1, pointset2, fundamental_matrix, image0, image1, is_rect):
    base_lines1, base_lines2 = [], []
    img_epi1 = image0.copy()
    img_epi2 = image1.copy()

    for i in range(pointset1.shape[0]):
        xset_1 = np.array([pointset1[i,0], pointset1[i,1], 1]).reshape(3,1)
        xset_2 = np.array([pointset2[i,0], pointset2[i,1], 1]).reshape(3,1)

        line2 = np.dot(fundamental_matrix, xset_1)
        base_lines2.append(line2)

        line1 = np.dot(fundamental_matrix.T, xset_2)
        base_lines1.append(line1)

        #solve for in and max x values based on equation of line
        if (not is_rect):
            y2_min = 0
            y2_max = image1.shape[0]
            x2_min = x_value_of_line(line2, y2_min)
            x2_max = x_value_of_line(line2, y2_max)

            y1_min = 0
            y1_max = image0.shape[0]
            x1_min = x_value_of_line(line1, y1_min)
            x1_max = x_value_of_line(line1, y1_max)
        else:
            x2_min = 0
            x2_max = image1.shape[1] - 1
            y2_min = -line2[2]/line2[1]
            y2_max = -line2[2]/line2[1]

            x1_min = 0
            x1_max = image0.shape[1] -1
            y1_min = -line1[2]/line1[1]
            y1_max = -line1[2]/line1[1]

        # Draw circles to show points along epilines
        cv2.circle(img_epi2, (int(pointset2[i,0]),int(pointset2[i,1])), 10, (255,0,255), -1)
        img_epi2 = cv2.line(img_epi2, (int(x2_min), int(y2_min)), (int(x2_max), int(y2_max)), (255, 255, int(i*2.55)), 2)
    

        cv2.circle(img_epi1, (int(pointset1[i,0]),int(pointset1[i,1])), 10, (255,0,255), -1)
        img_epi1 = cv2.line(img_epi1, (int(x1_min), int(y1_min)), (int(x1_max), int(y1_max)), (255, 255, int(i*2.55)), 2)

    #Resize images and concatenate back together
    image_1, image_2 = resize_images([img_epi1, img_epi2])
    result_image = np.concatenate((image_1, image_2), axis = 1)
    result_image = cv2.resize(result_image, (1920, 660))

    return base_lines1, base_lines2, result_image