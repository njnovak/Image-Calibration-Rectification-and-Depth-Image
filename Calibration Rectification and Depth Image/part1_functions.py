import numpy as np
import numpy.linalg as la
import cv2
import math

# Nicholas Novak
# ENPM 673 Spring 2023

# File contains:
# Reference functions for use in calibration (part 1), and some in rectification (part 2)

def resize_images(imgs):
    size_base = []
    resized_results = []
    for image in imgs:
        x, y, z = image.shape
        size_base.append([x, y, z])

    sizes = np.array(size_base)
    x_target, y_target, _ = np.max(sizes, axis = 0)
    
    for i, image in enumerate(imgs):
        image_resized = np.zeros((x_target, y_target, sizes[i, 2]), np.uint8)
        image_resized[0:sizes[i, 0], 0:sizes[i, 1], 0:sizes[i, 2]] = image
        resized_results.append(image_resized)

    return resized_results

def normalize(x_value):
    
    x_y = np.mean(x_value, axis=0)
    x_x ,y_y = x_y[0], x_y[1]

    x_capacity = x_value[:,0] - x_x
    y_capacity = x_value[:,1] - y_y

    rms_result = np.sqrt(2/np.mean(x_capacity**2 + y_capacity**2))
    T_scale = np.diag([rms_result,rms_result,1])
    T_trans = np.array([[1,0,-x_x],[0,1,-y_y],[0,0,1]])
    norm_matrix = T_scale.dot(T_trans)

    x_stack = np.column_stack((x_value, np.ones(len(x_value))))
    x_norm = (norm_matrix.dot(x_stack.T)).T
    return  x_norm, norm_matrix

def find_fundamental_matrix(match_results):

    x1 = match_results[:,0:2]
    x2 = match_results[:,2:4]

    x1_norm, t_mat_1 = normalize(x1)
    x2_norm, t_mat_2 = normalize(x2)
    A_matrix = np.zeros((len(x1_norm),9))
    for i in range(0, len(x1_norm)):
        x_1,y_1 = x1_norm[i][0], x1_norm[i][1]
        x_2,y_2 = x2_norm[i][0], x2_norm[i][1]
        A_matrix[i] = np.array([x_1*x_2, x_2*y_1, x_2, y_2*x_1, y_2*y_1, y_2, x_1, y_1, 1])

    # Using SVD for solving for Fundamental Matrix
    _, _, VT = la.svd(A_matrix, full_matrices=True)
    F = VT.T[:, -1]
    F = F.reshape(3,3)

    u, s, vt = la.svd(F)
    s = np.diag(s)
    s[2,2] = 0
    F = np.dot(u, np.dot(s, vt))
    F = np.dot(t_mat_2.T, np.dot(F, t_mat_1))

    
    return F

def find_essential_matrix(K1, F): # Just need one camera matrix
    K = K1
    E = np.dot(np.dot(K.T,F),K)
    U,S,V = la.svd(E)
    S[2] = 0
    E = np.dot(U,np.dot(np.diag(S),V))
    return E

def find_pose(essential_matrix):
    U, _, V_T = la.svd(essential_matrix)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    rotation_results = []
    translation_results = []
    
    # Blocks of append below to build rotation and translation matrices
    rotation_results.append(np.dot(U, np.dot(W, V_T)))
    rotation_results.append(np.dot(U, np.dot(W, V_T)))
    rotation_results.append(np.dot(U, np.dot(W.T, V_T)))
    rotation_results.append(np.dot(U, np.dot(W.T, V_T)))
    
    translation_results.append(U[:, 2])
    translation_results.append(-U[:, 2])
    translation_results.append(U[:, 2])
    translation_results.append(-U[:, 2])

    for i in range(4):
        if (la.det(rotation_results[i]) < 0): # Check for negative determinants
            rotation_results[i] = -rotation_results[i]
            translation_results[i] = -translation_results[i]

    return rotation_results, translation_results

def post_process_points(K, inliers, rot_mat, trans_mat):
    points = []
    rot_mat_eye = np.identity(3)
    trans_mat_z = np.zeros((3,1))
    eye = np.identity(3)
    # Stack vals horixontally to get points:
    point1 = np.dot(K, np.dot(rot_mat_eye, np.hstack((eye, -trans_mat_z.reshape(3,1)))))

    

    for val in range(len(trans_mat)):
        transition_vector = -trans_mat[val].reshape(3,1)
        first_point = inliers[:,0:2].T
        second_point = inliers[:,2:4].T
        rotation_val = rot_mat[val]

        point2 = np.dot(K, np.dot(rotation_val, np.hstack((eye, transition_vector))))

        triangulated_x = cv2.triangulatePoints(point1, point2, first_point, second_point)  
        points.append(triangulated_x)
        
    return points

def find_positive_points(input_points, rot, c_mat):
    # Find amount of points in front of the camera for each iteration to see which has the best resutls
    # Best results are most positive
    fetched_point = np.dot(rot, np.hstack((np.identity(3), -c_mat.reshape(3,1))))
    fetched_point = np.vstack((fetched_point, np.array([0,0,0,1]).reshape(1,4)))
    n_positive = 0
    
    # Figure out which points lie in frot of the camera
    for point in range(input_points.shape[1]):
        x_point = input_points[:,point]
        x_point = x_point.reshape(4,1)
        z = (np.dot(fetched_point, x_point) / np.dot(fetched_point, x_point)[3])[2]
        if z > 0:
            n_positive += 1

    return n_positive

def get_best_guesses(results1, results2, rot_result, tran_result, processed_points):
        tr = int(processed_points[0].shape[1] / 2)
        res1 = np.intersect1d(np.where(results1 > tr), np.where(results2 > tr))
        res2 = np.intersect1d(np.where(results1 > tr), np.where(results2 > tr))
        best_rot = rot_result[res1[0]]
        best_tran = tran_result[res2[0]]
        return best_rot, best_tran