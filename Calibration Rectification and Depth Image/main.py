import numpy as np
import numpy.linalg as la
import cv2
import math
from part1_functions import *    
from part2_functions import *  
import matplotlib.pyplot as plt
 

# Nicholas Novak
# ENPM 673 Spring 2023

# File contains:
# Main loop for running stereo vision steps

if __name__ == '__main__':
    
    
    
    
    """---------Pre-First Step processes to define dataset---------"""
    
    image_number = int(input("What image set would you like?\n \t1. Artroom\n\t2.Chessboard\n\t3. Ladder\n\tYour input: "))
    # for dataset_number in range(1,4):
    # We will define constants from the [calib.txt] file in each of the folders based on the input
    if image_number == 1:
        k_mat_artroom1 = np.array([[1733.74, 0, 792.27], [0, 1733.74, 541.89], [0, 0, 1]]).reshape(3,3)
        k_mat_artroom2 = np.array([[1733.74, 0, 792.27], [0, 1733.74, 541.89], [0, 0, 1]]).reshape(3,3)
        focal_artroom = 1733.74
        base_artroom = 536.62
        image_location = './artroom/'
        file_location_1 = './artroom/im0.png'
        file_location_2 = './artroom/im1.png'    
        given_disparity = 170
        k_mat_1 = k_mat_artroom1
        k_mat_2 = k_mat_artroom2
        base = base_artroom
        focal = focal_artroom
    if image_number == 2:
        k_mat_chess1 = np.array([[1758.23, 0, 829.15], [0, 1758.23, 552.78], [0, 0, 1]]).reshape(3,3)
        k_mat_chess2 = np.array([[1758.23, 0, 829.15], [0, 1758.23, 552.78], [0, 0, 1]]).reshape(3,3)
        focal_chess = 1758.23
        base_chess = 97.99
        image_location = './chess/'
        file_location_1 = './chess/im0.png'
        file_location_2 = './chess/im1.png'    
        given_disparity = 220
        k_mat_1 = k_mat_chess1
        k_mat_2 = k_mat_chess2
        base = base_chess
        focal = focal_chess
    if image_number == 3:
        k_mat_ladder1 = np.array([[1734.16, 0, 333.49], [0, 1734.16, 958.05], [0, 0, 1]]).reshape(3,3)
        k_mat_ladder2 = np.array([[1734.16, 0, 333.49], [0, 1734.16, 958.05], [0, 0, 1]]).reshape(3,3)
        focal_ladder = 1734.16
        base_ladder = 228.38
        image_location = './ladder/'
        file_location_1 = './ladder/im0.png'
        file_location_2 = './ladder/im1.png'    
        given_disparity = 110
        k_mat_1 = k_mat_ladder1
        k_mat_2 = k_mat_ladder2
        base = base_ladder
        focal = focal_ladder
        
    # Read images and put them into an array for processing:
    image_list = []
    for im_name in range(2):
        image_path = image_location + "/im" + str(im_name) + ".png"
        img = cv2.imread(image_path)
        image_list.append(img)
        
        
        
        
        
    """---------Part 1: Calibration---------"""
    
    numIter = 1000
    orb = cv2.ORB_create(numIter)
    im1 = image_list[0].copy()
    im2 = image_list[1].copy()
    im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY) 
    im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    kp1, des1 = orb.detectAndCompute(im1_gray, None)
    kp2, des2 = orb.detectAndCompute(im2_gray, None)
    # Perform matching
    bf_match = cv2.BFMatcher()
    matches = bf_match.match(des1,des2)
    matches = sorted(matches, key = lambda x :x.distance)
    first_matches = matches[0:100]

    # Match the feature pairs:
    matched_pairs = []
    pts1 = []
    pts2 = []
    for _, point in enumerate(first_matches):
        pt1 = kp1[point.queryIdx].pt
        pt2 = kp2[point.trainIdx].pt
        pts1.append(pt1)
        pts2.append(pt2)
        matched_pairs.append([pt1[0], pt1[1], pt2[0], pt2[1]])
    matched_pairs = np.array(matched_pairs).reshape(-1, 4)

    # Display matches on images:
    im1, im2 = resize_images([im1, im2])
    combined_imgs = np.concatenate((im1, im2), axis = 1)

    # May need null-check here
    corners_1_x = matched_pairs[:,0].copy().astype(int)
    corners_1_y = matched_pairs[:,1].copy().astype(int)
    corners_2_x = matched_pairs[:,2].copy().astype(int)
    corners_2_y = matched_pairs[:,3].copy().astype(int)
    corners_2_x += im1.shape[1]

    for i in range(corners_1_x.shape[0]):
        cv2.line(combined_imgs, (corners_1_x[i], corners_1_y[i]), (corners_2_x[i] ,corners_2_y[i]), [0,255,255], 2)
    plt.imshow(combined_imgs)
    cv2.imwrite("ORB_Results_"+image_location[2:-1]+".png", combined_imgs)

    # Find the fundamental and essential matrices, 
    # and use the RANSAC method for the fundamental matrix estimation:
    
    iterations = 980 # Tune this parameter
    thresh = 0.01 # Tune this too
    inliers_cap = 0
    results = []
    fund_matrix = []

    # Looking for error:
    for _ in range(0, iterations):
        indices = []

        # Select 8 random points:
        rand_feature = matched_pairs[np.random.choice(matched_pairs.shape[0], size=8), :] 
       
        # Get F matrix from the points:
        f_matrix_resulting = find_fundamental_matrix(rand_feature)
       
        for y in range(matched_pairs.shape[0]):
            first_feat = matched_pairs[y][0:2]
            second_feat = matched_pairs[y][2:4]
            err = np.dot(np.transpose(np.array([first_feat[0], first_feat[1], 1])), np.dot(f_matrix_resulting, np.array([second_feat[0], second_feat[1], 1])))
            err = np.abs(err)
            if err < thresh:
                indices.append(y)

        if len(indices) > inliers_cap:
            inliers_cap = len(indices)
            results = indices
            fund_matrix = f_matrix_resulting
    matched_pairs_inliers = matched_pairs[results, :]
    # print(matched_pairs_inliers)

    print("Computed fundamental matrix to be:\n", fund_matrix)

    # Essential matrix:
    ess_matrix = find_essential_matrix(k_mat_1, fund_matrix)
    print("Computed essential matrix to be:\n", ess_matrix)

    # Decompose into rotation and translation
    rot_result, tran_result = find_pose(ess_matrix)
    processed_points = post_process_points(k_mat_1, matched_pairs_inliers, rot_result, tran_result)



    results1 = []
    results2 = []
    rot_id = np.identity(3)
    tran_zs = np.zeros((3,1))
    for i in range(len(processed_points)):
        projected_pts = processed_points[i]/processed_points[i][3, :] 
        pos_points_1 = find_positive_points(projected_pts, rot_result[i], tran_result[i])
        pos_points_2 = find_positive_points(projected_pts, rot_id, tran_zs)
        results2.append(pos_points_1)
        results1.append(find_positive_points(projected_pts, rot_id, tran_zs))

    results1 = np.array(results1)
    results2 = np.array(results2)

    
    best_rot, best_tran = get_best_guesses(results1, results2, rot_result, tran_result, processed_points)
    # Intersection of results above threshold value gives best guesses at rotation and translation
    
    print("Computed Rotation: \n", best_rot)
    print("Computed Translation:\n ",best_tran)

    # End of calibration step
    
    
    
    
    
    """---------Part 2: Rectification---------"""
    
    matchset1, matchset2 = matched_pairs_inliers[:,0:2], matched_pairs_inliers[:,2:4]
    lines1, lines2, unrectified_result = find_epilines(matchset1, matchset2, fund_matrix, im1, im2, False)

    # Write out the current, unrectified code
    cv2.imwrite(image_location + "epipolarLines_"+image_location[2:-1]+"_unrectified.png", unrectified_result)
    height1, width1 = im1.shape[:2]
    height2, width2 = im2.shape[:2]
    # Algorithm that performs stereo rectification. Does not compute stereo correspondences as far as I am aware
    _, homog1, homog2 = cv2.stereoRectifyUncalibrated(np.float32(matchset1), np.float32(matchset2), fund_matrix, imgSize=(width1, height1))
    print("Image 1 Homography is:\n", homog1)
    print("Image 2 Homography is:\n", homog2)


    # Get new fundamental matrix through linalg methods:
    homog2_trans_i = la.inv(homog2.T)
    homog1_i = la.inv(homog1)
    rectified_fund_matrix = np.dot(homog2_trans_i, np.dot(fund_matrix, homog1_i))

    # Transform images with the new homoghraphy matrices
    set1_rect = cv2.perspectiveTransform(matchset1.reshape(-1, 1, 2), homog1).reshape(-1,2)
    set2_rect = cv2.perspectiveTransform(matchset2.reshape(-1, 1, 2), homog2).reshape(-1,2)

    # Draw rectified epilines:
    img1_rect = cv2.warpPerspective(im1, homog1, (width1, height1))
    img2_rect = cv2.warpPerspective(im2, homog2, (width2, height2))
    lines1_rectified, lines2_recrified, result_rec = find_epilines(set1_rect, set2_rect, rectified_fund_matrix, img1_rect, img2_rect, True)
    cv2.imwrite(image_location + "epipolarLines_"+image_location[2:-1]+"_rectified.png", result_rec)
    # End of part 2
    
    
    
    """---------Part 3: Correspondence---------"""
    
    gray_im1_rect = cv2.cvtColor(img1_rect,cv2.COLOR_BGR2GRAY)
    gray_im2_rect = cv2.cvtColor(img2_rect,cv2.COLOR_BGR2GRAY)

    disparities = 13 # Tunable parameter for each iteration
    pix_size = 4 # PIxel size from the first image, tunable as well

    h, w = gray_im1_rect.shape
    disparity_res = np.zeros(shape = (h,w))


    # SSD implementation
    # This takes a significant amount of time
    print("Computing correspondence section, it will take a while.")
    # Known method for constructing SSD matrix:
    for x in range(pix_size, gray_im1_rect.shape[0] - pix_size - 1):
        for y in range(pix_size + disparities, gray_im1_rect.shape[1] - pix_size - 1):
            ssd_matrix = np.empty([disparities, 1])
            left = gray_im1_rect[(x - pix_size):(x + pix_size), (y - pix_size):(y + pix_size)]
            h, w = left.shape
            for disp in range(0, disparities):
                right = gray_im2_rect[(x - pix_size):(x + pix_size), (y - disp - pix_size):(y - disp + pix_size)]
                ssd_matrix[disp] = np.sum((left[:,:]-right[:,:])**2)
            disparity_res[x, y] = np.argmin(ssd_matrix)
    print("Computed Disparity to be:\n",ssd_matrix)
    # Rescale SSD matrix:
    img_with_disparity = ((disparity_res/disparity_res.max())*255).astype(np.uint8)
    # Write out disparity image
    cv2.imwrite(image_location + "Disparity_grayscale_"+image_location[2:-1]+".png", img_with_disparity)

    # Build heatmap:
    cmap = plt.get_cmap('viridis')
    img_hmap = (cmap(img_with_disparity) * 2**16).astype(np.uint16)[:,:,:3]
    img_hmap = cv2.cvtColor(img_hmap, cv2.COLOR_RGB2BGR)
    # Write out heatmap disparity image
    cv2.imwrite(image_location + "Disparity_heatmap_"+image_location[2:-1]+".png", img_hmap)
    print("Correspondence complete and images written.")
    # Part 3 complete
        
    
    
    
    """---------Part 4: Depth Image---------"""	
    
    z_dist = np.zeros(shape=gray_im1_rect.shape).astype(float)
    z_dist[img_with_disparity > 0] = (focal * base) / (img_with_disparity[img_with_disparity > 0])


    # Rescale again
    depth_applied = ((z_dist/z_dist.max())*255).astype(np.uint8)
    # Write out depth image
    cv2.imwrite(image_location + "Depth_grayscale_"+image_location[2:-1]+".png", depth_applied)

    # Reinitialize cmap in case it was lost or overwritten
    cmap = plt.get_cmap('viridis')
    hmap_depth_applied = (cmap(depth_applied) * 2**16).astype(np.uint16)[:,:,:3]
    hmap_depth_applied  = cv2.cvtColor(hmap_depth_applied, cv2.COLOR_RGB2BGR)

    # Write out heatmap depth image
    cv2.imwrite(image_location + "Depth_heatmap_"+image_location[2:-1]+".png", hmap_depth_applied)
    print("Cleaning up and exiting computation flow...")
    # Part 4 complete
    
    
    
