import cv2 as cv
import numpy as np
import time

#Depth Mode Flags
enable_sgbm = True
enable_sbm = True

#Load dataset images
left = 'images/stereoLeft/imageL0.png'
right = 'images/stereoRight/imageR0.png'

imgL = cv.imread(left)
imgR = cv.imread(right)

cv.imshow('left',imgL)
cv.imshow('right',imgR)

cv.waitKey(0)

#Load important parameters from file
cv_file = cv.FileStorage()

##Undistort and Rectification Map
cv_file.open('calib/stereoMap.xml', cv.FileStorage_READ)

stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()
Q = cv_file.getNode('q').mat()

cv_file.release()

##SGBM Calibration Data
cv_file.open('calib/sgbmCalib.txt',cv.FileStorage_READ)

sgbm_blockSize = int(cv_file.getNode('blockSize').real())
sgbm_numDisparities = int(cv_file.getNode('numDisparities').real())
sgbm_minDisparity = int(cv_file.getNode('minDisparity').real())
sgbm_uniquenessRatio = int(cv_file.getNode('uniquenessRatio').real())
sgbm_speckleWindow = int(cv_file.getNode('speckleWindowSize').real())
sgbm_speckleRange = int(cv_file.getNode('speckleRange').real())
sgbm_P1 = int(cv_file.getNode('P1').real())
sgbm_P2 = int(cv_file.getNode('P2').real())
sgbm_filter_lambda = cv_file.getNode('lambda').real()
sgbm_filter_sigma = cv_file.getNode('sigmaColor').real()

cv_file.release()

##SBM Calibration Data
cv_file.open('calib/sbmCalib.txt',cv.FileStorage_READ)

sbm_blockSize = int(cv_file.getNode('blockSize').real())
sbm_numDisparities = int(cv_file.getNode('numDisparities').real())
sbm_minDisparity = int(cv_file.getNode('minDisparity').real())
sbm_speckleWindow = int(cv_file.getNode('speckleWindowSize').real())
sbm_speckleRange = int(cv_file.getNode('speckleRange').real())
sbm_filter_lambda = cv_file.getNode('lambda').real()
sbm_filter_sigma = cv_file.getNode('sigmaColor').real()

cv_file.release()

#Initialize Stereo Matchers and Filters
sgbm = cv.StereoSGBM_create(minDisparity=sgbm_minDisparity,
							numDisparities=sgbm_numDisparities,
							blockSize=sgbm_blockSize,
							P1=sgbm_P1,
							P2=sgbm_P2,
							uniquenessRatio=sgbm_uniquenessRatio,
							speckleWindowSize=sgbm_speckleWindow,
							speckleRange=sgbm_speckleRange)

sgbm_filter = cv.ximgproc.createDisparityWLSFilter(matcher_left=sgbm)
sgbm_filter.setLambda(sgbm_filter_lambda)
sgbm_filter.setSigmaColor(sgbm_filter_sigma)

sbm = cv.StereoBM_create(sbm_numDisparities)
sbm.setBlockSize(sbm_blockSize)
sbm.setMinDisparity(sbm_minDisparity)
sbm.setSpeckleWindowSize(sbm_speckleWindow)
sbm.setSpeckleRange(sbm_speckleRange)

sbm_filter = cv.ximgproc.createDisparityWLSFilter(matcher_left=sbm)
sbm_filter.setLambda(sbm_filter_lambda)
sbm_filter.setSigmaColor(sbm_filter_sigma)

#Calculate Disparity Maps, Filter
imgR = cv.remap(imgR, stereoMapR_x, stereoMapR_y, cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)
imgL = cv.remap(imgL, stereoMapL_x, stereoMapL_y, cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)

lg = cv.cvtColor(imgL,cv.COLOR_BGR2GRAY)
rg = cv.cvtColor(imgR,cv.COLOR_BGR2GRAY)

##SGBM Left and Right
start_time = time.time()
sgbm_left = sgbm.compute(lg,rg)
sgbm_time = time.time() - start_time
start_time = time.time()
sgbm_right = cv.ximgproc.createRightMatcher(sgbm).compute(rg,lg)
sgbm_filtered = sgbm_filter.filter(sgbm_left,lg,disparity_map_right=sgbm_right)
sgbm_filter_time = time.time() - start_time

##SBM Right
start_time = time.time()
sbm_left = sbm.compute(lg,rg)
sbm_time = time.time()-start_time
start_time = time.time()
sbm_right = cv.ximgproc.createRightMatcher(sbm).compute(rg,lg)
sbm_filtered = sbm_filter.filter(sbm_left,lg,disparity_map_right=sbm_right)
sbm_filter_time = time.time() - start_time

#Display Depth Maps for Visual Verification
sgbm_norm =  cv.normalize(sgbm_left,None,alpha=0,beta=255,norm_type=cv.NORM_MINMAX,dtype=cv.CV_8U)
sbm_norm =  cv.normalize(sbm_left,None,alpha=0,beta=255,norm_type=cv.NORM_MINMAX,dtype=cv.CV_8U)

sgbm_filtered_norm =  cv.normalize(sgbm_filtered,None,alpha=0,beta=255,norm_type=cv.NORM_MINMAX,dtype=cv.CV_8U)
sbm_filtered_norm =  cv.normalize(sbm_filtered,None,alpha=0,beta=255,norm_type=cv.NORM_MINMAX,dtype=cv.CV_8U)

def coverage(disp):
    valid = np.count_nonzero(disp > 0)
    temp = np.count_nonzero(disp > disp.min())
    total = disp.shape[0]*disp.shape[1]
    return valid/temp

def shrinkage(disp,orig):
    disp_size = np.count_nonzero(disp > 1)
    orig_size = orig.shape[0]*orig.shape[1]
    return disp_size/orig_size

sgbm_coverage = coverage(sgbm_left)
sgbm_filtered_coverage = coverage(sgbm_filtered)
sbm_coverage = coverage(sbm_left)
sbm_filtered_coverage = coverage(sbm_filtered)

sgbm_shrinkage = shrinkage(sgbm_left, imgL)
sgbm_filtered_shrinkage = shrinkage(sgbm_filtered, imgL)
sbm_shrinkage = shrinkage(sbm_left, imgL)
sbm_filtered_shrinkage = shrinkage(sbm_filtered, imgL)

print(sgbm_left)

print(sgbm_coverage)
print(sgbm_filtered_coverage)
print(sbm_coverage)
print(sbm_filtered_coverage)
print(sgbm_shrinkage)
print(sgbm_filtered_shrinkage)
print(sbm_shrinkage)
print(sbm_filtered_shrinkage)

print(sgbm_time)
print(sgbm_filter_time)
print(sbm_time)
print(sbm_filter_time)

cv_file = cv.FileStorage()
cv_file.open('data.txt',cv.FileStorage_WRITE)
cv_file.write('sgbm_coverage',sgbm_coverage)
cv_file.write('sgbm_filtered_coverage',sgbm_filtered_coverage)
cv_file.write('sbm_coverage',sbm_coverage)
cv_file.write('sbm_filtered_coverage',sbm_filtered_coverage)

cv_file.write('sgbm_shrinkage',sgbm_shrinkage)
cv_file.write('sgbm_filtered_shrinkage',sgbm_filtered_shrinkage)
cv_file.write('sbm_shrinkage',sbm_shrinkage)
cv_file.write('sbm_filtered_shrinkage',sbm_filtered_shrinkage)

cv_file.write('sgbm_time',sgbm_time)
cv_file.write('sgbm_filtered_time',sgbm_filter_time)
cv_file.write('sbm_time',sbm_time)
cv_file.write('sbm_filtered_time',sbm_filter_time)

cv_file.release()

cv.imshow('sgbm',sgbm_norm)
cv.imshow('sbm',sbm_norm)
cv.imshow('sgbm_filtered',sgbm_filtered_norm)
cv.imshow('sbm_filtered',sbm_filtered_norm)

cv.waitKey(0)

cv.imwrite('depth/sgbm.png',sgbm_norm)
cv.imwrite('depth/sbm.png',sbm_norm)
cv.imwrite('depth/sgbm_filtered.png',sgbm_filtered_norm)
cv.imwrite('depth/sbm_filtered.png',sbm_filtered_norm)

#Calculate Points From Filtered Disparity Maps
sgbm_filtered_norm = np.float32(np.divide(sgbm_filtered_norm,16.0))
sbm_filtered_norm = np.float32(np.divide(sbm_filtered_norm,16.0))

sgbm_mask = sgbm_filtered_norm > sgbm_filtered_norm.min()
sbm_mask = sbm_filtered_norm > sbm_filtered_norm.min()

sgbm_3d = cv.reprojectImageTo3D(sgbm_filtered_norm,Q,handleMissingValues=True)
sbm_3d = cv.reprojectImageTo3D(sbm_filtered_norm,Q,handleMissingValues=True)

colors = cv.cvtColor(imgL,cv.COLOR_BGR2RGB)

sgbm_pts = sgbm_3d[sgbm_mask]
sgbm_colors = colors[sgbm_mask]

sbm_pts = sbm_3d[sbm_mask]
sbm_colors = colors[sbm_mask]

#Convert to Point Clouds, Save to File
def create_point_cloud_file(vertices, colors, filename):
    colors = colors.reshape(-1,3)
    vertices = np.hstack([vertices.reshape(-1,3),colors])

    ply_header = '''ply
        format ascii 1.0
        element vertex %(vert_num)d
        property float x
        property float y
        property float z
        property uchar red
        property uchar green
        property uchar blue
        end_header
        '''
    with open(filename, 'w') as f:
        f.write(ply_header %dict(vert_num=len(vertices)))
        np.savetxt(f,vertices,'%f %f %f %d %d %d')

create_point_cloud_file(sgbm_pts,sgbm_colors,'points/sgbmPts.ply')
create_point_cloud_file(sbm_pts,sbm_colors,'points/sbmPts.ply')