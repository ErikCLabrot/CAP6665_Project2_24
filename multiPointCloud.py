import cv2 as cv
import numpy as np
import glob
import open3d as o3d

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

#Load intrinsics, extrinsics, rectification map
##Rectification Map
cv_file = cv.FileStorage()

cv_file.open('calib/stereoMap.xml', cv.FileStorage_READ)

lx = cv_file.getNode('lx').mat()
ly = cv_file.getNode('ly').mat()
rx = cv_file.getNode('rx').mat()
ry = cv_file.getNode('ry').mat()

Q = cv_file.getNode('q').mat()

cv_file.release()

##Intrinsics, Extrinsics
cv_file.open('calib/stereoExtrinsics.txt',cv.FileStorage_READ)

K = cv_file.getNode('KL').mat()

cv_file.release()

##Stereo Matcher Params
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

cv_file.open('calib/sbmCalib.txt',cv.FileStorage_READ)

sbm_blockSize = int(cv_file.getNode('blockSize').real())
sbm_numDisparities = int(cv_file.getNode('numDisparities').real())
sbm_minDisparity = int(cv_file.getNode('minDisparity').real())
sbm_speckleWindow = int(cv_file.getNode('speckleWindowSize').real())
sbm_speckleRange = int(cv_file.getNode('speckleRange').real())
sbm_filter_lambda = cv_file.getNode('lambda').real()
sbm_filter_sigma = cv_file.getNode('sigmaColor').real()

cv_file.release()

#Load and Prepare Images
imagesLeft = sorted(glob.glob('images/stereoLeft/*.png'))
imagesRight = sorted(glob.glob('images/stereoRight/*.png'))

leftRaw = []
rightRaw = []

leftRect = []
rightRect = []

leftG = []
rightG = []

for IL, IR in zip(imagesLeft,imagesRight):
	left = cv.imread(IL)
	right = cv.imread(IR)

	leftRaw.append(left)
	rightRaw.append(right)

	lr = cv.remap(left, lx, ly, cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)
	rr = cv.remap(right, rx, ry, cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)

	lg = cv.cvtColor(lr,cv.COLOR_BGR2GRAY)
	rg = cv.cvtColor(rr,cv.COLOR_BGR2GRAY)

	leftRect.append(lr)
	rightRect.append(rr)

	leftG.append(lg)
	rightG.append(rg)

cv.imshow('leftg',leftG[0])
cv.waitKey(0)

#Compute Transforms
orb = cv.ORB_create()
matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck = True)

rotations = []
translations = []
for i in range(len(leftG) - 1):
	kp1, d1 = orb.detectAndCompute(leftG[i],None)
	kp2, d2 = orb.detectAndCompute(leftG[i+1],None)

	matches = matcher.match(d1,d2)
	matches = sorted(matches,key= lambda x: x.distance)

	src_pts = np.float32([kp1[m.queryIdx].pt for m in matches])
	dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches])

	E, mask = cv.findEssentialMat(src_pts, dst_pts, K, method=cv.RANSAC, prob=0.999, threshold=1.0)

	_,R,T,mask = cv.recoverPose(E,src_pts,dst_pts,K)

	t1 = float(T[0])
	t2 = float(T[1])
	t3 = float(T[2])
	t = np.array([t1,t2,t3])
	print(t)
	rotations.append(np.array(R))
	translations.append(t)

#Compute Depth Maps

##Create Matchers and Filters
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

sgbm_pcls = []
sbm_pcls = []

sgbm_colors = []
sbm_colors = []

i = 0

for lg,rg in zip(leftG, rightG):
	##SGBM Left and Right
	sgbm_left = sgbm.compute(lg,rg)
	sgbm_right = cv.ximgproc.createRightMatcher(sgbm).compute(rg,lg)
	sgbm_filtered = sgbm_filter.filter(sgbm_left,lg,disparity_map_right=sgbm_right)

	##SBM Right
	sbm_left = sbm.compute(lg,rg)
	sbm_right = cv.ximgproc.createRightMatcher(sbm).compute(rg,lg)
	sbm_filtered = sbm_filter.filter(sbm_left,lg,disparity_map_right=sbm_right)

	##Calculate 3D points
	sgbm_filtered_norm =  cv.normalize(sgbm_filtered,None,alpha=0,beta=1,norm_type=cv.NORM_MINMAX,dtype=cv.CV_32F)
	sbm_filtered_norm = cv.normalize(sbm_filtered,None,alpha=0,beta=1,norm_type=cv.NORM_MINMAX,dtype=cv.CV_32F)
	cv.imshow("test",sgbm_filtered_norm)
	cv.waitKey(0)
	sgbm_filtered_norm = np.float32(np.divide(sgbm_filtered_norm,16.0))
	sbm_filtered_norm = np.float32(np.divide(sbm_filtered,16.0))



	sgbm_mask = sgbm_filtered_norm > 0 #sgbm_filtered_norm.min()
	sbm_mask = sbm_filtered_norm > 0 #sbm_filtered_norm.min()

	sgbm_3d = cv.reprojectImageTo3D(sgbm_filtered_norm,Q,handleMissingValues=False)
	sbm_3d = cv.reprojectImageTo3D(sbm_filtered_norm,Q,handleMissingValues=False)

	temp = cv.cvtColor(leftRaw[i],cv.COLOR_BGR2RGB)

	sgbm_points = sgbm_3d[sgbm_mask]
	sbm_points = sbm_3d[sbm_mask]

	sgbm_colors.append(temp[sgbm_mask])
	sbm_colors.append(temp[sbm_mask])

	create_point_cloud_file(sgbm_points,temp[sgbm_mask], fname)

	sgbm_pcls.append(sgbm_points)
	sbm_pcls.append(sbm_points)
	i = i+1

sgbm_tf = []
sgbm_tf.append(sgbm_pcls[0])

sbm_tf = []
sbm_tf.append(sbm_pcls[0])

r_comb = np.eye(3)
t_comb = np.zeros(3)

rotations = np.array(rotations)
translations = np.array(translations)
print(translations.shape)
print(t_comb.shape)
print(translations.T)

i = 1
#Merge Depth Maps
for R, T in zip(rotations,translations):
	t_comb = r_comb @ T + t_comb
	r_comb = r_comb @ R
	sgbm_pt_tf = (r_comb @ sgbm_pcls[i].T).T + t_comb
	sbm_pt_tf = (r_comb @ sbm_pcls[i].T).T + t_comb
	sgbm_tf.append(sgbm_pt_tf)
	sbm_tf.append(sbm_pt_tf)
	i = i + 1
	print(r_comb)
	print(t_comb)

sgbm_pcd = o3d.geometry.PointCloud()
sbm_pcd = o3d.geometry.PointCloud()

for sgbm_cloud, sbm_cloud, sgbm_color, sbm_color in zip(sgbm_tf,sbm_tf,sgbm_colors,sbm_colors):
	pcd = o3d.geometry.PointCloud()
	pcd.points = o3d.utility.Vector3dVector(sgbm_cloud)
	pcd.colors = o3d.utility.Vector3dVector(sgbm_color/255.0)
	sgbm_pcd = sgbm_pcd + pcd

	pcd.points = o3d.utility.Vector3dVector(sbm_cloud)
	pcd.colors = o3d.utility.Vector3dVector(sbm_color/255.0)
	sbm_pcd = sbm_pcd + pcd

test_pcd = o3d.geometry.PointCloud()
test_pcd.points = o3d.utility.Vector3dVector(sgbm_pcls[0])
test_pcd.colors = o3d.utility.Vector3dVector(sgbm_colors[0])
o3d.visualization.draw_geometries([sgbm_pcd])