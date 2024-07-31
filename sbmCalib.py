import cv2 as cv

def nothing(x):
	pass

#Read Parameters
cv_file = cv.FileStorage()
cv_file.open('calib/stereoMap.xml', cv.FileStorage_READ)

stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()

Q = cv_file.getNode('q').mat()

cv_file.release()

#Set up OpenCV GUI
capL = cv.VideoCapture(1)
capR = cv.VideoCapture(0)

cv.namedWindow('Left')
cv.namedWindow('Right')
cv.namedWindow('Disparity')
cv.namedWindow('Filtered')


cv.createTrackbar('Blocksize','Disparity',1,100,nothing)
cv.createTrackbar('numDisparities','Disparity',1,16,nothing)
cv.createTrackbar('minDisparity','Disparity',0,100,nothing)
#cv.createTrackbar('uniquenessRatio','Disparity',0,10,nothing)
cv.createTrackbar('speckleWindowSize','Disparity',0,200,nothing)
cv.createTrackbar('speckleRange','Disparity',0,10,nothing)

cv.createTrackbar('Lambda','Filtered',8000,16000,nothing)
cv.createTrackbar('Sigma','Filtered',0,250,nothing)

stereo = cv.StereoBM_create(numDisparities=16)
stereo.setDisp12MaxDiff(2)

wls_filter = cv.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
wls_filter.setLambda(8000.0)
wls_filter.setSigmaColor(1.5)

run = True

while run: 
	retl, imgL = capL.read()
	retr, imgR = capR.read()

	if(retl and retr):
		cv.imshow('Left',imgL)
		cv.imshow('Right',imgR)

	if cv.waitKey(1) & 0xFF == ord('q'):
		run = False

	blocksize = cv.getTrackbarPos('Blocksize','Disparity')
	numDisp = cv.getTrackbarPos('numDisparities','Disparity')
	minDisp = cv.getTrackbarPos('minDisparity','Disparity')
#	ur = cv.getTrackbarPos('uniquenessRatio','Disparity')
	speckleSize = cv.getTrackbarPos('speckleWindowSize','Disparity')
	speckleRange = cv.getTrackbarPos('speckleRange','Disparity')

	l = cv.getTrackbarPos('Lambda','Filtered')
	s = cv.getTrackbarPos('Sigma','Filtered')

	s = s / 100.0

	blocksize = blocksize * 2 + 1

	if blocksize < 5:
		blocksize = 5

	if numDisp < 1:
		numDisp = 1

	numDisp = numDisp * 16

	stereo.setBlockSize(blocksize)
	stereo.setNumDisparities(numDisp)
	stereo.setMinDisparity(minDisp)
#	stereo.setUniquenessRatio(ur)
	stereo.setSpeckleWindowSize(speckleSize)
	stereo.setSpeckleRange(speckleRange)

	imgR = cv.remap(imgR, stereoMapR_x, stereoMapR_y, cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)
	imgL = cv.remap(imgL, stereoMapL_x, stereoMapL_y, cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)

	imgLgray = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
	imgRgray = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)

	dispmap = stereo.compute(imgLgray,imgRgray)
	norm_image = cv.normalize(dispmap, None, alpha = 0, beta = 255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)

	wls_filter.setLambda(l)
	wls_filter.setSigmaColor(s)

	disp_right = cv.ximgproc.createRightMatcher(stereo).compute(imgRgray,imgLgray)
	filtered_disp = wls_filter.filter(dispmap,imgLgray,disparity_map_right=disp_right)
	filtered_disp = cv.normalize(filtered_disp,None,alpha=0,beta=255,norm_type=cv.NORM_MINMAX,dtype=cv.CV_8U)

	cv.imshow('Disparity',norm_image)
	cv.imshow('Filtered',filtered_disp)

cv.imwrite('images/sbmCalib/sbmDisp.png',norm_image)
cv.imwrite('images/sbmCalib/sbmFiltered.png',filtered_disp)
cv.imwrite('images/sbmCalib/sbm_left_raw.png',imgL)

cv_file = cv.FileStorage("calib/sbmCalib.txt",cv.FileStorage_WRITE)

cv_file.write("blockSize",blocksize)
cv_file.write("numDisparities",numDisp)
cv_file.write("minDisparity",minDisp)
#cv_file.write("uniquenessRatio",ur)
cv_file.write("speckleWindowSize",speckleSize)
cv_file.write("speckleRange",speckleRange)

cv_file.write("lambda",l)
cv_file.write("sigmaColor",s)

cv_file.release()