import numpy as np
import cv2 as cv
import glob
from scipy.spatial.transform import Rotation as R

mtx = np.array(
  [[761.81488037, 0.0, 646.52478027], 
   [0.0, 761.15325928, 361.41662598], 
   [0.0, 0.0, 1.0]]
)
dist = np.array(
  [7.613864421844482, 17.2153263092041, 0.00012345814320724458, -0.00020271474204491824, -1.0174853801727295, 7.641190528869629, 19.72230339050293, 1.2936210632324219, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
)
board_size = (8,6)
square_size = 0.025


def draw(img, corners, imgpts):
  # breakpoint()
  corner = tuple(corners[0].ravel())
  x_axis = tuple(imgpts[0].ravel())
  y_axis = tuple(imgpts[1].ravel())
  z_axis = tuple(imgpts[2].ravel())
  img = cv.line(img, (int(corner[0]), int(corner[1])), (int(x_axis[0]), int(x_axis[1])), (0,0,255), 5)
  img = cv.line(img, (int(corner[0]), int(corner[1])), (int(y_axis[0]), int(y_axis[1])), (0,255,0), 5)
  img = cv.line(img, (int(corner[0]), int(corner[1])), (int(z_axis[0]), int(z_axis[1])), (255,0,0), 5)
  return img

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((board_size[0]*board_size[1],3), np.float32)
objp[:,:2] = np.mgrid[0:board_size[0],0:board_size[1]].T.reshape(-1,2)
# From checkerboard unit to meters
objp = objp*square_size
axis = np.float32([[3*square_size,0,0], [0,3*square_size,0], [0,0,-3*square_size]]).reshape(-1,3)

for i, fname in enumerate(glob.glob('*.jpg')):
  img = cv.imread(fname)
  gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
  ret, corners = cv.findChessboardCorners(gray, (board_size[0],board_size[1]),None)

  if ret == True:
    corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

  # Find the rotation and translation vectors.
  ret,rvecs, tvecs = cv.solvePnP(objp, corners2, mtx, dist)
  # breakpoint()
  orientation = R.from_rotvec(rvecs.reshape((3,)))

  print(f"For image-{i}:")
  print(f"Rotation_matrix: camera_optical2chess")
  print(f"{orientation.as_matrix()}")
  print(f"Translation_vectors: camera_optical2chess")
  print(f"{tvecs}\n")
  
  # project 3D points to image plane
  imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)
  # breakpoint()
  img = draw(img,corners2,imgpts)
  cv.imshow('img',img)
  cv.waitKey(0)

cv.destroyAllWindows()