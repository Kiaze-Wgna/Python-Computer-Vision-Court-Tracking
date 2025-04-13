import os
import cv2
import time
import numpy as np
from Computer_Vision_Tools import court_detection_points

real_top_left=[0,0]
real_top_right=[610,0]
real_bottom_left=[0,1341]
real_bottom_right=[610,1341]
real_points=np.array([real_top_left,real_top_right,real_bottom_left,real_bottom_right],dtype=np.int32)

os.chdir(os.path.dirname(__file__))
font=cv2.FONT_HERSHEY_COMPLEX
cam = cv2.VideoCapture("../Tester_Videos/Court_video7.mp4")

#time.sleep(10)
print("stop waiting")

while True:
  status,frame=cam.read()
  frame_width=int(cam.get(3))
  frame_height=int(cam.get(4))
  modified_frame=frame.copy()

  court_points=court_detection_points(frame,frame_width,frame_height)
  print(type(court_points))
  if type(court_points)==type(None):
    cv2.imshow("ogframe",frame)
    cv2.imshow("frame",modified_frame)
    if cv2.waitKey(1)==ord("q"):
      break
    continue
  court_points=np.array(court_points,dtype=np.int32)
  court_top_left,court_top_right,court_bottom_left,court_bottom_right=court_points
  real2frame_matrix=cv2.getPerspectiveTransform(np.array(real_points,dtype=np.float32),np.array(court_points,dtype=np.float32))

  def real2frame(real_point_x,real_point_y,matrix=real2frame_matrix):
    real_point=np.array([[[real_point_x,real_point_y]]],dtype=np.float32)
    a=cv2.perspectiveTransform(real_point,matrix)
    a=(round(a[0][0][0]),round(a[0][0][1]))
    return a

  class lines_real():
    def __init__(self,frame,color,thickness):
      self.frame=frame
      self.color=color
      self.thickness=thickness
    def c(self,real_point_x1,real_point_y1,real_point_x2,real_point_y2):
      cv2.line(self.frame,real2frame(real_point_x1,real_point_y1),real2frame(real_point_x2,real_point_y2),self.color,self.thickness)

  def construct_field(frame,color,thickness):
    b=lines_real(frame,color,thickness)
    #outside_lines
    b.c(real_points[0][0],real_points[0][1],real_points[1][0],real_points[1][1])
    b.c(real_points[1][0],real_points[1][1],real_points[3][0],real_points[3][1])
    b.c(real_points[3][0],real_points[3][1],real_points[2][0],real_points[2][1])
    b.c(real_points[2][0],real_points[2][1],real_points[0][0],real_points[0][1])
    
    #horizontal lines
    b.c(0,76,610,76)
    b.c(0,76+396,610,76+396)
    b.c(0,76+396+198,610,76+396+198)
    b.c(0,76+396+2*198,610,76+396+2*198)
    b.c(0,76+2*(396+198),610,76+2*(396+198))

    #vertical lines
    b.c(46,0,46,1341)
    b.c(46+2*259,0,46+2*259,1341)

    #center lines
    b.c(46+259,0,46+259,76+396)
    b.c(46+259,76+396+2*198,46+259,2*(76+396+198))

  cv2.circle(modified_frame, court_top_left, radius=0, color=(255, 0, 255), thickness=10)
  cv2.circle(modified_frame, court_top_right, radius=0, color=(255, 0, 255), thickness=10)
  #cv2.circle(modified_frame, court_bottom_left, radius=0, color=(255, 0, 255), thickness=10)
  #cv2.circle(modified_frame, court_bottom_right, radius=0, color=(255, 0, 255), thickness=10)
  construct_field(modified_frame,(0,0,0),10)

  cv2.imshow("ogframe",frame)
  cv2.imshow("frame",modified_frame)
  if cv2.waitKey(1)==ord("q"):
    break
cv2.destroyAllWindows()
cam.release()
