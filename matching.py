import cv2
import numpy as np
import scipy as sp
from scipy import linalg, mat, dot
import random
import math

#start of Image class
class Image:

    # --------
    # init: 
    #

    def __init__(self, img):
        self.img = img # image
        self.dhog = None;
        self.siftDes = None;
        self.siftKP = None;
    
    #return dhog feature vector
    def getDhog(self):
        if self.dhog is None:
            img = self.img
            scale = 2
            h, w = img.shape[:2]
            h = h/scale 
            w = w/scale
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)

            hog = cv2.HOGDescriptor()
            self.dhog = hog.compute(img, winStride=(64,128), padding=(0, 0))

        return self.dhog


    # return sift key points and descriptor
    def getSiftDescriptor(self):
        if self.siftDes is None:
            img1 = self.img
            img1 = cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY)
            img1 = cv2.resize(img1, (256, 250))  
            detector = cv2.FeatureDetector_create("SIFT")  # SURF, FAST, SIFT
            descriptor = cv2.DescriptorExtractor_create("SIFT") # SURF, SIFT

            # find the keypoints with the chosen detector 
            self.siftKP = detector.detect(img1)
            

            # find the descriptors with the chosen descriptor
            [k1, self.siftDes] = descriptor.compute(img1,self.siftKP)
        return [self.siftKP, self.siftDes]

#End of Image class 

#return similarity between two dhog features
def getSimilarity(im1, im2):
    dhog1 = im1.getDhog()
    dhog2 = im2.getDhog()
    similarity = np.dot(dhog1.T,dhog2)/(linalg.norm(dhog1)*linalg.norm(dhog2))
    return similarity



def geometryMatching(im1, im2):
    MIN_MATCH_COUNT = 10
    [kp1, des1] =im1.getSiftDescriptor()
    [kp2, des2] =im2.getSiftDescriptor()
    # for each feature in img1, find its two nearest neighbors in img2
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)


    
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
            
    if len(good)>MIN_MATCH_COUNT:
      src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
      dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

      M, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.RANSAC, 5.0)

      matchesMask = mask.ravel().tolist()

    else:
      print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
      matchesMask = None
      return 0
    z = np.count_nonzero(matchesMask)
    total = len(matchesMask)
    per = (float(z)/total)*100.0
    return per
#visual odometry
def PfromE(E):
  
  U, D, VT = np.linalg.svd(E)
  
  Z = [[0, 1, 0], [-1, 0, 0], [0, 0, 0]]
  W = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
  
  if np.linalg.det(U) < 0:
    U = -U 
  if np.linalg.det(VT) < 0:
    VT = -VT
      
  P1 = U*W*VT
  P1 = np.hstack((P1, U[:, 2]))
  P2 = U*W*VT 
  P2 = np.hstack((P2, -U[:, 2]))
      
  WT = map(list, zip(*W))
  P3 = U*WT*VT
  P3 = np.hstack((P3, U[:, 2]))
  P4 = U*WT*VT
  P4 = np.hstack((P4, -U[:, 2]))

  return(P1, P2, P3, P4)

def visualOdometry(im1, im2):
  
  [kp1, des1] = im1.getSiftDescriptor();
  [kp2, des2] = im2.getSiftDescriptor();
  # BFMatcher with default params
  bf = cv2.BFMatcher()
  matches = bf.knnMatch(des1,des2, k=2)

  good = []

  # Apply ratio test
  for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)
  src_pts=[]
  dst_pts =[]
  MIN_MATCH_COUNT = 10
  if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    F, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.RANSAC, 5.0)

    matchesMask = mask.ravel().tolist()

  else:
    print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
    matchesMask = None
    
  src_pts = src_pts[:, 0, :]    # adjust the dimensions of two feature lists
  dst_pts = dst_pts[:, 0, :]    

  v = np.ones((len(src_pts), 1))

  src_pts = np.c_[src_pts, v]  # and express them in homogeneous coordinates
  dst_pts = np.c_[dst_pts, v] 

  E = K.T*F*K 
  P = PfromE(E)

  q = src_pts[0,:]
  qp = dst_pts[0,:]

  E = np.array(E)
 
  foundP = False
  foundI = None
  for i in range(4):
    # Pick the correct P according to Nister's five-ponit algorithm
    Pa = P[:][:][i]

    a = np.dot(E.T, qp)
    b = np.cross(q, np.dot(np.diag([1,1,0]), a))
    c = np.cross(qp, np.dot(np.dot(np.diag([1,1,0]), E), q))
    d = np.cross(a, b)
    C = np.array(np.dot(Pa.T, c).T)
    C = C[:,0]
    Q = np.append(d.T*C[3], -(d[0]*C[0]+d[1]*C[1]+d[2]*C[2]))

    # Test the solution
 
    c1 = Q[2]*Q[3]
    t = np.array(np.dot(Pa, Q))
    t = t[0,:]
    c2 = t[2]*Q[3]
    if c1 > 0 and c2 > 0:
      foundP = True
      foundI = i

    if foundP:
      Pa = P[:][:][foundI]
      th = math.atan2(Pa[0, 2], Pa[2, 2])
    else:       
      th = None

  return th
K = np.matrix([[543.3046, 0, 309.1810], 
                 [0, 542.2960, 232.7158], 
                 [0.0, 0.0, 1.0]])

if __name__ == '__main__':
 


    #----------------------------------
    #test -----------------------------
    img1 = cv2.imread('./loop1/1.jpg')
    img2 = cv2.imread('./loop1/2.jpg')
    im1 = Image(img1)
    im2 = Image(img2)
    print getSimilarity(im1, im2)
    print geometryMatching(im1, im2)
    print visualOdometry(im1, im2)
    #print im.getDhog()
