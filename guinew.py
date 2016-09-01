"""
This script supercedes the previous gui.py and is an improved version
wrt visual odometry computation.  The important changes are in "data2" 
function (which generates data to plot in subplot #2).  Of interest to
you are these variables:
  
* poses: poses according to (correted) wheel odometry, to be
  used for VERTEXes in g2o  

* dx, dy, dt: incremental changes in dx, dy, and heading in the local
  robot frame, to be used as local constraints EDGEs in g2o.

* pose: an intermediate list that stores the raw wheel odometry (x, y, and 
  theta) before correction

"""

import os
import sys
import glob
import re
import time
import math
import numpy as np
import scipy as sp
import cv2
import subprocess
from threading import *
from matplotlib import pyplot as plt
import pylab as plb
import matplotlib.animation as animation
import sayem as sym
import copy
gp = True
def points2(data2):
  global loopfroms,looptos
  xs.append(data2[0][0])
  ys.append(data2[0][1])
  label_text.set_text(label_template%(data2[1]))
  line2.set_data(ys, xs)
  for i in range (0,len(loopfroms)):
    tmpxs = [xs[loopfroms[i]],xs[looptos[i]]]
    tmpys =[ys[loopfroms[i]],ys[looptos[i]]]
    ax2.plot(tmpys,tmpxs,'r')
    del loopfroms[i]
    del looptos[i]
  return line2, label_text

def updateImg(*args):
  global im1

  img1 = np.fliplr(im1.reshape(-1,3)).reshape(im1.shape)
  im.set_array(img1)
  return im,

def updateLoop(*args):
    
  global im4
  '''
  global sImg
  global dog2o
  lImg = copy.deepcopy(sImg)
  images.append(lImg)
  #print "well weell"
  
  if len(images)>12 : # discard the last 12 images
    lastimg = len(images)-1
    hogmatched =[]
    
    for j in range(0,lastimg-12):
      if sym.getSimilarity(lImg,images[j])>.9:
        hogmatched.append(j)
        
    
    if len(hogmatched)>0 :
        loopIndex = max(hogmatched)
        dog2o = True 
        print "loop closer with " +str(lastimg)+" "+ str(loopIndex)
    
    
    if len(hogmatched)>0:
      ks = [sym.geometryMatching(lImg,images[k])  for k in hogmatched ]
      if ks !=None and max(ks)>.9:
        loopedIndex = hogmatched[ks.index(max(ks))]
        print "loop closer with "+str(lastimg)+"  "+str(loopedIndex)
        dog2o = True
    '''
  return im4
def updateCorrectedPos(*args):
  global dog2o
  dog2o=True
  if dog2o:  
      p = subprocess.call(["g2o","-o", "xx","-typeslib","libvertigo-g2o", "g2oInput.g2o"],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
      
      f =open("xx","r")
      #print "lll"
      #p.wait()
      #print "lsdkfj"
      xs3=[]
      ys3=[]

      g2corPos=[]
      #print out
      for line in f:
        #print line
          ll = line.split()
          #print x
          if len(ll)>0 and ll[0]=="VERTEX_SE2":         #VERTEX_SE2 1200 155.961 -366.893 -2.51722
              xs3.append(ll[2])
              ys3.append(ll[3])
       
          elif len(ll)>0 and ll[0]=="EDGE_SE2":
              break
      line3.set_data(xs3,ys3)
              #g2corPos.append([ll[2],ll[3],ll[4]])
         # send to point2 to plot
      f.close()
      dog2o = False
      return line3,

def add_state():
    global hist_filter
    for i in reversed(range(1,len(hist_filter))):
            hist_filter[i]=motion_model[0]*hist_filter[i]+motion_model[1]*hist_filter[i-1]
    hist_filter[0]= hist_filter[0]*motion_model[0]
    #l = len(hist_filter)
    #s = sum(hist_filter)
    #if l != 0:
    # hist_filter.append(s/l)
    #else:
    # hist_filter.append(.1)
    '''
    if len(hist_filter)==0:
        hist_filter =[1]
        return  
    else:
        hist_filter.append(0)
        for i in reversed(range(1,len(hist_filter))):
            hist_filter[i]=motion_model[0]*hist_filter[i]+motion_model[1]*hist_filter[i-1]
        hist_filter[0]= hist_filter[0]*motion_model[0]
    ''' 
def norm_state():
    global hist_filter
    s = float(sum(hist_filter))
    hist_filter = [float(i)/float(s) for i in hist_filter]
def data2():
  global im1
  global sImg
  global gx,gy
  global hist_filter
  global loopfroms, looptos
  im1 = cv2.imread(imFiles[0]) 
  truePositive = 0
  trueNegative =0
  falsePositive =0
  falseNegative =0
  q = [0,0,0,1]
  pose = []          # previous pose according to wheel odometry
  poses = [[]]       # corrected poses 
  g2vertex=[]
  g2edges=[]
  g2loops=[]
  offset=[]
  hist_filter =[1]*len(imFiles)
  norm_state()
  print hist_filter
  loopclosed = False
  for i in range(len(imFiles)):
    #print len(images)
    #print len(hist_filter)
    print imFiles[i]
    im1 = cv2.imread(imFiles[i])
    odo = open(odoFiles[i]) 
    lines = []
    for line in odo:
      lines.append(line)
    if loopclosed:
      add_state()
    #print hist_filter
    #print hist_filter.index(max(hist_filter))
    #print(sum(hist_filter))
    #print hist_filter
    # read (x,y) of the next pose according to odometry
    x = float(re.findall("-?\d+.\d+", lines[1])[0])
    y = float(re.findall("-?\d+.\d+", lines[2])[0])

    # read orientation expressed in quaternion and compute Euler angle th_z 

    q[0] = float(re.findall("-?\d+.\d+", lines[5])[0])
    q[1] = float(re.findall("-?\d+.\d+", lines[6])[0])
    q[2] = float(re.findall("-?\d+.\d+", lines[7])[0])
    q[3] = float(re.findall("-?\d+.\d+", lines[8])[0])

    # for how this equation works, refer to 
    # en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
    # we deal with phi in the special case of q0 = q1 = 0

    th = math.atan2(2*q[2]*q[3], 1 - 2*q[2]*q[2])
  
    if i == 0:
      # initialize intermeidate and historic pose variables
      pose = [x, y, th]
    
      poses.append(pose)
      tmpvertex = "VERTEX_SE2 "+str(i)+" "+str(x)+" "+str(y)+" "+str(th)
      g2vertex.append(tmpvertex)
    elif i ==numPoses+1:		# initialize first pose of loop 2
        print imFiles[i]
        offset.append(pose[0] - poses[-1][0])
        offset.append(pose[1] - poses[-1][1])
        offset.append(pose[2] - poses[-1][2])
        poses.append([a-b for a, b in zip(pose, offset)])
        x = poses[-1][0] # [x, y, t] of 1st pose of 2nd loop = 
        y = poses[-1][1] # those of last pose of 1st loop
        t = poses[-1][2]
        dx = 0.0	 # skip computing incremental changes for this
        dy = 0.0	 # initialization step
        dt = 0.0
        tmpvertex = "VERTEX_SE2 "+str(i)+" "+str(x)+" "+str(y)+" "+str(t)
        g2vertex.append(tmpvertex)
    else:
      # compute incremental changes in x, y and theta in world frame   
      dxw = x - pose[0]
      dyw = y - pose[1]
      dt = th - pose[2]

      # update "pose" for next iteration
      pose = [x, y, th] 
     
      # make sure the angular change doesn't jump
      if dt > np.pi:
        dt -= 2*np.pi
      elif dt < -np.pi:
        dt += 2*np.pi

      # !!! this is robot-specific: our robot wheel odometry is not correct
      # 0.47 works but any number close should be fine
      dt = dt*0.5 
    
      # calculate linear change in x of local frame
      r = math.sqrt(dxw*dxw + dyw*dyw)
      dx = r*math.cos(dt)
      dy = r*math.sin(dt)       
      print 
      # finally, produce the correct pose
      t = poses[-1][2] + dt 
      x = poses[-1][0] + r*math.cos(t) 
      y = poses[-1][1] + r*math.sin(t)
      # if it is the first image just add the vertex:
      
      tmpvertex = "VERTEX_SE2 "+str(i)+" "+str(x)+" "+str(y)+" "+str(t)
      g2vertex.append(tmpvertex)
      if i!=0 and i!=numPoses+2:
          tmpedge="EDGE_SE2 "+str(i-1)+" "+str(i)+" "+str(dx)+" "+str(dy)+" "+str(dt)+" 100 0 0 100 0 1000"
          g2edges.append(tmpedge)
      
      #write the grond truth
      ''''
      if i>648:
          tmpLoop="EDGE_SE2 " +str(i-649)+ " " +str(i)+" 0 0 0 100 0 0 100 0 1000"
          g2loops.append(tmpLoop)
      '''
      poses.append([x, y, t])
      #Appearance SLAM:
    sImg = sym.Image(im1)
    images.append(sImg)
  
    
    if len(images)>12 : # discard the last 12 images
         #print ">>>>>>>>>>>>>>>"
         #print hist_filter
         #print "<<<<<<<<<<"
         lastimg = len(images)-1
         hogmatched =[]
         hogmatch_confidence=[]
         for j in range(0,lastimg-12):
           sim = sym.getSimilarity(sImg,images[j])          
           norm_state
           if sim>.80:
             #hist_filter[j]*=.95 
             hogmatched.append(j)
             hogmatch_confidence.append(float(sim))
         #print hist_filter
         #print hist_filter
         #norm_state()
         
         loopcloser =False
         if len(hogmatched)>0 :
           loopcloser=True
           maxmatch = max(hogmatch_confidence)
           loopIndex = hogmatched[hogmatch_confidence.index(maxmatch)]
           
           print "maxmatch "+ str(maxmatch)
           if maxmatch<=.9: 
             xx = float(sym.geometryMatching(images[i],images[loopIndex]))
             print "epipolar xx " + str(xx)
             if xx<80:
                 print "epipolar dismiss "+ str(i) +" " + str(loopIndex)
                 loopcloser=False
         if loopcloser:
           ax4.imshow(images[loopIndex].img)
          # loopfroms.append(i)
          # looptos.append(loopIndex) 
           if not loopclosed:
               loopclosed = True
           hist_filter[loopIndex]= hist_filter[loopIndex]*maxmatch
           for ot in range (0,len(hist_filter)):
               if ot !=loopIndex:
                  hist_filter[ot] =hist_filter[ot]*(1-maxmatch)
                
               norm_state()
           dog2o = True 
           print "loop closer with " +str(lastimg)+" "+ str(loopIndex)
           #hist_filter[loopIndex]*=maxmatch
           #norm_state()
           trueLoop = hist_filter.index(max(hist_filter[0:-13]))
           if trueLoop != loopIndex:
               print "ORANGE ALERT  " + str(loopIndex)+" " + str(trueLoop)
           
           #print "^^^^^^^^^^^^^^^^^^^^^^^^^^"
           #print max(hogmatch_confidence)
           #print maxmatch
           #print loopIndex
           #print hist_filter[loopIndex]
           #print hist_filter
           #print "^^^^^^^^^^^^^^^^^^^^^^^^^^^"
           #break   
           tmpLoop="EDGE_SE2 " +str(trueLoop)+ " " +str(i)+" 0 0 0 100 0 0 100 0 1000"
           g2loops.append(tmpLoop)
           if abs(i - (numPoses+trueLoop))<4:
               truePositive+=1
           else:
               falsePositive+=1
         else:
              if i > numPoses:
                  falseNegative+=1
              else:
                  trueNegative+=1
    g2oinput = open("g2oInput.g2o",'w')
    g2oinput.write("\n".join(g2vertex))
    g2oinput.write("\n")
    g2oinput.write("\n".join(g2edges))
    g2oinput.write("\n")
    g2oinput.write("\n".join(g2loops))
    print " true Positive : "+str(truePositive)
    print " false Positive : "+str(falsePositive)
    print " true Negative : "+str(trueNegative)
    print " false Negative : "+str(falseNegative)
    g2oinput.close()
    odo.close()
    yield [y, x], i
  # end for-loop 

  print 'Done simulation ..' 
  time.sleep(50000) 
  sys.exit(1)

imFiles=[]
odoFiles=[]
#os.remove("xx")
dog2o =False
images =[]
hist_filter =[]
motion_model =[.1,.9]
loopfroms=[]
looptos=[]
for dir in ["loop3/","loop4/"]:
    os.chdir(dir)
    tmpFiles=[]
    tmpOdo=[]
    for file in glob.glob("*.jpg"):
      tmpFiles.append(file)
    tmpFiles = sorted(tmpFiles, key=lambda x: int(x.split('.')[0]))
    tmpFiles = [dir+x for x in tmpFiles]
    
    imFiles.extend(tmpFiles)

    for file in glob.glob("*.txt"):
        tmpOdo.append(file)
    tmpOdo = sorted(tmpOdo, key=lambda x: int(x.split('.')[0]))
    tmpOdo=[dir+x for x in tmpOdo]
    odoFiles.extend(tmpOdo)
    os.chdir("..")
numPoses =  len(imFiles)/2   
print numPoses
fig = plt.figure(figsize=(12, 9))
g2oCorrectedPos=[[2,2],[1,2]]
ax1 = fig.add_subplot(221)
plb.setp(plb.gca(), 'xticks', [])
plb.setp(plb.gca(), 'yticks', [])

im1 = cv2.imread(imFiles[0]) 
im1 = np.fliplr(im1.reshape(-1,3)).reshape(im1.shape)
im = plt.imshow(im1)

ani1 = animation.FuncAnimation(fig, updateImg, interval=40, blit=False)

xs = []
ys = []
ax2 = fig.add_subplot(222)

line2, = ax2.plot(xs, ys)
ax2.set_ylim(-400, 200)
plb.setp(plb.gca(), 'yticks', [-300,-200, -100, 0, 100], 
                    'yticklabels', [-200, -100, 0, 100])
ax2.set_xlim(-300, 300)
plb.setp(plb.gca(), 'xticks', [-200,-100, -0, 100,200], 
                    'xticklabels', [-100, 0, 100])

label_template = '%3d steps'    # prints running number of steps
label_text = ax2.text(0.05, 0.9, '', transform=ax2.transAxes)

ani2 = animation.FuncAnimation(fig, points2, data2, blit=False,\
     interval=100, repeat=True)




ax3 = fig.add_subplot(223)
plb.setp(plb.gca(), 'xticks', [])
plb.setp(plb.gca(), 'yticks', [])


xs3 =[]
ys3=[]

line3, = ax3.plot(xs3,ys3)
ax3.set_ylim(-400, 200)
ax3.set_xlim(-300, 300)
ani3 = animation.FuncAnimation(fig, updateCorrectedPos, interval=40, blit=False)

ax4 = fig.add_subplot(224)
plb.setp(plb.gca(), 'xticks', [])
plb.setp(plb.gca(), 'yticks', [])

im4 = cv2.imread(imFiles[1]) 
im4 = np.fliplr(im4.reshape(-1,3)).reshape(im4.shape)

#imm = plt.imshow(im4)
#thr = Thread(target=animation.FuncAnimation, args=(fig, updateLoop), kwargs={"interval":100, "blit":False})
#thr.start() # will run "foo"
ani4 = animation.FuncAnimation(fig, updateLoop, interval=40, blit=False)



plt.show()
