#!/usr/bin/env python

from radarGuidance import *
from wallFollower import *

import random #used for the random choice of a strategy
import sys
import numpy as np
import math
import pickle

#--------------------------------------
# Position of the goal:
goalx = 300
goaly = 450
# Initial position of the robot:
initx = 300
inity = 35
# strategy choice related stuff:
choice = -1
choice_tm1 = -1
tLastChoice = 0
rew = 0

i2name=['wallFollower','radarGuidance']

# added
currTime = 0

# Parameters of State building:
# threshold for wall consideration
th_neglectedWall = 35
# threshold to consider that we are too close to a wall
# and a punishment should be delivered
th_obstacleTooClose = 13
# angular limits used to define states
angleLMin = 0
angleLMax = 55

angleFMin=56
angleFMax=143

angleRMin=144
angleRMax=199

# Q-learning related stuff:
# definition of states at time t and t-1
S_t = ''
S_tm1 = ''
choice_made = False
Q = {}

#--------------------------------------
# the function that selects which controller (radarGuidance or wallFollower) to use
# sets the global variable "choice" to 0 (wallFollower) or 1 (radarGuidance)
# * arbitrationMethod: how to select? 'random','randPersist','qlearning'
def strategyGating(arbitrationMethod,verbose=True):
  global choice
  global choice_tm1
  global currTime
  global rew
  global choice_made
  global Q
  
  
  # The chosen gating strategy is to be coded here:
  #------------------------------------------------
  if arbitrationMethod=='random':
    choice = random.randrange(2)
  #------------------------------------------------
  elif arbitrationMethod=='randomPersist':
    if time.time() > currTime + 2:
    	choice = random.randrange(2)
    	currTime = time.time()
  #------------------------------------------------
  elif arbitrationMethod=='qlearning':
      
      alpha = 0.4
      beta = 4
      gamma = 0.95
      
      if S_t not in Q:
          Q[S_t] = np.array([0.5,0.5])
          
      if S_tm1 not in Q:
          Q[S_tm1] = np.array([0.5,0.5])
      
      
      # update q
      if (S_tm1 != S_t) or (choice_made == True) or (rew != 0):
          err = rew + gamma*(max(Q[S_t])) - Q[S_tm1][choice_tm1]
          Q[S_tm1][choice_tm1] = Q[S_tm1][choice_tm1] + alpha*err
          rew = 0
      

      # update choice
      if (S_tm1 != S_t) or (rew != 0) or (time.time() > currTime + 2) :
          choice_tm1 = choice
          p = np.exp(beta*Q[S_t]) / np.sum(np.exp(beta*Q[S_t]))
          choice = np.random.choice([0,1], p=p)
          currTime = time.time()
          choice_made = True
      else:
          choice_made = False
  
      
      
      
  #------------------------------------------------
  else:
    print(arbitrationMethod+' unknown.')
    exit()

  if verbose:
    print("strategyGating: Active Module: "+i2name[choice])

#--------------------------------------
def buildStateFromSensors(laserRanges,radar,dist2goal):
  S   = ''
  # determine if obstacle on the left:
  wall='0'
  if min(laserRanges[angleLMin:angleLMax]) < th_neglectedWall:
    wall ='1'
  S += wall
  # determine if obstacle in front:
  wall='0'
  if min(laserRanges[angleFMin:angleFMax]) < th_neglectedWall:
    wall ='1'
    #print("Mur Devant")
  S += wall
  # determine if obstacle on the right:
  wall='0'
  if min(laserRanges[angleRMin:angleRMax]) < th_neglectedWall:
    wall ='1'
  S += wall

  S += str(radar)

  if dist2goal < 125:
    S+='0'
  elif dist2goal < 250:
    S+='1'
  else:
    S+='2'
  #print('buildStateFromSensors: State: '+S)

  return S

#--------------------------------------
def main():

    global S_t
    global S_tm1
    global rew

    for seed in range(10):
	
        random.seed(seed)
        
        settings = Settings('worlds/entonnoir.xml')
        
        env_map = settings.map()
        robot = settings.robot()
        
        d = Display(env_map, robot)
        
        method = 'qlearning'
        
        # experiment related stuff
        startT = time.time()
        trial = 0
        nbTrials = 40
        trialDuration = np.zeros((nbTrials))
        
        i = 0
        
        while trial<nbTrials:
            # update the display
            #-------------------------------------
            d.update()
            # get position data from the simulation
            #-------------------------------------
            pos = robot.get_pos()
            # print("##########\nStep "+str(i)+" robot pos: x = "+str(int(pos.x()))+" y = "+str(int(pos.y()))+" theta ="+str(int(pos.theta()/math.pi*180.)))

            # has the robot found the reward ?
            #------------------------------------
            dist2goal = math.sqrt((pos.x()-goalx)**2+(pos.y()-goaly)**2)
            # if so, teleport it to initial position, store trial duration, set reward to 1:
            if (dist2goal<20): # 30
                print('***** REWARD REACHED *****')
                pos.set_x(initx)
                pos.set_y(inity)
                robot.set_pos(pos) # format ?
                # and store information about the duration of the finishing trial:
                currT = time.time()
                trialDuration[trial] = currT - startT
                startT = currT
                print("Trial "+str(trial)+" duration:"+str(trialDuration[trial]))
                trial +=1
                rew = 1
                
            # get the sensor inputs:
            #------------------------------------
            lasers = robot.get_laser_scanners()[0].get_lasers()
            laserRanges = []
            for l in lasers:
                laserRanges.append(l.get_dist())
                
            radar = robot.get_radars()[0].get_activated_slice()
            
            bumperL = robot.get_left_bumper()
            bumperR = robot.get_right_bumper()
            
            
            # 2) has the robot bumped into a wall ?
            #------------------------------------
            if bumperR or bumperL or min(laserRanges[angleFMin:angleFMax]) < th_obstacleTooClose:
                rew = -1
                print("***** BING! ***** "+i2name[choice])
                
            # 3) build the state, that will be used by learning, from the sensory data
            #------------------------------------
            S_tm1 = S_t
            S_t = buildStateFromSensors(laserRanges,radar, dist2goal)
            
            #------------------------------------
            strategyGating(method,verbose=False)
            if choice==0:
                v = wallFollower(laserRanges,verbose=False)
            else:
                v = radarGuidance(laserRanges,bumperL,bumperR,radar,verbose=False)
                
            i+=1
            robot.move(v[0], v[1], env_map)
            time.sleep(0.01)

            
        # When the experiment is over:
        np.savetxt('log/seed' + str(seed)+ '_TrialDurations_' + method + '.txt', trialDuration)
        with open("log/seed" + str(seed)+ "_" + "Q_" + method + ".pkl", "wb") as f:
            pickle.dump(Q, f)

#--------------------------------------

if __name__ == '__main__':
  random.seed()
  main()
