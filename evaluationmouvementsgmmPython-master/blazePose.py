import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
file_to_write=open("data/SkeletonSequence1.txt","w+") #Where to write results

mode="video" #camera / video 
videoFile='data/Videos_exercices/rotationTronc/sub01_trial01/VideoColor_Correct1.avi' #For video input, if mode=="video"

#If there are nbDir directories (numeroted from 1 to nbDir) with nbSubDir subdirectories each (numeroted from 1 to nbSubDire) and with 
# nbEx correct exercices (numeroted from 0 to nbEx-1) in each of those, if mode=="video"
processAllVideos=False
nbDir=3
nbSubDir=3
nbEx=5
exercice="cacheTete"



# For webcam input:
if mode == "camera" :
  cap = cv2.VideoCapture(0)
  with mp_pose.Pose(
     min_detection_confidence=0.5,
      min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
      success, image = cap.read()
      if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue

      # Flip the image horizontally for a later selfie-view display, and convert
      # the BGR image to RGB.
      image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
      # To improve performance, optionally mark the image as not writeable to
      # pass by reference.
      image.flags.writeable = False
      results = pose.process(image)

      # Draw the pose annotation on the image.
      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
      mp_drawing.draw_landmarks(
          image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
      if not results.pose_landmarks:
          continue
      keypoints = [0 for i in range(132)]
      j=0 #To avoid appends
      for data_point in results.pose_landmarks.landmark: #Keypoints contain all of the landmarks
        keypoints[j]=data_point.x 
        j+=1
        keypoints[j]=data_point.y
        j+=1
        keypoints[j]=data_point.z
        j+=1
        keypoints[j]=data_point.visibility
        j+=1
    
      keraalLandmarks=[0 for i in range(75)] #No visibility parameter in it and only the 25 joints from Kinect
      #right -> left on blazepose and left -> right on blazepose
      for i in range(3):
        keraalLandmarks[i]=keypoints[24*4+i]+keypoints[23*4+i] #SpineBase
        keraalLandmarks[3+i]=keypoints[24*4+i]+keypoints[23*4+i]+keypoints[11*4+i]+keypoints[12*4+i] #SpineMid
        keraalLandmarks[6+i]=keypoints[10*4+i]+keypoints[9*4+i]+keypoints[11*4+i]+keypoints[12*4+i] #Neck
        keraalLandmarks[9+i]=keypoints[i] #Head
        keraalLandmarks[12+i]=keypoints[12*4+i] #Shoulder Left
        keraalLandmarks[15+i]=keypoints[14*4+i] #Elbow Left
        keraalLandmarks[18+i]=keypoints[16*4+i] #Wrist Left
        keraalLandmarks[21+i]=keypoints[16*4+i]+keypoints[18*4+i]+keypoints[20*4+i]+keypoints[22*4+i] #Hand Left
        keraalLandmarks[24+i]=keypoints[11*4+i] #Shoulder Right
        keraalLandmarks[27+i]=keypoints[13*4+i] #Elbow Right
        keraalLandmarks[30+i]=keypoints[15*4+i] #Wrist Right
        keraalLandmarks[33+i]=keypoints[15*4+i]+keypoints[17*4+i]+keypoints[19*4+i]+keypoints[21*4+i] #Hand right
        keraalLandmarks[36+i]=keypoints[24*4+i] #Hip Left
        keraalLandmarks[39+i]=keypoints[26*4+i] #Knee Left
        keraalLandmarks[42+i]=keypoints[28*4+i] #Ankle Left
        keraalLandmarks[45+i]=keypoints[28*4+i]+keypoints[30*4+i]+keypoints[32*4+i] #Foot Left
        keraalLandmarks[48+i]=keypoints[23*4+i] #Hip Right
        keraalLandmarks[51+i]=keypoints[25*4+i] #Knee Right
        keraalLandmarks[54+i]=keypoints[27*4+i] #Ankle Right
        keraalLandmarks[57+i]=keypoints[27*4+i]+keypoints[29*4+i]+keypoints[31*4+i] #Foot Right
        keraalLandmarks[60+i]=keypoints[11*4+i]+keypoints[12*4+i] #Spine Shoulder
        keraalLandmarks[63+i]=keypoints[20*4+i] #Hand Tip Left
        keraalLandmarks[66+i]=keypoints[22*4+i] #Thumb Left
        keraalLandmarks[69+i]=keypoints[19*4+i] #Hand Tip Right
        keraalLandmarks[72+i]=keypoints[21*4+i] #Thumb Right
      keraalLandmarks[0:3]= [r/2 for r in keraalLandmarks[0:3]]
      keraalLandmarks[3:6]=[r/4 for r in keraalLandmarks[3:6]]
      keraalLandmarks[6:9]=[r/4 for r in keraalLandmarks[6:9]]
      keraalLandmarks[21:24]=[r/4 for r in keraalLandmarks[21:24]]
      keraalLandmarks[33:36]=[r/4 for r in keraalLandmarks[33:36]]
      keraalLandmarks[45:48]=[r/3 for r in keraalLandmarks[45:48]]
      keraalLandmarks[57:60]=[r/3 for r in keraalLandmarks[57:60]]
      keraalLandmarks[60:63]=[r/2 for r in keraalLandmarks[60:63]]


      print(*keraalLandmarks,sep=" ",file=file_to_write)

      cv2.imshow('MediaPipe Pose', image)
      if cv2.waitKey(5) & 0xFF == 27:
        break
  cap.release()

elif mode == "video" :
  # For video input, to process all the files :
  if processAllVideos :
    for dir in range(1,nbDir+1) :
      for subdir in range(1,nbSubDir+1) :
        for k in range(0,nbEx) :
          file_to_write=open("data/SkeletonSequence"+str((k+1)+5*(subdir-1)+15*(dir-1)) + ".txt","w+")
          videoFile='data/Videos_exercices/'+str(exercice)+'/sub0'+str(dir)+'_trial0'+str(subdir)+'/VideoColor_Correct'+str(k)+'.avi'
          cap = cv2.VideoCapture(videoFile)
          with mp_pose.Pose(
              min_detection_confidence=0.5,
              min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
              success, image = cap.read()
              if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                break

              # Flip the image horizontally for a later selfie-view display, and convert
              # the BGR image to RGB.
              image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
              # To improve performance, optionally mark the image as not writeable to
              # pass by reference.
              image.flags.writeable = False
              results = pose.process(image)

              # Draw the pose annotation on the image.
              image.flags.writeable = True
              image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
              mp_drawing.draw_landmarks(
                  image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
              if not results.pose_landmarks:
                  continue
              keypoints = [0 for i in range(132)]
              j=0 #Avoid appends
              for data_point in results.pose_landmarks.landmark: #Keypoints contain all of the landmarks
                keypoints[j]=data_point.x 
                j+=1
                keypoints[j]=data_point.y
                j+=1
                keypoints[j]=data_point.z
                j+=1
                keypoints[j]=data_point.visibility
                j+=1
              
              keraalLandmarks=[0 for i in range(75)] #No visibility parameter in it and only the 25 joints from Kinect
              #right -> left on blazepose and left -> right on blazepose
              for i in range(3):
                keraalLandmarks[i]=keypoints[24*4+i]+keypoints[23*4+i] #SpineBase
                keraalLandmarks[3+i]=keypoints[24*4+i]+keypoints[23*4+i]+keypoints[11*4+i]+keypoints[12*4+i] #SpineMid
                keraalLandmarks[6+i]=keypoints[10*4+i]+keypoints[9*4+i]+keypoints[11*4+i]+keypoints[12*4+i] #Neck
                keraalLandmarks[9+i]=keypoints[i] #Head
                keraalLandmarks[12+i]=keypoints[12*4+i] #Shoulder Left
                keraalLandmarks[15+i]=keypoints[14*4+i] #Elbow Left
                keraalLandmarks[18+i]=keypoints[16*4+i] #Wrist Left
                keraalLandmarks[21+i]=keypoints[16*4+i]+keypoints[20*4+i] #Hand Left
                keraalLandmarks[24+i]=keypoints[11*4+i] #Shoulder Right
                keraalLandmarks[27+i]=keypoints[13*4+i] #Elbow Right
                keraalLandmarks[30+i]=keypoints[15*4+i] #Wrist Right
                keraalLandmarks[33+i]=keypoints[15*4+i]+keypoints[19*4+i] #Hand right
                keraalLandmarks[36+i]=keypoints[24*4+i] #Hip Left
                keraalLandmarks[39+i]=keypoints[26*4+i] #Knee Left
                keraalLandmarks[42+i]=keypoints[28*4+i] #Ankle Left
                keraalLandmarks[45+i]=keypoints[28*4+i]+keypoints[30*4+i]+keypoints[32*4+i] #Foot Left
                keraalLandmarks[48+i]=keypoints[23*4+i] #Hip Right
                keraalLandmarks[51+i]=keypoints[25*4+i] #Knee Right
                keraalLandmarks[54+i]=keypoints[27*4+i] #Ankle Right
                keraalLandmarks[57+i]=keypoints[27*4+i]+keypoints[29*4+i]+keypoints[31*4+i] #Foot Right
                keraalLandmarks[60+i]=keypoints[11*4+i]+keypoints[12*4+i] #Spine Shoulder
                keraalLandmarks[63+i]=keypoints[20*4+i] #Hand Tip Left
                keraalLandmarks[66+i]=keypoints[22*4+i] #Thumb Left
                keraalLandmarks[69+i]=keypoints[19*4+i] #Hand Tip Right
                keraalLandmarks[72+i]=keypoints[21*4+i] #Thumb Right
              keraalLandmarks[0:3]= [r/2 for r in keraalLandmarks[0:3]]
              keraalLandmarks[3:6]=[r/4 for r in keraalLandmarks[3:6]]
              keraalLandmarks[6:9]=[r/4 for r in keraalLandmarks[6:9]]
              keraalLandmarks[21:24]=[r/2 for r in keraalLandmarks[21:24]]
              keraalLandmarks[33:36]=[r/2 for r in keraalLandmarks[33:36]]
              keraalLandmarks[45:48]=[r/3 for r in keraalLandmarks[45:48]]
              keraalLandmarks[57:60]=[r/3 for r in keraalLandmarks[57:60]]
              keraalLandmarks[60:63]=[r/2 for r in keraalLandmarks[60:63]]


              print(*keraalLandmarks,sep=" ",file=file_to_write)

              #cv2.imshow('MediaPipe Pose', image)
              if cv2.waitKey(5) & 0xFF == 27:
                break
          cap.release()
  else :
    cap = cv2.VideoCapture(videoFile)
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
      while cap.isOpened():
        success, image = cap.read()
        if not success:
          print("Ignoring empty camera frame.")
          # If loading a video, use 'break' instead of 'continue'.
          break

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = pose.process(image)

        # Draw the pose annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        if not results.pose_landmarks:
            continue
        keypoints = [0 for i in range(132)]
        j=0 #Avoid appends
        for data_point in results.pose_landmarks.landmark: #Keypoints contain all of the landmarks
          keypoints[j]=data_point.x 
          j+=1
          keypoints[j]=data_point.y
          j+=1
          keypoints[j]=data_point.z
          j+=1
          keypoints[j]=data_point.visibility
          j+=1
        
        keraalLandmarks=[0 for i in range(75)] #No visibility parameter in it and only the 25 joints from Kinect
        #right -> left on blazepose and left -> right on blazepose
        for i in range(3):
          keraalLandmarks[i]=keypoints[24*4+i]+keypoints[23*4+i] #SpineBase
          keraalLandmarks[3+i]=keypoints[24*4+i]+keypoints[23*4+i]+keypoints[11*4+i]+keypoints[12*4+i] #SpineMid
          keraalLandmarks[6+i]=keypoints[10*4+i]+keypoints[9*4+i]+keypoints[11*4+i]+keypoints[12*4+i] #Neck
          keraalLandmarks[9+i]=keypoints[i] #Head
          keraalLandmarks[12+i]=keypoints[12*4+i] #Shoulder Left
          keraalLandmarks[15+i]=keypoints[14*4+i] #Elbow Left
          keraalLandmarks[18+i]=keypoints[16*4+i] #Wrist Left
          keraalLandmarks[21+i]=keypoints[16*4+i]+keypoints[20*4+i] #Hand Left
          keraalLandmarks[24+i]=keypoints[11*4+i] #Shoulder Right
          keraalLandmarks[27+i]=keypoints[13*4+i] #Elbow Right
          keraalLandmarks[30+i]=keypoints[15*4+i] #Wrist Right
          keraalLandmarks[33+i]=keypoints[15*4+i]+keypoints[19*4+i] #Hand right
          keraalLandmarks[36+i]=keypoints[24*4+i] #Hip Left
          keraalLandmarks[39+i]=keypoints[26*4+i] #Knee Left
          keraalLandmarks[42+i]=keypoints[28*4+i] #Ankle Left
          keraalLandmarks[45+i]=keypoints[28*4+i]+keypoints[30*4+i]+keypoints[32*4+i] #Foot Left
          keraalLandmarks[48+i]=keypoints[23*4+i] #Hip Right
          keraalLandmarks[51+i]=keypoints[25*4+i] #Knee Right
          keraalLandmarks[54+i]=keypoints[27*4+i] #Ankle Right
          keraalLandmarks[57+i]=keypoints[27*4+i]+keypoints[29*4+i]+keypoints[31*4+i] #Foot Right
          keraalLandmarks[60+i]=keypoints[11*4+i]+keypoints[12*4+i] #Spine Shoulder
          keraalLandmarks[63+i]=keypoints[20*4+i] #Hand Tip Left
          keraalLandmarks[66+i]=keypoints[22*4+i] #Thumb Left
          keraalLandmarks[69+i]=keypoints[19*4+i] #Hand Tip Right
          keraalLandmarks[72+i]=keypoints[21*4+i] #Thumb Right
        keraalLandmarks[0:3]= [r/2 for r in keraalLandmarks[0:3]]
        keraalLandmarks[3:6]=[r/4 for r in keraalLandmarks[3:6]]
        keraalLandmarks[6:9]=[r/4 for r in keraalLandmarks[6:9]]
        keraalLandmarks[21:24]=[r/2 for r in keraalLandmarks[21:24]]
        keraalLandmarks[33:36]=[r/2 for r in keraalLandmarks[33:36]]
        keraalLandmarks[45:48]=[r/3 for r in keraalLandmarks[45:48]]
        keraalLandmarks[57:60]=[r/3 for r in keraalLandmarks[57:60]]
        keraalLandmarks[60:63]=[r/2 for r in keraalLandmarks[60:63]]


        print(*keraalLandmarks,sep=" ",file=file_to_write)

        #cv2.imshow('MediaPipe Pose', image)
        if cv2.waitKey(5) & 0xFF == 27:
          break
    cap.release()


file_to_write.close()