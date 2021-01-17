import os
import cv2 as cv2
import dlib 
from gaze_tracking import GazeTracking
gaze = GazeTracking()

fcnt = 0
input_file = "test_videos/linh.mp4"
#input_file = "resized_videos/lihn2.mp4"
output_file = "processed_videos/linh.avi"

cap = cv2.VideoCapture(input_file)

#info about input video 
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) 
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

#resize if needed with ffmpeg 
#dimensions = (400,640) #target 
dimensions = (width,height) 

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_file, fourcc, fps, dimensions, True)

# #face detector
detector = dlib.get_frontal_face_detector()
#load landmarks
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print(f"Can't receive frame {fcnt} (stream end?). Exiting ...")
        break
    
    #resize 
    frame = cv2.resize(frame, dimensions)
 
 
    # GazeTracking 
    gaze.refresh(frame)
    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()
    

    cv2.circle(frame, left_pupil, 10, (220, 0, 0), 1)
    cv2.circle(frame, right_pupil, 10, (220, 0, 0), 1)

    #find the face in a gray scale version of the frame. 
    faces = detector(frame)
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        #find landmarks within the face area
        landmarks = predictor(frame,face)
        point_37 = landmarks.part(37).x
        point_40 = landmarks.part(40).x
        delta_40_37 = point_40-point_37; #scale of movement. 
        

    # write frame
    out.write(frame)

    #update count and show progression
    fcnt += 1; 
    _progress = ((fcnt/frame_count *10)*10)
    if _progress % 10 ==0:
        print(_progress,'%') 
    cv2.imshow('frame', frame)

    #ecs to exit
    if cv2.waitKey(1) == 27:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
print("script completed")