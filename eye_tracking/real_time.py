import cv2
import csv
import dlib
from gaze_tracking import GazeTracking
gaze = GazeTracking()

#GLOBALS:

#CONFIG 
webcame = False

filename = "data/data_set_1.csv" #used of data
debug = True
frames_of_interest = (10,300)
if webcame:
    cap = cv2.VideoCapture(0) 
else: 
    input_file = "demo_videos/linh.mp4" #path to input video
    cap = cv2.VideoCapture(input_file) 

# estimated looking direction: RIGHT, LEFT, CENTER
looking_dir = 'CENTER'

#save data 
DATA_POINTS = []
fcnt = 0

#dlib models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


#functions: 

"""
    relative position of point B between 2 points A and C
    A < B and B < C
"""
def relative_position_between_points(A,B,C):
    if not B:
        print("error pupil not found in frame")
        return -1
    x_delta_AC = C.x - A.x
    x_delta_AB = B[0] - A.x
    if x_delta_AC > x_delta_AB: 
        return x_delta_AB/x_delta_AC
    else: 
        print("error false coordinates in frame")
        return -1

'''transforms 2 values between 0 and 1 into one off 3 discreed values LEFT,RIGHT,CENTER'''
def estimate_lookin_dir(relative_pos_left_iris,relative_pos_right_iris ): 
    _avg = (relative_pos_left_iris + relative_pos_right_iris)/2 
    if _avg > 0.60: 
        return "LEFT"
    if _avg < 0.40: 
        return "RIGHT"
    else: 
        return "CENTER"


"""logs data to an specified list"""
def log_eye_data(current_frame_count, looking_direction_left_eye, looking_direction_right_eye, looking_dir, data_log):
    data_log.append( {'frame':current_frame_count,'L':looking_direction_left_eye, 'R':looking_direction_right_eye, 'avg_looking_dir':looking_dir} )

''' write log to file'''
def save_log(filename,dict_data):
    csv_columns = ['frame','L','R','avg_looking_dir']
    try:
        with open(filename, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for data in dict_data:
                writer.writerow(data)
    except IOError:
        print("I/O error")


while True:
    ret, frame = cap.read()
    fcnt += 1
    if not ret or fcnt > frames_of_interest[1]:  
        print("end of video reached")
        break
        #blur = cv2.blur

    if fcnt > frames_of_interest[0]:
        #find the face in a gray scale version of the frame. 
        faces = detector(frame)
        for face in faces:
            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2 = face.bottom()
            cv2.rectangle(frame, (x1-10, y1-10), (x2+10, y2+10), (0, 255, 0), 2)

            # We send this frame to GazeTracking to analyze it
            gaze.refresh(frame)
            left_pupil = gaze.pupil_left_coords()
            right_pupil = gaze.pupil_right_coords()

            #find landmarks within the face area
            landmarks = predictor(frame, face)
            for n in range(36, 47):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                if debug: 
                    cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

            #find landmarks within the face area
            landmarks = predictor(frame,face)
            #left eye
            point_36 = landmarks.part(36)
            point_39 = landmarks.part(39)
            #right eye
            point_42 = landmarks.part(42)
            point_45 = landmarks.part(45)


            #collect data 
            looking_dir_left_eye = relative_position_between_points(point_36,left_pupil,point_39)
            looking_dir_right_eye = relative_position_between_points(point_42,right_pupil,point_45)
            looking_dir = estimate_lookin_dir(looking_dir_left_eye,looking_dir_right_eye)
            log_eye_data(fcnt, looking_dir_left_eye, looking_dir_right_eye, looking_dir, DATA_POINTS)
           


            if not debug: 
                cv2.circle(frame, left_pupil, 10, (220, 0, 0), 1)
                cv2.circle(frame, right_pupil, 10, (220, 0, 0), 1)

            if debug:
                cv2.line(frame,(point_36.x, point_36.y),(point_39.x,point_39.y),(0,200,0),1)
                cv2.line(frame,(point_42.x, point_39.y),(point_45.x,point_45.y),(0,200,0),1)
                cv2.circle(frame, left_pupil, 5, (250, 200, 0), 1)
                cv2.circle(frame, right_pupil, 5, (250, 200, 0), 1)
                cv2.putText(frame, looking_dir, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

                print(relative_position_between_points(point_36,left_pupil,point_39))

        cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break



# Release everything if job is finished
cap.release()
cv2.destroyAllWindows()
#print(data)
print("script ended")

#save data
if input("save data [y/n]:  ")[0].lower() == "y":
    save_log(filename,DATA_POINTS)
    print(f"saved under {filename}")