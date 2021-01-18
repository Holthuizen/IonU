#ffmpeg -i test/video2.mp4 -vf scale=480:320 test/output_320.mp4
import cv2
import os
input_file = "demo_videos/linh.mp4"
output_file = "resized_videos/linh.mp4"
dimentions = (400,640)
#resize command
command = F"ffmpeg -i {input_file} -vf scale={dimentions[0]}:{dimentions[1]} {output_file}"
#inspect file
cap = cv2.VideoCapture(f'{input_file}')
# characteristics from the original video
w_frame, h_frame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#resize video
if w_frame != dimentions[0] or h_frame != dimentions[1]: 
    os.system(F'cmd /k "{command}"')





