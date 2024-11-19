import cv2
import os 
import face_recognition
import cvzone 
import pickle
import numpy as np


# Set up my own camera 
cap = cv2.VideoCapture(0)

# Set up the size of the window
cap.set(3, 640)
cap.set(4, 480)


# Get the encoded images 
file = open("Encodded_images.p" , 'rb')
encodded_images_id = pickle.load(file)
saved_encodded_images , id = encodded_images_id
file.close()
print("Loaded encodded images....")



# Loop through the video 
while True:
    # Read frame by frame 
    _ , img= cap.read()
    
    # Resize the images 
    resized_img = cv2.resize(img , (0 , 0) , None , 0.25 , .25)
    
    # Change the color from GBR to RGB
    # resized_img 
    resized_img = cv2.cvtColor(resized_img , cv2.COLOR_BGR2RGB)

    # Get the faces from each frame 
    faces_per_frame = face_recognition.face_locations(resized_img)
    
    # Pass the faces to be decoded
    encoded_faces = face_recognition.face_encodings(resized_img , faces_per_frame) 
 
    # Loop throgh encodded images to compare them
    for encoded_image , face_loc in zip(encoded_faces , faces_per_frame):

        matches = face_recognition.compare_faces(saved_encodded_images , encoded_image)
        distance = face_recognition.face_distance(saved_encodded_images, encoded_image)
    
        # Get the small distance 
        match_index = np.argmin(distance)

                # Get the faces locations
        y1 , x2 , y2 , x1 = face_loc
        y1, x2, y2, x1 = y1*4, x2*4 , y2*4 , x1*4
        
        # Draw the rectangle around the face
        cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,0), 3)
       
        if matches[match_index]:
            name = id[match_index]
        else:
            name = 'unkown'

        # Draw the text on the image
        cv2.putText(img, str(name), (x1, y1-10),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)





    # Show the img 
    cv2.imshow("Attendence sysem",img)
    cv2.waitKey(1)