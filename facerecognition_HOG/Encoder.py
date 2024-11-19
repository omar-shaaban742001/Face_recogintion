"""
This script will be used for encoding images that we have to compare it with the 
real time images 
"""
import cv2
import os 
import pickle
import face_recognition

# Get the images name
images_path = "F://programming//computer vision nanodegree//projects//facerecognition//images"
images_name = os.listdir(images_path) 

# Loop through the images to encode them
encodded_images = []
images_id =  []

print("Start encoding...")
for image_name in images_name:
    
    # Get the image path
    img = cv2.imread(os.path.join(images_path , image_name))
    
    # Change the image's color from BGR to RGB
    img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
    
    # Encode the image 
    encode = face_recognition.face_encodings(img)[0]

    # Save the encoded image into list 
    encodded_images.append(encode)

    # Get the name id 
    id = image_name.split(".")[0]

    # Save it to the student list
    images_id.append(id)

print('End encoding.....')

# Save each images with its id 
encodded_images_ids = [encodded_images ,images_id]

# Save the encoded images with ids into a file 
file = open("Encodded_images.p" , 'wb')
pickle.dump(encodded_images_ids , file)
file.close()

print("File saved")






