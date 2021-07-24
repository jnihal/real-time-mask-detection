import tensorflow as tf
import numpy as np
import cv2

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


def predict(frame, mask_model, face_model):
    height, width = frame.shape[:2]

    # Detect faces using the face_model
    blob = cv2.dnn.blobFromImage(frame, 1, (224, 224), (104, 117, 123))
    face_model.setInput(blob)
    detections = face_model.forward()

    coordinates = list()
    faces = list()
    predictions = list()

    # Loop through the faces
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Get boundary of the face
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            (x1, y1, x2, y2) = box.astype('int')
            coordinates.append((x1, y1, x2, y2))
            
            x1 -= 100
            x2 += 100
            y1 -= 100
            y2 += 100
            x1 = max(0, x1)
            y1 = max(0, y1)

            face = frame[y1:y2, x1:x2]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            faces.append(face)

    # Detect mask if faces are detected
    if len(faces) > 0:
        faces = np.array(faces, dtype='float32')
        predictions = mask_model(faces)
    
    return coordinates, predictions


def realTimeDetect():

    # Load the models for mask and face detection
    mask_model = tf.keras.models.load_model('MaskDetect')
    face_model = cv2.dnn.readNet('FaceDetect/deploy.prototxt', 'FaceDetect/res10_300x300_ssd_iter_140000.caffemodel')

    # Capture video using webcam
    video = cv2.VideoCapture(1)

    while True:

        # Get the frame
        ret, frame = video.read()

        if ret == True:

            # Mirror the video
            frame = cv2.flip(frame, 1)

            # Get the boundary and prediction
            coordinates, predictions =  predict(frame, mask_model, face_model)
            
            for coordinate, prediction in zip(coordinates, predictions):
                (x1, y1, x2, y2) = coordinate

                # Wearing a mask
                if prediction[0] > prediction[1]:
                    color = (118, 221, 118)
                    label = f"Mask {prediction[0] * 100:.2f}"

                # Not wearing a mask
                else:
                    color = (98, 105, 255)
                    label = f"No Mask {prediction[1] * 100:.2f}"

                # Draw appropriate labelling
                cv2.putText(frame, label, (x1, y2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4) 
            
            # Display the frame
            cv2.imshow('Mask Detector', frame)
            cv2.setWindowProperty('Mask Detector', cv2.WND_PROP_TOPMOST, 1)

            # Key to close the webcam
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        else:
            break
    
    # Release the webcam
    video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    realTimeDetect()
