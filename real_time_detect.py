from numpy.core.fromnumeric import shape
import tensorflow as tf
import numpy as np
import cv2

from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


def predict(frame, mask_model, face_model):
    height, width = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1, (224, 224), (104, 117, 123))
    face_model.setInput(blob)
    detections = face_model.forward()

    coordinates = list()
    faces = list()
    predictions = list()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

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

    if len(faces) > 0:
        faces = np.array(faces, dtype='float32')
        predictions = mask_model(faces)
    
    return coordinates, predictions


def realTimeDetect():
    mask_model = tf.keras.models.load_model('MaskDetect')
    face_model = cv2.dnn.readNet('FaceDetect/deploy.prototxt', 'FaceDetect/res10_300x300_ssd_iter_140000.caffemodel')

    video = cv2.VideoCapture(1)

    while True:
        ret, frame = video.read()

        if ret == True:
            frame = cv2.flip(frame, 1)

            coordinates, predictions =  predict(frame, mask_model, face_model)
            
            for coordinate, prediction in zip(coordinates, predictions):
                (x1, y1, x2, y2) = coordinate

                if prediction[0] > prediction[1]:
                    color = (118, 221, 118)
                    label = f"Mask {prediction[0] * 100:.2f}"
                else:
                    color = (98, 105, 255)
                    label = f"No Mask {prediction[1] * 100:.2f}"

                cv2.putText(frame, label, (x1, y2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4) 
            
            cv2.imshow('Mask Detector', frame)
            cv2.setWindowProperty('Mask Detector', cv2.WND_PROP_TOPMOST, 1)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        else:
            break
        

    video.release()
    cv2.destroyAllWindows()

realTimeDetect()