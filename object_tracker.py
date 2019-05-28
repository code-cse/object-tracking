from statistics import mode
import imutils
import cv2
import numpy as np
from imutils.video import VideoStream
import time
import datetime
from preprocessor import preprocess_input
from tracker.centroidtracker import CentroidTracker
import pickle as pkl


def detect_faces(detection_model, gray_image_array, conf):
    frame = gray_image_array
    (h,w) =  frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
    (300, 300), (104.0, 177.0, 123.0))
    detection_model.setInput(blob)
    predictions = detection_model.forward()
    coord_list = []
    count = 0
    for i in range(0, predictions.shape[2]):
        confidence = predictions[0,0,i,2]
        if confidence > conf:
            # Find box coordinates rescaled to original image
            box_coord = predictions[0,0,i,3:7] * np.array([w,h,w,h])
            conf_text = '{:.2f}'.format(confidence)
            # Find output coordinates
            xmin, ymin, xmax, ymax = box_coord.astype('int')
            coord_list.append([xmin, ymin, (xmax-xmin), (ymax-ymin)])
            
        print('Coordinate list:', coord_klist)

    return coord_list

def draw_text(coordinates, image_array, text, color, x_offset=0, y_offset=0,
                                                font_scale=2, thickness=2):
    x, y = coordinates[:2]
    cv2.putText(image_array, text, (x + x_offset, y + y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness, cv2.LINE_AA)


def draw_bounding_box(face_coordinates, image_array, color, identity):
    x, y, w, h = face_coordinates
    if "_" not in identity:
        cv2.rectangle(image_array, (x, y), (x + w, y + h), color, 3)
        cv2.putText(image_array, str(identity), (x+5,y-5), font, 1, color, 2)
    else:
        cv2.rectangle(image_array, (x, y), (x + w, y + h), color, 3)


def apply_offsets(face_coordinates, offsets):
    x, y, width, height = face_coordinates
    x_off, y_off = offsets
    return (x - x_off, x + width + x_off, y - y_off, y + height + y_off)


def load_detection_model(prototxt, weights):
    detection_model = cv2.dnn.readNetFromCaffe(prototxt, weights)
    return detection_model

def verify_employee_id(key_list):
    key_list = key_list
    unique_val = np.unique(np.array(key_list))
    list_id = []
    list_count = []
    for j in range(len(unique_val)):
        count = key_list.count(unique_val[j])
        list_id.append(unique_val[j])
        list_count.append(count)
    print(list_id)
    print(list_count)
    index = np.array(list_count).argmax()
    iid = list_id[index]
    return iid

font = cv2.FONT_HERSHEY_SIMPLEX

frame_window = 10
face_offsets = (30, 40)
confidence = 0.6

def video_predict(file_name, face_detection):

    ct = CentroidTracker()

    face_detection_size = (40, 40)
    counter = 0
    frame_process_counter = 0

    cv2.namedWindow('Tracking', cv2.WINDOW_NORMAL)
    video_capture = cv2.VideoCapture(file_name)

    time.sleep(1.0)
    
    while (video_capture.isOpened()):
        ret, bgr_image = video_capture.read()

        if ret == False:
            break

        counter += 1
        if counter % 1 == 0:
            frame_process_counter += 1
            gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            rgb_image1 = rgb_image
            faces = detect_faces(face_detection, bgr_image,confidence)
         
            identity = "Person"
            print("frame_process_counter : ", frame_process_counter)
            for face_coordinates in faces:
                x1, x2, y1, y2 = apply_offsets(face_coordinates, face_offsets)
                rgb_face = rgb_image[y1:y2, x1:x2]
                try:
                    rgb_face = cv2.resize(rgb_face, (face_detection_size))
                except:
                    continue
                rgb_face = np.expand_dims(rgb_face, 0)
                rgb_face = preprocess_input(rgb_face, False)
            
                color = (0, 0, 255)
                draw_bounding_box(face_coordinates, rgb_image, (35,255,255), identity)

            objects = ct.update(faces)
            for (key, centroid), face_coordinates in zip(objects.items(), faces):
                x1, x2, y1, y2 = apply_offsets(face_coordinates, face_offsets)
                rgb_face = rgb_image1[y1:y2, x1:x2]
                try:
                    rgb_face = cv2.resize(rgb_face, (face_detection_size))
                except:
                    continue
                rgb_face = np.expand_dims(rgb_face, 0)
                rgb_face = preprocess_input(rgb_face, False)
                rgb_image = imutils.resize(rgb_image, width=1080)
                text = "ID - {}".format(key)
                cv2.putText(rgb_image, text, (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(rgb_image, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

            bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            frame = bgr_image
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow('Tracking', bgr_image)
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print('Total frames processed:', counter, frame_process_counter)
            break

    video_capture.release()
    cv2.destroyAllWindows()
    return "successful"
