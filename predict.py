import os
import shutil
import object_tracker as vdf

prototxt = 'ckpt_/deploy.prototxt.txt'
weights = 'ckpt_/res10_300x300_ssd_iter_140000.caffemodel'

face_detection = vdf.load_detection_model(prototxt, weights)

print("Done model load")

file_name = "path to video"

a = vdf.video_predict(file_name, face_detection)

print(a)








