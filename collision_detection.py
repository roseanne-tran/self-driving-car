
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import pathlib


from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
#from IPython.display import display
import cv2


# Import the object detection module.

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

from ffpyplayer.player import MediaPlayer
#import pygame


# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile




def load_model(model_name):
  base_url = 'http://download.tensorflow.org/models/object_detection/'
  model_file = model_name + '.tar.gz'
  model_dir = tf.keras.utils.get_file(
    fname=model_name, 
    origin=base_url + model_file,
    untar=True)

  model_dir = pathlib.Path(model_dir)/"saved_model"

  model = tf.saved_model.load(str(model_dir))
  model = model.signatures['serving_default']

  return model


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'models/research/object_detection/data/mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)




# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.

'''
PATH_TO_TEST_IMAGES_DIR = pathlib.Path('models/research/object_detection/test_images')
TEST_IMAGE_PATHS = sorted(list(PATH_TO_TEST_IMAGES_DIR.glob("*.jpg")))
TEST_IMAGE_PATHS
'''


# # Detection

# Load an object detection model:

model_name = 'ssd_mobilenet_v1_coco_2017_11_17'
detection_model = load_model(model_name)


def run_inference_for_single_image(model, image):
  image = np.asarray(image)
  # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
  input_tensor = tf.convert_to_tensor(image)
  # The model expects a batch of images, so add an axis with `tf.newaxis`.
  input_tensor = input_tensor[tf.newaxis,...]

  # Run inference
  output_dict = model(input_tensor)

  # All outputs are batches tensors.
  # Convert to numpy arrays, and take index [0] to remove the batch dimension.
  # We're only interested in the first num_detections.
  num_detections = int(output_dict.pop('num_detections'))
  output_dict = {key:value[0, :num_detections].numpy() 
                 for key,value in output_dict.items()}
  output_dict['num_detections'] = num_detections

  # detection_classes should be ints.
  output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
   
  # Handle models with masks:
  if 'detection_masks' in output_dict:
    # Reframe the the bbox mask to the image size.
    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
              output_dict['detection_masks'], output_dict['detection_boxes'],
               image.shape[0], image.shape[1])      
    detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                       tf.uint8)
    output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
    
  return output_dict


def show_inference(model, image_path):
  # the array based representation of the image will be used later in order to prepare the
  
  # result image with boxes and labels on it.
#  image_np = np.array(Image.open(image_path))
  
  # Actual detection.
  output_dict = run_inference_for_single_image(model, image_path)
  
  # Visualization of the results of a detection.
  vis_util.visualize_boxes_and_labels_on_image_array(
      image_path,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks_reframed', None),
      use_normalized_coordinates=True,
      line_thickness=5)
   

  if output_dict['detection_classes'][2] == 3 or output_dict['detection_classes'][2] == 6 or output_dict['detection_classes'][2] == 8:
    if output_dict['detection_scores'][2] > 0.5:
      mid_x = (output_dict['detection_boxes'][2][3] + output_dict['detection_boxes'][2][1]) /2
      mid_y = (output_dict['detection_boxes'][2][2] + output_dict['detection_boxes'][2][0])/2
      apx_distance = round(((1 - (output_dict['detection_boxes'][2][3] - output_dict['detection_boxes'][2][1]))**4),1)
#      cv2.putText(image_path, '{}'.format(apx_distance), (int(mid_x*800),int(mid_y*450)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    
      if apx_distance <=0.5:
#        if mid_x > 0.1 and mid_x < 0.5:
        cv2.putText(image_path, 'WARNING!!!', (int(mid_x*1100),int(mid_y*450)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
   
  
  
  cv2.imshow('result', image_path) #cv2.resize(image_path, (800,600)))
  if cv2.waitKey(1) & 0xFF == ord('q'):
    cv2.destroyAllWindows()
    return


#warning_sound = pygame.mixer.Sound(' ')
#voice.play(warning_sound)
video = 'car-race.MOV'

cap = cv2.VideoCapture(video)
#audio = MediaPlayer(video)

while (cap.isOpened()):
  ret, frame = cap.read()
  show_inference(detection_model, frame)
#  audio_frame, val = audio.get_frame()


cap.release()
cv2.destroyAllWindows()
