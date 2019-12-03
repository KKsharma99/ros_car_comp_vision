#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Rospy imports
import rospy
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image as ImageMsg
import tensorflow as tf

# Additional Imports
import os
import sys
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
sys.path.append("..")
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


# Creates a class that will be used to run the code
class ImageProcessor:
    # Class initialization, this name MUST NOT be changed.
    # This function will be executed ONCE, when the object is created
    def __init__(self):
	# /image_raw
        rospy.Subscriber("/camera/color/image_raw", ImageMsg, self.ImageCallback,queue_size=1)
        # Load Frozen Graph
        self.detection_graph = self.load_frozen_graph()
        # List of the strings that is used to add correct label for each box.
        LABELS_PATH = os.path.join('object_detection','data', 'mscoco_label_map.pbtxt')
        self.category_index = label_map_util.create_category_index_from_labelmap(LABELS_PATH, use_display_name=True)
        self.bridge = CvBridge()
        self.IMAGE = None

    # Resize the Image
    def ImageCallback(self,data):
        tmp = self.bridge.imgmsg_to_cv2(data,"bgr8")
        #print('Image received')
        self.IMAGE = cv2.resize(tmp,None,fx=480.0/640,fy=352.0/480,interpolation=cv2.INTER_AREA)

    # Path to frozen detection graph. (SSD with Mobilenet) model that used for object detection.
    def load_frozen_graph(self):
        PATH_TO_FROZEN_GRAPH = 'object_detection/ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb'
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return detection_graph

    # Make a prediction given a single image
    def run_inference_for_single_image(self, image, graph):
        with graph.as_default():
            with tf.Session() as sess:
              # Get handles to input and output tensors
              ops = tf.get_default_graph().get_operations()
              all_tensor_names = {output.name for op in ops for output in op.outputs}
              tensor_dict = {}
              for key in [
                  'num_detections', 'detection_boxes', 'detection_scores',
                  'detection_classes', 'detection_masks']:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                  tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
                if 'detection_masks' in tensor_dict:
                    # The following processing is only for single image
                    detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                    detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                    # Reframe is required to translate mask from box coordinates to image coordinates to fit image.
                    real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                    detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                    detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                        detection_masks, detection_boxes, image.shape[1], image.shape[2])
                    detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                    # Follow the convention by adding back the batch dimension
                    tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)
              image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

              # Run inference
              output_dict = sess.run(tensor_dict,
                                     feed_dict={image_tensor: image})

              # all outputs are float32 numpy arrays, so convert types as appropriate
              output_dict['num_detections'] = int(output_dict['num_detections'][0])
              output_dict['detection_classes'] = output_dict[
                  'detection_classes'][0].astype(np.int64)
              output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
              output_dict['detection_scores'] = output_dict['detection_scores'][0]
              if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
        return output_dict


    def run(self):
	# Size, in inches, of the output images.

	IMAGE_SIZE = (12, 8)
        rospy.sleep(2)
        rate = rospy.Rate(30)
	print("Starting Object Detection")

        while not rospy.is_shutdown():
	    # the array based representation of the image will be used later in order to prepare the
	    # result image with boxes and labels on it.
	    image_np = self.IMAGE
	    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
	    image_np_expanded = np.expand_dims(image_np, axis=0)
	    # Actual detection.
	    output_dict = self.run_inference_for_single_image(image_np_expanded, self.detection_graph)

	    # These print statements can be used to help find what the class detection is and with what confidence
	    # Class and confidence is key information to be used downstream
	    print("Outputdict Detection Boxes")
	    print(output_dict['detection_boxes'])
	    print("Outputdict Detection Classes")
	    print(output_dict['detection_classes'])
	    print("Outputdict Detection Scores")
	    print(output_dict['detection_scores'])
	    print("Num Detections", output_dict['num_detections'])
		
	    # Visualization of the results of a detection.
	    vis_util.visualize_boxes_and_labels_on_image_array(
		image_np,
		output_dict['detection_boxes'],
		output_dict['detection_classes'],
		output_dict['detection_scores'],
		self.category_index,
		instance_masks=output_dict.get('detection_masks'),
		use_normalized_coordinates=True,
		line_thickness=8)
	    #plt.figure(figsize=IMAGE_SIZE)
	    plt.imshow(image_np)
	    plt.title("Processed Image")
	    plt.show()
	    plt.clf()
	    #print(image_np.shape) #Currently the Image Shape is 480 by 352 pixels
	    #processed_image = cv2.resize(image_np ,None,fx=480.0/640,fy=352.0/480,interpolation=cv2.INTER_AREA)
            rate.sleep()

	
if __name__=="__main__":
    rospy.init_node('image_processing', anonymous=True)
    obj = ImageProcessor()
    obj.run()

