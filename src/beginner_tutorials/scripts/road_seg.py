#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Rospy imports
import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image as ImageMsg
import tensorflow as tf

# Additional Imports
import os
import sys
from io import StringIO
from matplotlib import pyplot as plt
import numpy as np
#import pandas as pd
from PIL import Image


# Creates a class that will be used to run the code
class ImageSegment:
    # Class initialization, this name MUST NOT be changed.
    # This function will be executed ONCE, when the object is created
    def __init__(self):
	# /image_raw
        rospy.Subscriber("/camera/color/image_raw", ImageMsg, self.ImageCallback,queue_size=1)
        # Load Frozen Graph
        self.detection_graph = self.load_frozen_graph("road_seg/mobilenet_v2/frozen_inference_graph.pb")
        self.bridge = CvBridge()
        self.IMAGE = None

    # Resize the Image
    def ImageCallback(self,data):
        """
	Resize Image coming in from video feed
        Args:
            data (sensor_msgs,msg._Image.Image): Raw Image File from Camera Feed.
        """
        tmp = self.bridge.imgmsg_to_cv2(data,"bgr8")
        self.IMAGE = cv2.resize(tmp,None,fx=480.0/640,fy=480.0/480,interpolation=cv2.INTER_AREA) #fy=352.0/480

    def load_frozen_graph(self, frozen_graph_filename):
        """
        Args:
            frozen_graph_filename (str): Full path to the .pb file.
        """
        # We load the protobuf file from the disk and parse it to retrieve the
        # unserialized graph_def
        with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        # Then, we import the graph_def into a new Graph and returns it
        with tf.Graph().as_default() as graph:
            # The name var will prefix every op/nodes in your graph
            # Since we load everything in a new graph, this is not needed
            tf.import_graph_def(graph_def, name="prefix")
            print("Graph loaded from: " + frozen_graph_filename)
            return graph

    def segment(self, graph, image):
        """
        Does the segmentation on the given image. Image must be square.
        Args:
            graph (Tensorflow Graph)
            image (img): Image
        Returns:
            segmentation_mask (np.array): The segmentation mask of the image.
        """
        # We access the input and output nodes
        x = graph.get_tensor_by_name('prefix/ImageTensor:0')
        y = graph.get_tensor_by_name('prefix/SemanticPredictions:0')

        # We launch a Session
        with tf.Session(graph=graph) as sess:
            image_array = np.array(image)
            image_array = np.expand_dims(image_array, axis=0)

            # Note: we don't nee to initialize/restore anything
            # There is no Variables in this graph, only hardcoded constants
            pred = sess.run(y, feed_dict={x: image_array})
            pred = pred.squeeze()

        print("Prediction made")
        return pred

    def get_n_rgb_colors(self, n):
        """
        Get n evenly spaced RGB colors.
        Returns:
            rgb_colors (list): List of RGB colors.
        """
        max_value = 16581375 #255**3
        interval = int(max_value / n)
        colors = [hex(I)[2:].zfill(6) for I in range(0, max_value, interval)]

        rgb_colors = [(int(i[:2], 16), int(i[2:4], 16), int(i[4:], 16)) for i in colors]

        return rgb_colors

    def parse_pred(self, pred, n_classes):
        """
        Parses a prediction and returns the prediction as a PIL.Image.
        Args:
            pred (np.array)
        Returns:
            parsed_pred (PIL.Image): Parsed prediction that we can view as an image.
        """
        uni = np.unique(pred)
        print(uni)
        empty = np.empty((pred.shape[1], pred.shape[0], 3))
        colors = self.get_n_rgb_colors(n_classes)
        for i, u in enumerate(uni):
            idx = np.transpose((pred == u).nonzero())
            c = colors[u]
            empty[idx[:,0], idx[:,1]] = [c[0],c[1],c[2]]
        parsed_pred = np.array(empty, dtype=np.uint8)
        parsed_pred = Image.fromarray(parsed_pred)

        return parsed_pred

    def calc_road_pct(self, arr):
        """
        Given a row of pixels find the percentage that are the road (== 0)
        param arr (numpy arr): Array of classifications
        """
        return np.count_nonzero(arr == 0)/len(arr)

    def calc_focal_pt(self, pred, thresh):
        """
        Find the focus point for the car to go towards
        param pred (2D Numpy Array): Prediction Array
        param thresh (float): Minimum pct to consider a pixel line to contain road
        return x,y (tuple): Focal pixel point that marks center of the road at the furthest point
        """
        y = 0
        while y < len(pred)-1 and self.calc_road_pct(pred[y]) < thresh:
            y += 1
        pred_line = pred[y].tolist()
	try:         
	    x_min = pred_line.index(0)
	    x_max = len(pred_line) - 1 - pred_line[::-1].index(0)
	except:
	    x_min, x_max = (-1, -1)	
        x = int(np.mean([x_min, x_max]))
        return (x, y), (x_min, x_max)

    def vis_focal_pt(self, pred, focal_pt, x_bounds):
        """
        Visualize the focal point marking
        param pred (2D Numpy Array): Prediction Array
        param focal_pt (tuple): tuple of int containing the (x,y) coordinate of focal point
        param x_bounds (tuple): tuple of int containing the x bounds of where the road is on the focal_pt y line
        """
        x,y = focal_pt
	# If road was not found the x in the focal point will be -1
	if x > 0 and x < len(pred[0])-1: 
            for offset in [-1, 0, 1]:
                pred[y + offset][x_bounds[0]:x_bounds[1]] = [15]*(x_bounds[1] - x_bounds[0])
                for i in range(len(pred)): pred[i][x + offset] = 15
        pred_df = pd.DataFrame(pred)
        plt.imshow(pred_df, cmap='hot', interpolation='nearest')
        plt.show()

    def run(self):
	print("Starting Image Segmentation")
        rospy.sleep(2)
        rate = rospy.Rate(30)
	N_CLASSES = 19

	while not rospy.is_shutdown():
	    # Run Image Segmentation
	    image_np = self.IMAGE
	    prediction = self.segment(self.detection_graph, image_np)
	    segmented_img = self.parse_pred(prediction, n_classes=19)
	    print("Image segmented")
	    
	    # Calculate Focal Point
	    focal_pt, x_bounds = self.calc_focal_pt(prediction, thresh = 0.3)
	    print("The focal point is:", focal_pt)
	    # self.vis_focal_pt(prediction, focal_pt, x_bounds)
	    
	    # Show Original and Segmented Image
	    plt.imshow(image_np)
	    plt.title("Original Image")
	    plt.show()
	    plt.imshow(segmented_img)
	    plt.title("Processed Image")
	    plt.show()
	    plt.clf()
	    rate.sleep()
	
if __name__=="__main__":
    rospy.init_node('road_segment', anonymous=True)
    obj = ImageSegment()
    obj.run()


