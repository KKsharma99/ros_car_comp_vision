The computer vision project is named as "beginner tutorials". To find the main files go to "src/beginner_tutorials/scripts"

Inside you will see the following:
Folder "object_detection": Contains all of the files needed to run object detection in ROS. It also contains the model: "ssd_mobilenet_v1_coco_2017_11_17"

Folder "road_seg": Contains two models that can be loaded in the road_seg.py file. This includes the xception_71 model and the mobilenet_v2 model.

Image rc_pov.jpg: This is an image that was taken at the track about 2 feet from the ground and can be used for testing purposes.

Python file image_processor.py: This file containes the ImageProcessor class that subscribes to the the camera feed and detects objects in the images. From the input image, and output_dict is created and is used to create a bounding box visualization. Further work needs to be done in being able to extract the list of classes that were found and their confidence scores to be used by the RC car. The Google Object Detection Github will be useful for this as this is the repo that the code is based on https://github.com/tensorflow/models/tree/master/research/object_detection/.

Python file road_seg.py: This file contains the ImageSegment class that loads one of the models from the "road_seg" folder. It conducts the segmentation on the input camera feed and displays the segmentation results. There are also a few functions in this file that calculate the focal point the RC car can use for driving based on the road pixels identified in the segmented image. Further work needs to be done to use the focal point to calculate the steering angle and pass this data to a controller node.

If you have questions you can email me at kunalsharma@gatech.edu
