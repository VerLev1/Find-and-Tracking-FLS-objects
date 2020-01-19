#! /usr/bin/env python

from __future__ import print_function
import rospy

# Brings in the OpenCV library
import cv2
import numpy as np

# Brings in the required library to convert ros imgmsg to cv img
from cv_bridge import CvBridge, CvBridgeError

from sensor_msgs.msg import Image

# Brings in the SimpleActionClient
import actionlib

# Brings in the messages used by the FLS action, including the
# goal message, result message and the feedback message
from learning_image_transport.msg import FLSAction, FLSGoal, FLSResult, FLSFeedback

# Brings in Vered's funtions
import objDetect
import tracking

def FLS_Client():
    
    global img_list
    global range_list
    img_list = []
    def feedback_cb (feedback):
	#print ('[Feedback] img_ctr = %d'%(feedback.img_ctr()))
	br = CvBridge()
	cv_img  = br.imgmsg_to_cv2(feedback.img_msg,"mono8")
	ctr = feedback.img_ctr
	filename = "Test%i.jpg" %ctr
	cv2.imwrite (filename,cv_img)
	img_list.append (cv_img)
	
    # Creates the SimpleActionClient, passing the type of the action
    # (FLSAction) to the constructor.
    client = actionlib.SimpleActionClient('FLS', FLSAction)
    
    # Waits until the action server has started up and started
    # listening for goals.
    client.wait_for_server()
	
    # Creates a goal to send to the action server.
    #goal = learning_image_transport.msg.FLSGoal(get_n_img=1)
    goal = FLSGoal()
    goal.get_n_img = 3
    # Sends the goal to the action server.
    client.send_goal(goal,feedback_cb=feedback_cb)
    print ("goal sent")
    
    client.wait_for_result()
    
    print('[Result] State: %d'%(client.get_state()))
    print('[Result] Status: %s'%(client.get_goal_status_text()))
    
   
    
    # Vered
    t = tracking.Tracking()
    index = 0
    for cv_img in img_list:
        index += 1
	      #cv2.imshow('tracking', cv_img)
        #cv2.waitKey(0)
        
        # reading the frame
	      img = cv_img #grayscale image
        
        rows, cols = img.shape
        color_img = np.zeros((rows,cols,3))
        color_img[:,:,0] = img
        color_img[:,:,1] = img
        color_img[:,:,2] = img
        
        #cv2.waitKey(0)

        # clean and find ROIs in the frame
        rois = objDetect.ROIfind(img)

        mask, centroids, bboxes, tiny_masks = rois.apply()

        # cleaning by tracking
        t.add_next_frame_features(color_img, mask, centroids, bboxes, tiny_masks)
        t.detection_to_tracks_assignment()
        t.update_tracks()
        t.create_new_tracks()
#        t.show_tracks()
#     plt.show()
    t.return_circles(index) # makes circles out of the objects - list
    t.show_circles()

    # Prints out the result of executing the action
    return client.get_result()  

if __name__ == '__main__':
    try:
        # Initializes a rospy node so that the SimpleActionClient can
        # publish and subscribe over ROS.
	
        rospy.init_node('FLS_Client_py')
        result = FLS_Client() 
	
        
    except rospy.ROSInterruptException:
        print("program interrupted before completion", file=sys.stderr)


    
	
