#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

def main():
    rospy.init_node('webcam_driver_node')
    pub = rospy.Publisher('/camera/image_raw', Image, queue_size=10)
    bridge = CvBridge()

    # cap = cv2.VideoCapture(0)
    # Connect to an http webcam
    cap = cv2.VideoCapture('http://192.168.1.101:8080/video')
    if not cap.isOpened():
        rospy.logerr("Could not open webcam")
        return
    rospy.loginfo("Webcam driver started")
    rate = rospy.Rate(10)  # 10 Hz

    while not rospy.is_shutdown():
        ret, frame = cap.read()
        if not ret:
            rospy.logwarn("No frame from webcam")
            continue

        msg = bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "camera_frame"
        msg.width = frame.shape[1]
        msg.height = frame.shape[0]
        pub.publish(msg)
        rate.sleep()

    cap.release()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
