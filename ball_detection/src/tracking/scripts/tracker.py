#!/usr/bin/env python3

import cv2
import torch
import numpy as np

import rospy
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "bytetrack"))
from bytetrack.yolox.tracker.byte_tracker import BYTETracker

class Args:
    track_thresh = 0.01
    track_buffer = 30
    match_thresh = 0.8
    aspect_ratio_thresh = 9999
    min_box_area = 0
    mot20 = False

class ByteTrackNode:
    def __init__(self):
        rospy.init_node("bytetrack_node")

        self.tracker = BYTETracker(Args(), frame_rate=30)
        self.bridge = CvBridge()

        self.height = None
        self.width = None

        self.sub = rospy.Subscriber("/ball_detector/detections", Detection2DArray, self.detection_callback, queue_size=1)
        self.sub_img = rospy.Subscriber("/camera/image_raw", Image, self.image_callback, queue_size=1)
        self.pub = rospy.Publisher("/bytetrack/tracks", Detection2DArray, queue_size=10)

        rospy.loginfo("ByteTrackNode ready")

    def detection_callback(self, msg: Detection2DArray):
        start = rospy.get_time()

        detections = []

        for det in msg.detections:
            if not det.results:
                continue
            hypothesis = det.results[0]
            score = hypothesis.score
            class_id = hypothesis.id

            x = det.bbox.center.x - det.bbox.size_x / 2
            y = det.bbox.center.y - det.bbox.size_y / 2
            w = det.bbox.size_x
            h = det.bbox.size_y
            x2 = x + w
            y2 = y + h

            detections.append([x, y, x2, y2, 0.9, int(class_id)])

        rospy.logdebug(f"Received {len(detections)} detections")

        for d in detections:
            x1, y1, x2, y2 = d[:4]
            area = (x2 - x1) * (y2 - y1)
            if area <= 0:
                print(f"⚠️ Invalid box: {d}")

        if not detections:
            return

        if self.height is None or self.width is None:
            rospy.logwarn("Image dimensions not set yet.")
            return
        detections_tensor = torch.tensor(detections, dtype=torch.float32)
        tracked = self.tracker.update(detections_tensor, img_info=(self.height, self.width), img_size=(self.height, self.width))

        # Prepare output message
        tracked_msg = Detection2DArray()
        tracked_msg.header = msg.header

        for t in tracked:
            x, y, w_box, h_box = t.tlwh
            tid = t.track_id

            detection = Detection2D()
            detection.header = msg.header
            detection.bbox.center.x = x + w_box / 2
            detection.bbox.center.y = y + h_box / 2
            detection.bbox.size_x = w_box
            detection.bbox.size_y = h_box

            hyp = ObjectHypothesisWithPose()
            hyp.id = tid  # Use tracking ID
            hyp.score = t.score
            detection.results.append(hyp)

            tracked_msg.detections.append(detection)

        rospy.logdebug(f"Tracked {len(tracked)} detections")

        self.pub.publish(tracked_msg)

        end = rospy.get_time()

        elapsed_time = (end - start) * 1000
        rospy.loginfo(f"Processing time: {elapsed_time:.2f} ms")

    def image_callback(self, msg: Image):
        # Convert ROS Image message to OpenCV format
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        # Update height and width based on the image
        self.height = cv_image.shape[0]
        self.width = cv_image.shape[1]

if __name__ == "__main__":
    try:
        ByteTrackNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
