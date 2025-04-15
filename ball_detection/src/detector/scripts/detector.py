#!/usr/bin/env python3

import cv2
from ultralytics import YOLO
import logging
logging.getLogger('ultralytics').setLevel(logging.CRITICAL)

import rospy
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
from cv_bridge import CvBridge


class BallDetectorNode:
    def __init__(self):
        rospy.init_node('ball_detector_node')
        self.bridge = CvBridge()

        self.model = YOLO("/ball_detection/src/detector/weights/yolo11n.pt")
        self.ball_class_id = 32

        self.pub_image = rospy.Publisher('/ball_detector/image', Image, queue_size=10)
        self.pub_detections = rospy.Publisher('/ball_detector/detections', Detection2DArray, queue_size=10)
        rospy.Subscriber('/camera/image_raw', Image, self.image_callback)

        rospy.loginfo("BallDetectorNode is ready")
        rospy.spin()

    def image_callback(self, msg):

        start = rospy.get_time()
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            rospy.logerr(f"cv_bridge error: {e}")
            return

        results = self.model(frame, stream=True, conf=0.25)

        detection_array_msg = Detection2DArray()
        detection_array_msg.header = msg.header

        frame_with_boxes = frame.copy()

        for result in results:
            if result.boxes is not None and len(result.boxes.cls) > 0:
                for box, cls_id, score in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
                    if int(cls_id) == self.ball_class_id:
                        # Bounding box
                        x1, y1, x2, y2 = map(float, box)
                        width = x2 - x1
                        height = y2 - y1
                        cx = x1 + width / 2
                        cy = y1 + height / 2

                        detection = Detection2D()
                        detection.header = msg.header
                        detection.bbox.center.x = cx
                        detection.bbox.center.y = cy
                        detection.bbox.size_x = width
                        detection.bbox.size_y = height

                        # Hypothesis
                        hyp = ObjectHypothesisWithPose()
                        hyp.id = int(cls_id)
                        hyp.score = float(score)
                        detection.results.append(hyp)

                        detection_array_msg.detections.append(detection)

                frame_with_boxes = self.draw_bounding_boxes(frame, result)
                # break  # procesamos solo el primer resultado

        # Publicar imagen anotada
        msg_out = self.bridge.cv2_to_imgmsg(frame_with_boxes, encoding='bgr8')
        self.pub_image.publish(msg_out)

        # Publicar detecciones
        self.pub_detections.publish(detection_array_msg)

        end = rospy.get_time()
        elapsed_time_ms = (end - start) * 1000
        rospy.loginfo(f"Processing time: {elapsed_time_ms:.2f} ms")

    def draw_bounding_boxes(self, frame, result):
        frame_cp = frame.copy()
        boxes = result.boxes
        if boxes is not None:
            for box, cls_id in zip(boxes.xyxy, boxes.cls):
                if int(cls_id) == self.ball_class_id:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame_cp, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame_cp, "ball", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        return frame_cp


if __name__ == '__main__':
    try:
        BallDetectorNode()
    except rospy.ROSInterruptException:
        pass
