
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import cv2 as cv
import numpy as np
import pyautogui
import time
import mss
import imutils
import tensorflow as tf


class DetectorAPI:
    def __init__(self, path_to_ckpt):
        self.path_to_ckpt = path_to_ckpt

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(self.path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.default_graph = self.detection_graph.as_default()
        self.sess = tf.compat.v1.Session(graph=self.detection_graph)

        # Definite input and output Tensors for detection_graph
        for op in self.sess.graph.get_operations():
            print(op.name, op.type)
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def processFrame(self, image):
        # Expand dimensions since the trained_model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image, axis=0)
        # Actual detection.
        start_time = time.time()
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})
        end_time = time.time()

        #print("Elapsed Time:", end_time-start_time)

        im_height, im_width,_ = image.shape
        boxes_list = [None for i in range(boxes.shape[1])]
        for i in range(boxes.shape[1]):
            boxes_list[i] = (int(boxes[0,i,0] * im_height),
                        int(boxes[0,i,1]*im_width),
                        int(boxes[0,i,2] * im_height),
                        int(boxes[0,i,3]*im_width))

        return boxes_list, scores[0].tolist(), [int(x) for x in classes[0].tolist()], int(num[0])

    def close(self):
        self.sess.close()
        self.default_graph.close()










def take_screenshot(monitor_number):
    # for i in range(0, 10):
    #     pyautogui.moveRel(100, 50)
    #     time.sleep(0.05)

    # face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
    # full_body = cv.CascadeClassifier('haarcascade_fullbody.xml')
    # eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')
    # upper_body = cv.CascadeClassifier('haarcascade_upperbody.xml')
    #
    model_path = 'frozen_inference_graph.pb'
    #model_path = './frozen_models/frozen_graph.pb'
    #model_path = 'saved_model.pb'
    odapi = DetectorAPI(path_to_ckpt=model_path)
    threshold = 0.7
    #
    # hog = cv.HOGDescriptor()
    # hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())
    img = None
    sct = mss.mss()

    while (True):

        #time.sleep(0.01)


        # Get information of monitor 2
        #monitor_number = 1
        mon = sct.monitors[monitor_number]

        # The screen part to capture
        monitor = {
            "top": mon["top"],
            "left": mon["left"],
            "width": mon["width"],
            "height": mon["height"],
            "mon": monitor_number,
        }
        #output = "sct-mon{mon}_{top}x{left}_{width}x{height}.png".format(**monitor)

        # Grab the data
        #sct_img = sct.grab(monitor)
        img = np.array(sct.grab(monitor))  # BGR Image

        img = cv.cvtColor(img, cv.COLOR_BGRA2BGR)
        #img = cv.resize(img, (1280, 720))

        boxes, scores, classes, num = odapi.processFrame(img)

        for i in range(len(boxes)):
            # Class 1 represents human
            if classes[i] == 1 and scores[i] > threshold:
                box = boxes[i]
                cv.rectangle(img, (box[1], box[0]), (box[3], box[2]), (255, 0, 0), 2)


        # convert to gray scale of each frames
        # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Resizing the Image
        # img = imutils.resize(img,
        #                        width=min(800, img.shape[1]))
        #
        # (regions, _) = hog.detectMultiScale(img,
        #                                     winStride=(8, 8),
        #                                     )
        #
        # # Drawing the regions in the Image
        # for (x, y, w, h) in regions:
        #     cv.rectangle(img, (x, y),
        #                   (x + w, y + h),
        #                   (0, 0, 255), 2)





        # Detects faces of different sizes in the input image
        #faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        # faces = upper_body.detectMultiScale(gray,1.3,6)

        # for (x, y, w, h) in faces:
        #     # To draw a rectangle in a face
        #     cv.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
            # roi_gray = gray[y:y + h, x:x + w]
            # roi_color = img[y:y + h, x:x + w]
            #
            # # Detects eyes of different sizes in the input image
            # eyes = eye_cascade.detectMultiScale(roi_gray)
            #
            # # To draw a rectangle in eyes
            # for (ex, ey, ew, eh) in eyes:
            #     cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 127, 255), 2)

        cv.imshow('Computer Vision', img)

        if cv.waitKey(1) == ord('q'):
            cv.destroyAllWindows()
            break


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    take_screenshot(1)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
