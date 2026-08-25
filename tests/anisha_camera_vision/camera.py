import numpy as np
import cv2

# initialize the camera
width = 1920
height = 1080
cam = cv2.VideoCapture(0)   # 1 -> index of camera (0 for device native, 1 for USB webcam)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
s, img = cam.read()

if s:    # frame captured without any errors
    img_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # get aruco markers
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, rejected = detector.detectMarkers(img_bw)
    # print("Detected markers:", ids)
    if ids is not None:
        top_left = [1, 1]
        top_right = [img_bw.shape[1]-2, 1]
        bottom_right = [img_bw.shape[1]-2, img_bw.shape[0]-2]
        bottom_left = [1, img_bw.shape[0]-2]
        for i in range(ids.shape[0]):
            # print(ids[i], corners[i])

            if ids[i][0] == 10:
                bottom_right = corners[i][0][2]
            if ids[i][0] == 20:
                top_right = corners[i][0][2]
            if ids[i][0] == 30:
                bottom_left = corners[i][0][2]
            if ids[i][0] == 40:
                top_left = corners[i][0][0]

        sorted_corners = np.array(
            [top_left, top_right, bottom_right, bottom_left], dtype="float32")
        
        # Define the size of the new image
        width, height = int(500*1.294), 500
        destination_corners = np.array(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")

        # Get the perspective transform matrix
        matrix = cv2.getPerspectiveTransform(
            sorted_corners, destination_corners)

        # Apply the perspective transformation
        transformed_image = cv2.warpPerspective(img_bw, matrix, (width, height))

        # cv2.imshow('Cropped Image', transformed_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        cv2.imwrite("cropped_image.jpg", transformed_image) # save image

        _, thresh = cv2.threshold(transformed_image, 90, 255, cv2.THRESH_BINARY_INV)
        cv2.imwrite("threshold.jpg", thresh) # save image
    
        # cv2.aruco.drawDetectedMarkers(img_bw, rejected)
        cv2.aruco.drawDetectedMarkers(img_bw, corners, ids)
        cv2.imshow('Detected Markers', img_bw)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # cv2.namedWindow("cam-test", cv2.WINDOW_AUTOSIZE)
    # cv2.imshow("cam-test", img_bw)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    cv2.imwrite("captured_image.jpg", img_bw) # save image
else:
    print("Error: Unable to capture image from camera.")