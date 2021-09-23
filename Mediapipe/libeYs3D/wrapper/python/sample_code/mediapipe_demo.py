"""
It's a demo to preview 2D frame with opencv.

Usage:
    Hot Keys:
        * Q/q/Esc: Quit
"""

import sys
import time

import cv2

import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


from eys3d import Pipeline, logger

# For depth-roi calculated
x = y = 0
# For cv preview
COLOR_ENABLE = DEPTH_ENABLE = False


def mediapipe_demo(device, config):
    global COLOR_ENABLE, DEPTH_ENABLE

    pipe = Pipeline(device=device)
    conf = config
    if conf.get_config()['colorHeight']:
        COLOR_ENABLE = True
    if conf.get_config()['depthHeight']:
        DEPTH_ENABLE = True
    pipe.start(conf)

    # Flag defined
    flag = dict({
        'exposure': True,
        'Extend_IR': True,
        'HW_pp': True,
    })

    camera_property = device.get_cameraProperty()
    ir_property = device.get_IRProperty()
    ir_value = ir_property.get_IR_value()
    status = 'play'

    depth_roi = 10  # default is 10

    # default value of z range
    z_range = device.get_z_range()
    ZNEAR_DEFAULT = z_range["Near"]
    ZFAR_DEFAULT = z_range["Far"]
    logger.info("Default ZNear: {}, ZFar: {}".format(ZNEAR_DEFAULT,
                                                     ZFAR_DEFAULT))

    with mp_hands.Hands(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            max_num_hands=2) as hands:

        while 1:
            try:
                if COLOR_ENABLE:
                    color_success, color_frame = pipe.wait_color_frame()

                    if color_success:
                        rgb_cframe = color_frame.get_rgb_data().reshape(color_frame.get_height(),
                                                                        color_frame.get_width(), 3)
                        rgb_cframe.flags.writeable = False
                        results = hands.process(rgb_cframe)
                        processed_img = cv2.cvtColor(rgb_cframe, cv2.COLOR_RGB2BGR)
                        if results.multi_hand_landmarks:
                            for hand_landmarks in results.multi_hand_landmarks:
                                mp_drawing.draw_landmarks(
                                    processed_img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                        cv2.imshow('MediaPipe Hands5', processed_img)

                if DEPTH_ENABLE:
                    dret, dframe = pipe.wait_depth_frame()
                    if dret:
                        bgr_dframe = cv2.cvtColor(
                            dframe.get_rgb_data().reshape(dframe.get_height(),
                                                          dframe.get_width(), 3),
                            cv2.COLOR_RGB2BGR)
                        cv2.imshow("Depth image", bgr_dframe)
                        z_map = dframe.get_depth_ZD_value().reshape(
                            dframe.get_height(), dframe.get_width())
                        cv2.setMouseCallback("Depth image", depth_roi_callback)
                        cv2.displayStatusBar(
                            "Depth image", " Z = {:.2f}, Z-ROI = {}".format(
                                calculate_roi(x, y, dframe.get_width(),
                                              dframe.get_height(), depth_roi,
                                              z_map), depth_roi), 1000)
            except (TypeError, ValueError, cv2.error, KeyError) as e:
                print(e)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                pipe.pause()
                break

    pipe.stop()



def calculate_roi(x, y, w, h, depth_roi, z_map):
    if depth_roi > 1:
        roi_x = max(x - depth_roi / 2.0, 0)
        roi_y = max(y - depth_roi / 2.0, 0)
        roi_x2 = roi_x + depth_roi
        roi_y2 = roi_y + depth_roi

        if roi_x2 > w:
            roi_x2 = w
            roi_x = roi_x2 - depth_roi
        if roi_y2 > h:
            roi_y2 = h
            roi_y = roi_y2 - depth_roi

        depth_roi_sum = 0
        depth_roi_count = 0

        for y_ in range(int(roi_y), int(roi_y2)):
            for x_ in range(int(roi_x), int(roi_x2)):
                z_value = z_map[y_][x_]
                if z_value:
                    depth_roi_sum += z_value
                    depth_roi_count += 1
        if depth_roi_count:
            z_value = depth_roi_sum / depth_roi_count
        else:
            z_value = 0
    else:
        z_value = z_map[y][x]

    return z_value


def depth_roi_callback(event, x_, y_, flag, param):
    # Update coord x and y
    global x, y
    x = x_
    y = y_
