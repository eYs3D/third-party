"""
It's a demo to preview 2D frame with opencv.

Usage:
    Hot Keys:
        * Q/q/Esc: Quit
        * M increase IR emitter level
        * N decrease IR emitter level
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
            status = {
                -1: status,
                27: 'exit',  # Esc
                ord('q'): 'exit',
                ord('Q'): 'exit',
                ord('e'): 'exposure',
                ord('E'): 'exposure',
                65470: 'snapshot',  # F1
                65471: 'dump_frame_info',  # F2
                ord('i'): 'extend_IR',
                ord('I'): 'extend_IR',
                ord('m'): 'increased_IR',
                ord('M'): 'increased_IR',
                ord('n'): 'decreased_IR',
                ord('N'): 'decreased_IR',
                ord('0'): 'reset-z-range',
                ord('1'): 'z-range-setting-1',
                ord('2'): 'z-range-setting-2',
                65361: 'play',  # Left arrow
            }[cv2.waitKeyEx(2)]
            if status == 'play':
                continue
            if status == 'exit':
                cv2.destroyAllWindows()
                pipe.pause()
                break
            if status == 'exposure':
                flag["exposure"] = not (flag["exposure"])
                if flag["exposure"]:
                    logger.info("Enable exposure")
                    camera_property.enable_AE()
                else:
                    logger.info("Disable exposure")
                    camera_property.disable_AE()
                status = 'play'
            if status == 'snapshot':
                device.do_snapshot()
                logger.info(status)
                status = 'play'
            if status == 'dump_frame_info':
                device.dump_frame_info()
                logger.info(status)
                status = 'play'
            if status == 'extend_IR':
                flag["Extend_IR"] = not (flag["Extend_IR"])
                if flag["Extend_IR"]:
                    logger.info("Enable extend IR")
                    ir_property.enable_extendIR()
                else:
                    logger.info("Disable extend IR")
                    ir_property.disable_extendIR()
                status = 'play'
            if status == 'increased_IR':
                ir_value = min(ir_value + 1, ir_property.get_IR_max())
                ir_property.set_IR_value(ir_value)
                time.sleep(0.1)
                logger.info("Increase IR, current value = {}".format(ir_value))
                status = 'play'
            if status == 'decreased_IR':
                ir_value = max(ir_value - 1, ir_property.get_IR_min())
                ir_property.set_IR_value(ir_value)
                time.sleep(0.1)
                logger.info("Decrease IR, current value = {}".format(ir_value))
                status = 'play'
            if status == 'reset-z-range':
                logger.info("Reset z range")
                device.set_z_range(ZNEAR_DEFAULT, ZFAR_DEFAULT)
                z_range = device.get_z_range()
                logger.info("ZNear: {}, ZFar:{}".format(
                    z_range["Near"], z_range["Far"]))
                status = 'play'
            if status == 'z-range-setting-1':
                device.set_z_range(1234, 5678)
                z_range = device.get_z_range()
                logger.info("ZNear: {}, ZFar:{}".format(
                    z_range["Near"], z_range["Far"]))
                status = 'play'
            if status == 'z-range-setting-2':
                device.set_z_range(1200, 1600)
                z_range = device.get_z_range()
                logger.info("ZNear: {}, ZFar:{}".format(
                    z_range["Near"], z_range["Far"]))
                status = 'play'


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
