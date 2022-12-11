from assignment_2_lib import take_a_photo, drive

import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.stats import linregress

BALL_RADIUS = 0.3 / 2
DEG_TO_RAD = math.pi / 180
CAMERA_FOV_RAD = 75 * DEG_TO_RAD


def preprocess_raw_image(img_raw):
    cropped_raw = img_raw[:400, :]
    bgr = cv2.cvtColor(cropped_raw, cv2.COLOR_RGBA2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return hsv


def get_ball_mask(img_hsv):
    lower1 = np.array([0, 100, 100])
    upper1 = np.array([10, 255, 255])
    lower2 = np.array([170, 100, 100])
    upper2 = np.array([179, 255, 255])
    mask = cv2.inRange(img_hsv, lower1, upper1) + \
        cv2.inRange(img_hsv, lower2, upper2)
    return mask


def detect_ball(ball_mask):
    mask = ball_mask > 0
    sum_ver = np.sum(mask, axis=0)
    sum_hor = np.sum(mask, axis=1)

    center_ver = np.arange(len(sum_ver))[sum_ver > 0]
    center_ver = np.sum(center_ver) / len(center_ver)

    center_hor = np.arange(len(sum_hor))[sum_hor > 0]
    center_hor = np.sum(center_hor) / len(center_hor)

    center = [np.rint(center_ver).astype(int), np.rint(center_hor).astype(int)]
    visible_diameter = max(np.max(sum_ver), np.max(sum_hor))
    return center, visible_diameter


def get_camera_distance_to_ball(visible_diameter, img_width):
    # formulas obtained by solving the problem on the sheet of paper
    a = (visible_diameter / img_width) * math.tan(CAMERA_FOV_RAD / 2)
    b = (BALL_RADIUS / a) ** 2
    return math.sqrt(b + BALL_RADIUS ** 2)


def obtain_distance_per_step(_):
    # this has been used to obtain distance the car travels per physics step
    # it is not used anywhere now
    dists = []
    steps = []

    for i in range(30):
        car = 1
        photo = take_a_photo(car, False)
        # dists.append(_forward_distance(photo))
        steps.append(i * 100)
        drive(car, True, 0)

    lin = linregress(steps, dists)

    ys = [x * lin.slope + lin.intercept for x in steps]
    print('slope', lin.slope)

    plt.plot(steps, dists)
    plt.plot(steps, ys)
    plt.show()


def forward_distance(img_raw):
    img_hsv = preprocess_raw_image(img_raw)
    ball_mask = get_ball_mask(img_hsv)
    _, visible_diameter = detect_ball(ball_mask)

    distance = get_camera_distance_to_ball(
        visible_diameter, img_width=img_hsv.shape[1])

    approx_car_length = 0.6
    distance_to_drive = distance - BALL_RADIUS - approx_car_length

    # obtained experimentally using obtain_distance_per_step()
    # in scenario it's been written that drive() simulates car for 100 steps, but in code this value's been set to 250
    # so we divide obtained speed by 2.5
    distance_per_step = 0.001054868274005231 / 2.5
    steps = round(distance_to_drive / distance_per_step)
    return steps


def find_a_ball(car):
    # TODO: you should write this function using
    # - take_a_photo(car)
    # - drive(car, forward, direction)
    pass


def move_a_ball(car):
    # TODO: you should write this function using
    # - take_a_photo(car)
    # - drive(car, forward, direction)
    pass
