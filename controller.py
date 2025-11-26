import numpy as np
from numpy.typing import ArrayLike

WHEELBASE = 3.6

_DT = 0.01

_steer_integral = 0.0
_steer_prev_error = 0.0

def find_nearest_index(state, racetrack):
    pos = state[0:2]
    d = np.linalg.norm(racetrack.centerline - pos, axis=1)
    return int(np.argmin(d))

def compute_curvature(racetrack, i):
    n = len(racetrack.centerline)

    p_prev = racetrack.centerline[(i - 1) % n]
    p_curr = racetrack.centerline[i]
    p_next = racetrack.centerline[(i + 1) % n]

    a = np.linalg.norm(p_curr - p_prev)
    b = np.linalg.norm(p_next - p_curr)
    c = np.linalg.norm(p_next - p_prev)

    if a == 0 or b == 0 or c == 0:
        return 0.0

    area = 0.5 * abs(np.cross(p_curr - p_prev, p_next - p_prev))

    return 4.0 * area / (a * b * c)
    
def speed_reference(state, racetrack, idx):

    start_x, start_y = racetrack.centerline[0]
    dx = state[0] - start_x
    dy = state[1] - start_y
    dist_from_start = np.hypot(dx, dy)

    max_k = 0.0
    N = len(racetrack.centerline)

    for j in range(30):  # lookahead window
        k = compute_curvature(racetrack, (idx + j) % N)
        max_k = max(max_k, k)

    for j in range(45):
        k = compute_curvature(racetrack, (idx + j) % N)
        if k > 0.06:
            return 7.0  
    
    if max_k > 0.07:
        return 4.0       # hairpin / U-turn
    elif max_k > 0.05:
        return 7.0
    elif max_k > 0.03:
        return 42.0
    elif max_k > 0.02:
        return 55.0
    elif max_k > 0.01:
        return 60.0
    else:
        return 90.0       # straight-line speed

def steering_reference(state, racetrack, idx):
    max_k = 0.0
    N = len(racetrack.centerline)
    for j in range(10):  # lookahead window
        k = compute_curvature(racetrack, (idx + j) % N)
        max_k = max(max_k, k)
    
    if max_k > 0.06: 
        LA = 2
    elif max_k > 0.05: 
        LA = 4
    else: 
        LA = 4
    tgt_idx = (idx + LA) % len(racetrack.centerline)

    car_x, car_y = state[0], state[1]
    phi = state[4]

    tx, ty = racetrack.centerline[tgt_idx]

    dx = tx - car_x
    dy = ty - car_y

    desired_heading = np.arctan2(dy, dx)

    heading_error = np.arctan2(
        np.sin(desired_heading - phi),
        np.cos(desired_heading - phi)
    )
    L = WHEELBASE
    dist = np.hypot(dx, dy)

    if dist < 1e-3:
        return 0.0

    delta_r = np.arctan2(2 * L * np.sin(heading_error), dist)
    return delta_r

def longitudinal_control(state, vr):
    v = state[3]
    e = vr - v

    Kp = 4.0
    return Kp * e *10000000

def steering_control(state, delta_r):
    global _steer_integral, _steer_prev_error
    delta = state[2]
    error = delta_r - delta

    Kp = 6.0
    Ki = 27.0
    Kd = 0.05
    _steer_integral += error * _DT
    _steer_integral = np.clip(_steer_integral, -2.0, 2.0)

    derivative = (error - _steer_prev_error) / _DT

    v_delta = Kp * error + Ki * _steer_integral + Kd * derivative

    _steer_prev_error = error 

    return v_delta

def controller(state: ArrayLike, parameters: ArrayLike, racetrack):
    idx = find_nearest_index(state, racetrack)
    # print("nearest idx: ", idx)
    vr = speed_reference(state, racetrack, idx)
    delta_r = steering_reference(state, racetrack, idx)
    return np.array([delta_r, vr])

v_delta_history = []

def lower_controller(state: ArrayLike, desired: ArrayLike, parameters: ArrayLike):
    global v_delta_history
    delta_r, vr = desired[0], desired[1]
    a = longitudinal_control(state, vr)
    v_delta = steering_control(state, delta_r)

    clipped_v_delta = np.clip(v_delta, parameters[7], parameters[9])
    clipped_a       = np.clip(a,        parameters[8], parameters[10])

    v_delta_history.append(clipped_v_delta)
    # print("v_delta:", clipped_v_delta, ", a:", clipped_a)

    return np.array([clipped_v_delta, clipped_a])