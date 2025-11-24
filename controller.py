import numpy as np
from numpy.typing import ArrayLike

###############################################
# Helper: Find nearest point on centerline
###############################################
def find_nearest_index(state, racetrack):
    car_pos = state[0:2]
    dists = np.linalg.norm(racetrack.centerline - car_pos, axis=1)
    return np.argmin(dists)


###############################################
# Helper: Compute curvature at index i
###############################################
def compute_curvature(racetrack, i):
    # wrap indices
    p_prev = racetrack.centerline[(i - 1) % len(racetrack.centerline)]
    p_curr = racetrack.centerline[i]
    p_next = racetrack.centerline[(i + 1) % len(racetrack.centerline)]

    v1 = p_curr - p_prev
    v2 = p_next - p_curr

    # curvature = angle difference / distance
    angle1 = np.arctan2(v1[1], v1[0])
    angle2 = np.arctan2(v2[1], v2[0])

    dtheta = np.unwrap([angle1, angle2])[1] - angle1
    ds = np.linalg.norm(v1)

    if ds < 1e-6:
        return 0.0
    return dtheta / ds


###############################################
# S1: Speed reference generator
###############################################
def speed_reference(state, racetrack, i):
    # compute curvature from current steering, not raceline
    curvature = abs(np.tan(state[2]) / 3.6)


    # Conservative speed profile
    if curvature > 0.03:
        vr = 6     # tight turns
    elif curvature > 0.02:
        vr = 9     # medium turns
    elif curvature > 0.01:
        vr = 14     # mild turns
    else:
        vr = 20 
    return vr


###############################################
# S2: Steering reference generator (δr)
###############################################
def steering_reference(state, racetrack, i, lookahead=2):
    # lookahead point
    idx = (i + lookahead) % len(racetrack.centerline)
    target = racetrack.centerline[idx]

    car_pos = state[0:2]
    heading = state[4]

    # direction to target
    dx = target[0] - car_pos[0]
    dy = target[1] - car_pos[1]

    desired_heading = np.arctan2(dy, dx)
    heading_error = np.arctan2(np.sin(desired_heading - heading),
                               np.cos(desired_heading - heading))

    # convert heading error -> steering angle
    wheelbase = 3.6
    # δr approx = arctan(2*L*sin(err)/dist)
    dist = np.linalg.norm([dx, dy])
    if dist < 1e-3:
        return 0.0

    delta_r = np.arctan2(2 * wheelbase * np.sin(heading_error), dist)
    return delta_r


###############################################
# C1: Longitudinal controller → a
###############################################
def longitudinal_control(state, vr):
    v = state[3]
    error = vr - v

    Kp = 20
    a = Kp * error
    return np.clip(a, -10, 10)


###############################################
# C2: Steering controller → vδ
###############################################
def steering_control(state, delta_r):
    delta = state[2]
    error = delta_r - delta

    Kp = 2.0
    v_delta = Kp * error

    # limit steering rate
    return np.clip(v_delta, -0.4, 0.4)


###############################################
# HIGH-LEVEL CONTROLLER (calculates δr, vr)
###############################################
def controller(state: ArrayLike, parameters: ArrayLike, racetrack) -> ArrayLike:
    # 1. nearest track point
    idx = find_nearest_index(state, racetrack)

    # 2. S1: speed reference
    vr = speed_reference(state, racetrack, idx)

    # 3. S2: steering reference
    delta_r = steering_reference(state, racetrack, idx)

    return np.array([delta_r, vr])


###############################################
# LOWER-LEVEL CONTROLLER (calculates vδ, a)
###############################################
def lower_controller(
    state: ArrayLike, desired: ArrayLike, parameters: ArrayLike
) -> ArrayLike:

    delta_r = desired[0]
    vr = desired[1]

    # C1
    a = longitudinal_control(state, vr)

    # C2
    v_delta = steering_control(state, delta_r)

    return np.array([v_delta, a])
