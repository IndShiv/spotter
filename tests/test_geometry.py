import numpy as np

from spotter.signals.geometry import (
    bilateral_average,
    height_ratio,
    joint_angle,
    torso_lean,
    vertical_displacement,
)


def test_joint_angle_straight_line_is_180():
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    c = np.array([0.0, 2.0, 0.0])
    assert abs(joint_angle(a, b, c) - 180.0) < 1e-6


def test_joint_angle_right_angle_is_90():
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    c = np.array([1.0, 1.0, 0.0])
    assert abs(joint_angle(a, b, c) - 90.0) < 1e-6


def test_joint_angle_degenerate_zero_length_returns_zero():
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([0.0, 0.0, 0.0])
    c = np.array([1.0, 1.0, 0.0])
    assert joint_angle(a, b, c) == 0.0


def test_vertical_displacement_sign_and_scale():
    point = np.array([0.0, 0.8, 0.0])
    reference = np.array([0.0, 0.5, 0.0])
    assert vertical_displacement(point, reference, scale=0.3) > 0  # point is lower
    assert vertical_displacement(reference, point, scale=0.3) < 0  # reference is lower


def test_height_ratio_upright_vs_curled():
    hip = np.array([0.0, 0.5, 0.0])
    shoulder = np.array([0.0, 0.2, 0.0])
    upright = height_ratio(hip, shoulder)
    curled_hip = np.array([0.0, 0.35, 0.0])
    curled = height_ratio(curled_hip, shoulder)
    assert curled > upright  # hips closer to shoulders -> ratio closer to 1


def test_torso_lean_upright_is_zero():
    shoulder = np.array([0.5, 0.2, 0.0])
    hip = np.array([0.5, 0.5, 0.0])
    assert torso_lean(shoulder, hip) < 1e-6


def test_torso_lean_increases_with_forward_offset():
    hip = np.array([0.5, 0.5, 0.0])
    small = torso_lean(np.array([0.52, 0.2, 0.0]), hip)
    large = torso_lean(np.array([0.6, 0.2, 0.0]), hip)
    assert large > small > 0


def test_bilateral_average_weights_by_visibility():
    # Occluded side (low visibility) should barely move the result.
    result = bilateral_average(left=100.0, right=0.0, left_vis=1.0, right_vis=0.05)
    assert result > 90.0


def test_bilateral_average_equal_visibility_is_plain_mean():
    assert bilateral_average(10.0, 20.0, 1.0, 1.0) == 15.0
