import cv2
import numpy as np
from scipy.spatial import Delaunay

from ekman_expressions.normalization import get_points, draw_points_and_triangles

def draw_expression(predictor, maxx, maxy, vertical_space, exp_name='anger', 
                    color=(255, 255, 255), show_landmarks=False):
    """ Draw an expression in an image.

    Args:
        predictor: Predictor to find points in image.
        maxx: Maximum width.
        maxy: Maximum height.
        vertical_space: Vertical space added.
        exp_name: str. Expression name.
        color: tuple. Color to draw.
        show_landmarks: bool. Show landmarks.

    Returns:
        Image with the expression drawn
    """
    img_result = np.zeros([maxy, maxx, 3], dtype='uint8')
    points = get_points(img_result, predictor, maxx, maxy, vertical_space)

    exp_name = exp_name.lower()

    if exp_name == 'anger':
        draw_anger(img_result, points, color)
    elif exp_name == 'fear':
        draw_fear(img_result, points, color)
    elif exp_name == 'disgust':
        draw_disgust(img_result, points, color)
    elif exp_name == 'happiness':
        draw_happiness(img_result, points, color)
    elif exp_name == 'sadness':
        draw_sadness(img_result, points, color)
    elif exp_name == 'surprise':
        draw_surprise(img_result, points, color)

    if show_landmarks:
        triangles = Delaunay(points) # Get triangles
        draw_points_and_triangles(img_result, points, triangles)

    return img_result

def draw_anger(image_result, points_mask, color):
    """ Draw anger expression in an image.

    Args:
        image_result: Resulting image with the expression.
        points_mask: Points in the image.
        color: Color to draw.
    """
    draw_triangle_index(color, image_result, points_mask, 38, 39, 20)
    draw_triangle_index(color, image_result, points_mask, 21, 39, 20)
    draw_triangle_index(color, image_result, points_mask, 19, 20, 38)
    draw_triangle_index(color, image_result, points_mask, 19, 38, 37)
    draw_triangle_index(color, image_result, points_mask, 21, 22, 39)
    draw_triangle_index(color, image_result, points_mask, 22, 39, 40)
    draw_triangle_index(color, image_result, points_mask, 41, 39, 40)
    draw_triangle_index(color, image_result, points_mask, 41, 39, 38)
    draw_triangle_index(color, image_result, points_mask, 41, 42, 38)
    draw_triangle_index(color, image_result, points_mask, 37, 42, 38)
    p74a = (points_mask[19-1][0], points_mask[74-1][1])
    draw_triangle_iip(color, image_result, points_mask, 19, 20, p74a)
    draw_triangle_iip(color, image_result, points_mask, 75, 20, p74a)
    draw_triangle_index(color, image_result, points_mask, 21, 75, 20)
    draw_triangle_index(color, image_result, points_mask, 21, 75, 76)
    draw_triangle_index(color, image_result, points_mask, 21, 77, 76)
    draw_triangle_index(color, image_result, points_mask, 21, 77, 22)
    draw_triangle_index(color, image_result, points_mask, 23, 77, 22)
    draw_triangle_index(color, image_result, points_mask, 23, 77, 24)
    draw_triangle_index(color, image_result, points_mask, 78, 77, 24)
    draw_triangle_index(color, image_result, points_mask, 78, 25, 24)
    p79a = (points_mask[26-1][0], points_mask[79-1][1])
    draw_triangle_iip(color, image_result, points_mask, 78, 25, p79a)
    draw_triangle_iip(color, image_result, points_mask, 26, 25, p79a)

    draw_triangle_strip_index(color, image_result, points_mask, [40, 22, 28, 23, 43, 23, 44, 24, 44, 25, 45, 26, 46])
    draw_triangle_strip_index(color, image_result, points_mask, [43, 44, 48, 45, 47, 46])

    draw_triangle_strip_index(color, image_result, points_mask, [33, 34, 51, 52, 62, 63, 68, 67, 59, 58, 8, 9])
    draw_triangle_strip_index(color, image_result, points_mask, [35, 34, 53, 52, 64, 63, 66, 67, 57, 58, 10, 9])

    p37b = (points_mask[37-1][0], points_mask[30-1][1])
    p40b = (points_mask[40-1][0], points_mask[30-1][1])
    p43b = (points_mask[43-1][0], points_mask[30-1][1])
    p46b = (points_mask[46-1][0], points_mask[30-1][1])
    draw_triangle_ipp(color, image_result, points_mask, 37, p40b, p37b)
    draw_triangle_iip(color, image_result, points_mask, 37, 40, p40b)

    draw_triangle_ipp(color, image_result, points_mask, 43, p46b, p43b)
    draw_triangle_iip(color, image_result, points_mask, 43, 46, p46b)

def draw_fear(image_result, points_mask, color):
    """ Draw fear expression in an image.

    Args:
        image_result: Resulting image with the expression.
        points_mask: Points in the image.
        color: Color to draw.
    """
    draw_triangle_index(color, image_result, points_mask, 38, 39, 20)
    draw_triangle_index(color, image_result, points_mask, 21, 39, 20)
    draw_triangle_index(color, image_result, points_mask, 19, 20, 38)
    draw_triangle_index(color, image_result, points_mask, 19, 38, 37)
    draw_triangle_index(color, image_result, points_mask, 21, 22, 39)
    draw_triangle_index(color, image_result, points_mask, 22, 39, 40)
    draw_triangle_index(color, image_result, points_mask, 41, 39, 40)
    draw_triangle_index(color, image_result, points_mask, 41, 39, 38)
    draw_triangle_index(color, image_result, points_mask, 41, 42, 38)
    draw_triangle_index(color, image_result, points_mask, 37, 42, 38)
    p74a = (points_mask[19-1][0], points_mask[74-1][1])
    draw_triangle_iip(color, image_result, points_mask, 19, 20, p74a)
    draw_triangle_iip(color, image_result, points_mask, 75, 20, p74a)
    draw_triangle_index(color, image_result, points_mask, 21, 75, 20)
    draw_triangle_index(color, image_result, points_mask, 21, 75, 76)
    draw_triangle_index(color, image_result, points_mask, 21, 77, 76)
    draw_triangle_index(color, image_result, points_mask, 21, 77, 22)
    draw_triangle_index(color, image_result, points_mask, 23, 77, 22)
    draw_triangle_index(color, image_result, points_mask, 23, 77, 24)
    draw_triangle_index(color, image_result, points_mask, 78, 77, 24)
    draw_triangle_index(color, image_result, points_mask, 78, 25, 24)
    p79a = (points_mask[26-1][0], points_mask[79-1][1])
    draw_triangle_iip(color, image_result, points_mask, 78, 25, p79a)
    draw_triangle_iip(color, image_result, points_mask, 26, 25, p79a)

    draw_triangle_strip_index(color, image_result, points_mask, [40, 22, 28, 23, 43, 23, 44, 24, 44, 25, 45, 26, 46])
    draw_triangle_strip_index(color, image_result, points_mask, [43, 44, 48, 45, 47, 46])

    p37b = (points_mask[37-1][0], points_mask[30-1][1])
    p40b = (points_mask[40-1][0], points_mask[30-1][1])
    p43b = (points_mask[43-1][0], points_mask[30-1][1])
    p46b = (points_mask[46-1][0], points_mask[30-1][1])
    draw_triangle_ipp(color, image_result, points_mask, 37, p40b, p37b)
    draw_triangle_iip(color, image_result, points_mask, 37, 40, p40b)

    draw_triangle_ipp(color, image_result, points_mask, 43, p46b, p43b)
    draw_triangle_iip(color, image_result, points_mask, 43, 46, p46b)

    draw_triangle_iip(color, image_result, points_mask, 29, 40, p40b)
    draw_triangle_index(color, image_result, points_mask, 29, 40, 28)
    draw_triangle_index(color, image_result, points_mask, 29, 28, 43)
    draw_triangle_iip(color, image_result, points_mask, 29, 43, p43b)

    pt5 = points_mask[5 - 1]
    pt49 = points_mask[49-1]
    pt5_49 = (int((pt5[0]+ pt49[0])/2), int((pt5[1]+ pt49[1])/2))
    pt13 = points_mask[13 - 1]
    pt55 = points_mask[55-1]
    pt13_55 = (int((pt13[0]+ pt55[0])/2), int((pt13[1]+ pt55[1])/2))

    draw_triangle_ipp(color, image_result, points_mask, 49, p37b, pt5_49)
    draw_triangle_ipp(color, image_result, points_mask, 49, p37b, p40b)
    draw_triangle_iip(color, image_result, points_mask, 49, 50, p40b)

    draw_triangle_ipp(color, image_result, points_mask, 55, p46b, pt13_55)
    draw_triangle_ipp(color, image_result, points_mask, 55, p46b, p43b)
    draw_triangle_iip(color, image_result, points_mask, 55, 54, p43b)

def draw_disgust(image_result, points_mask, color):
    """ Draw disgust expression in an image.

    Args:
        image_result: Resulting image with the expression.
        points_mask: Points in the image.
        color: Color to draw.
    """
    pt5 = points_mask[5 - 1]
    pt49 = points_mask[49-1]
    pt5_49 = (int((pt5[0]+ pt49[0])/2), int((pt5[1]+ pt49[1])/2))
    draw_triangle_iip(color, image_result, points_mask, 37, 60, pt5_49)
    draw_triangle_index(color, image_result, points_mask, 60, 37, 42)
    draw_triangle_index(color, image_result, points_mask, 60, 42, 41)
    draw_triangle_index(color, image_result, points_mask, 60, 41, 40)

    draw_triangle_strip_index(color, image_result, points_mask, [41, 49, 40, 50, 22, 68, 28, 67, 23, 66, 43, 54, 48, 55])

    pt13 = points_mask[13 - 1]
    pt55 = points_mask[55-1]
    pt13_55 = (int((pt13[0]+ pt55[0])/2), int((pt13[1]+ pt55[1])/2))
    draw_triangle_iip(color, image_result, points_mask, 46, 56, pt13_55)
    draw_triangle_index(color, image_result, points_mask, 56, 46, 47)
    draw_triangle_index(color, image_result, points_mask, 56, 47, 48)
    draw_triangle_index(color, image_result, points_mask, 56, 48, 43)

    draw_triangle_index(color, image_result, points_mask, 22, 23, 28)

def draw_happiness(image_result, points_mask, color):
    """ Draw happiness expression in an image.

    Args:
        image_result: Resulting image with the expression.
        points_mask: Points in the image.
        color: Color to draw.
    """
    draw_triangle_strip_index(color, image_result, points_mask, [40, 39, 41, 38, 42, 37])
    draw_triangle_strip_index(color, image_result, points_mask, [43, 44, 48, 45, 47, 46])

    p37b = (points_mask[37-1][0], points_mask[30-1][1])
    p40b = (points_mask[40-1][0], points_mask[30-1][1])
    p43b = (points_mask[43-1][0], points_mask[30-1][1])
    p46b = (points_mask[46-1][0], points_mask[30-1][1])
    draw_triangle_ipp(color, image_result, points_mask, 37, p40b, p37b)
    draw_triangle_iip(color, image_result, points_mask, 37, 40, p40b)

    draw_triangle_ipp(color, image_result, points_mask, 43, p46b, p43b)
    draw_triangle_iip(color, image_result, points_mask, 43, 46, p46b)

    draw_triangle_iip(color, image_result, points_mask, 29, 40, p40b)
    draw_triangle_index(color, image_result, points_mask, 29, 40, 28)
    draw_triangle_index(color, image_result, points_mask, 29, 28, 43)
    draw_triangle_iip(color, image_result, points_mask, 29, 43, p43b)

    pt5 = points_mask[5 - 1]
    pt49 = points_mask[49-1]
    pt5_49 = (int((pt5[0]+ pt49[0])/2), int((pt5[1]+ pt49[1])/2))
    pt13 = points_mask[13 - 1]
    pt55 = points_mask[55-1]
    pt13_55 = (int((pt13[0]+ pt55[0])/2), int((pt13[1]+ pt55[1])/2))

    draw_triangle_ipp(color, image_result, points_mask, 49, p37b, pt5_49)
    draw_triangle_ipp(color, image_result, points_mask, 49, p37b, p40b)
    draw_triangle_iip(color, image_result, points_mask, 49, 50, p40b)

    draw_triangle_ipp(color, image_result, points_mask, 55, p46b, pt13_55)
    draw_triangle_ipp(color, image_result, points_mask, 55, p46b, p43b)
    draw_triangle_iip(color, image_result, points_mask, 55, 54, p43b)


    draw_triangle_index(color, image_result, points_mask, 50, 32, 40)
    draw_triangle_fan_index(color, image_result, points_mask, [49, 32, 5, 6, 7, 60, 50])

    draw_triangle_index(color, image_result, points_mask, 54, 36, 43)
    draw_triangle_fan_index(color, image_result, points_mask, [55, 36, 13, 12, 11, 56, 54])

def draw_sadness(image_result, points_mask, color):
    """ Draw sadness expression in an image.

    Args:
        image_result: Resulting image with the expression.
        points_mask: Points in the image.
        color: Color to draw.
    """
    p37b = (points_mask[37-1][0], points_mask[30-1][1])
    p40b = (points_mask[40-1][0], points_mask[30-1][1])
    p43b = (points_mask[43-1][0], points_mask[30-1][1])
    p46b = (points_mask[46-1][0], points_mask[30-1][1])
    p21b = (points_mask[21-1][0], points_mask[77-1][1])
    p24b = (points_mask[24-1][0], points_mask[77-1][1])
    draw_triangle_ipp(color, image_result, points_mask, 37, p40b, p37b)
    draw_triangle_iip(color, image_result, points_mask, 37, 40, p40b)

    draw_triangle_ipp(color, image_result, points_mask, 43, p46b, p43b)
    draw_triangle_iip(color, image_result, points_mask, 43, 46, p46b)

    draw_triangle_iip(color, image_result, points_mask, 29, 40, p40b)
    draw_triangle_index(color, image_result, points_mask, 29, 40, 28)
    draw_triangle_index(color, image_result, points_mask, 29, 28, 43)
    draw_triangle_iip(color, image_result, points_mask, 29, 43, p43b)

    pt5 = points_mask[5 - 1]
    pt49 = points_mask[49-1]
    pt5_49 = (int((pt5[0]+ pt49[0])/2), int((pt5[1]+ pt49[1])/2))
    pt13 = points_mask[13 - 1]
    pt55 = points_mask[55-1]
    pt13_55 = (int((pt13[0]+ pt55[0])/2), int((pt13[1]+ pt55[1])/2))

    draw_triangle_ipp(color, image_result, points_mask, 49, p37b, pt5_49)
    draw_triangle_ipp(color, image_result, points_mask, 49, p37b, p40b)
    draw_triangle_iip(color, image_result, points_mask, 49, 50, p40b)

    draw_triangle_ipp(color, image_result, points_mask, 55, p46b, pt13_55)
    draw_triangle_ipp(color, image_result, points_mask, 55, p46b, p43b)
    draw_triangle_iip(color, image_result, points_mask, 55, 54, p43b)

    draw_triangle_strip_index(color, image_result, points_mask, [37, 42, 38, 41, 39, 40])
    draw_triangle_strip_index(color, image_result, points_mask, [46, 47, 45, 48, 44, 43])

    draw_triangle_iip(color, image_result, points_mask, 50, 60, pt5_49)
    draw_triangle_iip(color, image_result, points_mask, 56, 54, pt13_55)

    pt1_86 = interpolate(points_mask, 1, 86, 0.5)
    draw_triangle_index(color, image_result, points_mask, 1, 37, 19)
    draw_triangle_iip(color, image_result, points_mask, 1, 37, p37b)
    draw_triangle_iip(color, image_result, points_mask, 1, 19, pt1_86)


    pt17_88 = interpolate(points_mask, 17, 88, 0.5)
    draw_triangle_index(color, image_result, points_mask, 17, 46, 26)
    draw_triangle_iip(color, image_result, points_mask, 17, 46, p46b)
    draw_triangle_iip(color, image_result, points_mask, 17, 26, pt17_88)

    draw_triangle_strip_index(color, image_result, points_mask, [39,21,40,22,28,23,43,24,44])
    draw_triangle_fan_index(color, image_result, points_mask, [77, 21,22,23,24])
    draw_triangle_iip(color, image_result, points_mask, 21, 77, p21b)
    draw_triangle_iip(color, image_result, points_mask, 24, 77, p24b)

    draw_triangle_iip(color, image_result, points_mask, 24, 77, p24b)

def draw_surprise(image_result, points_mask, color):
    """ Draw surprise expression in an image.

    Args:
        image_result: Resulting image with the expression.
        points_mask: Points in the image.
        color: Color to draw.
    """
    p23b = (points_mask[23-1][0], points_mask[77-1][1])
    p26b = (points_mask[26-1][0], points_mask[77-1][1])
    draw_triangle_iip(color, image_result, points_mask, 23, 26, p23b)
    draw_triangle_ipp(color, image_result, points_mask, 26, p26b, p23b)
    draw_triangle_strip_index(color, image_result, points_mask, [43, 23, 48, 24, 47, 25, 46, 26])

    p22b = (points_mask[22-1][0], points_mask[77-1][1])
    p19b = (points_mask[19-1][0], points_mask[77-1][1])
    draw_triangle_iip(color, image_result, points_mask, 22, 19, p22b)
    draw_triangle_ipp(color, image_result, points_mask, 19, p19b, p22b)
    draw_triangle_strip_index(color, image_result, points_mask, [40, 22, 41, 21, 42, 20, 37, 19])

    draw_triangle_strip_index(color, image_result, points_mask, [33, 32, 51, 50, 62, 60, 68, 7, 59, 8])
    draw_triangle_strip_index(color, image_result, points_mask, [33, 34, 51, 52, 62, 63, 68, 67, 59, 58, 8, 9])
    draw_triangle_strip_index(color, image_result, points_mask, [35, 34, 53, 52, 64, 63, 66, 67, 57, 58, 10, 9])
    draw_triangle_strip_index(color, image_result, points_mask, [35, 36, 53, 54, 64, 56, 66, 11, 57, 10])

def draw_triangle_iip(color, image, points, index1, index2, pt3):
    """ Draw a triangle with two indexes and a point.

    Args:
        color: Color to draw.
        image: Image to draw.
        points: Points in the image.
        index1: Index of the first point.
        index2: Index of the second point.
        pt3: Third point.
    """
    pt1 = points[index1-1]
    pt2 = points[index2-1]
    draw_triangle(color, image, pt1, pt2, pt3)

def draw_triangle_ipp(color, image, points, index1, pt2, pt3):
    """ Draw a triangle with one index and two points.

    Args:
        color: Color to draw.
        image: Image to draw.
        points: Points in the image.
        index1: Index of the first point.
        pt2: Second point.
        pt3: Third point.
    """
    pt1 = points[index1-1]
    draw_triangle(color, image, pt1, pt2, pt3)

def draw_triangle_strip_index(color, image, points_mask, list):
    """ Draw a triangle strip with indexes.

    Args:
        color: Color to draw.
        image: Image to draw.
        points_mask: Points in the image.
        list: List of indexes.
    """
    index = 0
    while index < len(list)-2:
        draw_triangle_index(color, image, points_mask, list[index], list[index+1], list[index+2])
        index+=1

def draw_triangle_fan_index(color, image, points_mask, list):
    """ Draw a triangle fan with indexes.

    Args:
        color: Color to draw.
        image: Image to draw.
        points_mask: Points in the image.
        list: List of indexes.
    """
    index = 0
    while index < len(list)-2:
        draw_triangle_index(color, image, points_mask, list[0], list[index+1], list[index+2])
        index+=1


def draw_triangle_index(color, image, points, index1, index2, index3):
    """ Draw a triangle with indexes.

    Args:
        color: Color to draw.
        image: Image to draw.
        points: Points in the image.
        index1: Index of the first point.
        index2: Index of the second point.
        index3: Index of the third point.
    """
    pt1 = points[index1-1]
    pt2 = points[index2-1]
    pt3 = points[index3-1]
    draw_triangle(color, image, pt1, pt2, pt3)

def draw_triangle(color, image, pt1, pt2, pt3):
    """ Draw a triangle using three points.

    Args:
        color: Color to draw.
        image: Image to draw.
        pt1: First point.
        pt2: Second point.
        pt3: Third point.
    """
    triangle_cnt = np.array( [pt1, pt2, pt3] )
    cv2.drawContours(image, [triangle_cnt], 0, color, -1)

def interpolate(points_mask, index1, index2, factor):
    """ Interpolate two points.

    Args:
        points_mask: Points in the image.
        index1: Index of the first point.
        index2: Index of the second point.
        factor: Factor of interpolation.

    Returns:
        Interpolated point.
    """
    return (int(points_mask[index1-1][0]*factor + points_mask[index2-1][0] *(1-factor)), int(points_mask[index1-1][1]*factor + points_mask[index2-1][1] * (1-factor)))
