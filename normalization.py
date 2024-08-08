import numpy as np
from imutils import face_utils
import dlib
import cv2
from scipy.spatial import Delaunay


def transform_mask(predictor, maxx, maxy, vertical_space, image_cropped, image_lime):
    """ Transforms an image to normalize it.
    
    Args:
        predictor: Predictor to find points in image_cropped.
        maxx: Maximum width.
        maxy: Maximum height.
        vertical_space: Vertical space added.
        image_cropped: Image with the face where points are detected.
        image_lime: Image with the mask to normalize.

    Returns:
        image_cropped: Image with points and triangles.
        image_lime: Mask with points and triangles.
        image_transformed: Normalized image.
        lime_transformed: Normalized mask.
    """

    # Init transformed images
    image_transformed = np.zeros([maxy, maxx, 3], dtype='uint8')
    lime_transformed = np.zeros([maxy, maxx, 3], dtype='uint8')

    # Get points from the RGB image and from the transformed RGB image
    points = get_points(image_cropped, predictor, image_cropped.shape[1], image_cropped.shape[0], 0)
    points_transformed = get_points(image_transformed, predictor, maxx, maxy, vertical_space)
    
    # Get triangles
    triangles = Delaunay(points)
    triangle_list = triangles.simplices.copy()
    np.insert(triangle_list, 1, triangles.simplices[0].copy())

    # Transform RGB and LIME mask images
    copy_images(maxx, maxy, image_cropped, image_transformed, image_lime, lime_transformed, points_transformed, points, triangle_list)
                 
    # Dibujar puntos y triangulos sobre cada imagen
    draw_points_and_triangles(image_cropped, points, triangles)
    draw_points_and_triangles(image_lime, points, triangles)
    # draw_points_and_triangles(image_transformed, points_transformed, triangles)
    # draw_points_and_triangles(lime_transformed, points_transformed, triangles)

    return image_cropped, image_lime, image_transformed, lime_transformed


def get_points(img, predictor, width_image, height_image, vertical_space):
    """ Get points from an image.
    
    Args:
        img: Image.
        predictor: Predictor to find points in image.
        width_image: Image width.
        height_image: Image height.
        vertical_space: Vertical space added.

    Returns:
        points: Points.
    """

    # Frame where the face is located (in this case, since it is already a face image, we take the size of the image
    # If it were an image that included a landscape with a person, at this point a face detector should be applied
    # and the coordinates of where the face is should be obtained
    dlib_rect = dlib.rectangle(0, 0, width_image, height_image - vertical_space)

    # Detect landmarks in image
    detected_landmarks = predictor(img, dlib_rect)
    detected_landmarks = face_utils.shape_to_np(detected_landmarks)

    # Copies the 17 chin points to the top of the image.
    for i in range(0, 17):
        newdl = detected_landmarks[i].copy()
        newdl[1] = 0
        detected_landmarks = np.append(detected_landmarks, [newdl], axis=0)

    # Add borders
    detected_landmarks = np.append(detected_landmarks, [[0, 0]], axis=0)
    detected_landmarks = np.append(detected_landmarks, [[0, height_image]], axis=0)
    detected_landmarks = np.append(detected_landmarks, [[width_image, 0]], axis=0)
    detected_landmarks = np.append(detected_landmarks, [[width_image, height_image]], axis=0)

    points = detected_landmarks

    return points


def draw_points_and_triangles(img, points, triangles):
    """ Draw points and triangles in an image.

    Args:
        img: Image.
        points: Points.
        triangles: Triangles.
    """

    # Draw triangles on the image
    for triangulo in triangles.simplices:
        cv2.line(img, (points[triangulo[0], 0], points[triangulo[0], 1]),
                 (points[triangulo[1], 0], points[triangulo[1], 1]), (255, 255, 0), 1)
        cv2.line(img, (points[triangulo[1], 0], points[triangulo[1], 1]),
                 (points[triangulo[2], 0], points[triangulo[2], 1]), (255, 255, 0), 1)
        cv2.line(img, (points[triangulo[2], 0], points[triangulo[2], 1]),
                 (points[triangulo[0], 0], points[triangulo[0], 1]), (255, 255, 0), 1)
    
    # Draw points on the image
    for (x, y) in points:
        cv2.circle(img, (x, y), 2, (0, 0, 255), -1)


def sign(p1, p2, p3):
    """ Determines on which side of the line p2-p3 the point p1 is located.
        Although it returns a number, it is determined by the sign.

    Args:
        p1: Point 1.
        p2: Point 2.
        p3: Point 3.

    Returns:
        Sign.
    """

    return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])


def point_in_triangle(pt, v1, v2, v3):
    """ Determines if the point pt is inside the triangle formed by the vertices v1, v2, and v3.

    Args:
        pt: Point.
        v1: Vertex 1.
        v2: Vertex 2.
        v3: Vertex 3.

    Returns:
        True if the point is inside the triangle, False otherwise.
    """

    d1 = sign(pt, v1, v2)
    d2 = sign(pt, v2, v3)
    d3 = sign(pt, v3, v1)

    # It is inside if all three are negative or all three are positive
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

    return not (has_neg and has_pos)


def copy_images(maxx, maxy, image, image_result, mask, mask_result, points_mask, points, triangle_list):
    """ Function that transforms an image to normalize.

    Args:
        maxx: Maximum width to process in the resulting image.
        maxy: Maximum height to process in the resulting image.
        image: Source image.
        image_result: Resulting image.
        points_mask: Normalized mask points.
        points: Points from the image to normalize.
        triangle_list: List of triangles used for transformation, defined according to points and points_mask.
    """
    width_image = image.shape[0]
    height_image = image.shape[1]
    for x in range(0, maxx):
        for y in range(0, maxy):
            pt = np.array([x, y])
            for tri in triangle_list:
                v1 = points_mask[tri[0], :]
                v2 = points_mask[tri[1], :]
                v3 = points_mask[tri[2], :]
                if point_in_triangle(pt, v1, v2, v3):
                    v1v2 = v2 - v1
                    v1v3 = v3 - v1

                    N = np.cross(v1v2, v1v3)
                    area = np.linalg.norm(N) / 2
                    if area == 0:
                        continue

                    edge1 = v3 - v2
                    vp1 = pt - v2
                    C = np.cross(edge1, vp1)
                    u = (np.linalg.norm(C) / 2) / area

                    edge2 = v1 - v3
                    vp3 = pt - v3
                    C = np.cross(edge2, vp3)
                    v = (np.linalg.norm(C) / 2) / area

                    w = 1 - u - v

                    v1o = points[tri[0], :]
                    v2o = points[tri[1], :]
                    v3o = points[tri[2], :]
                    pto = u * v1o + v * v2o + w * v3o

                    try:
                        ptox = int(pto[0])
                        ptoy = int(pto[1])
                        if ptox >= 0 and ptox < width_image and ptoy >= 0 and ptoy < height_image:
                            image_result[y, x] = image[ptoy, ptox]
                            mask_result[y, x] = mask[ptoy, ptox]
                    except:
                        print(pt)
                        print(v1)
                        print(v2)
                        print(v3)
                        print(u)
                        print(v)
                        print(w)
                        print(pto)
                    break
