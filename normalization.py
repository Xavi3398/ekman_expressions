import numpy as np
from imutils import face_utils
import dlib
import cv2
from scipy.spatial import Delaunay


def transform_mask(predictor, maxx, maxy, vertical_space, image_cropped, image_lime, method='warpaffine'):
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

    # Transform RGB and LIME mask images using WarpAffine. This is the most efficient method.
    if method == 'warpaffine':
        copy_images = copy_images_warpaffine
    # Transform RGB and LIME mask images using neighbor triangle search
    elif method == 'neighbors':
        copy_images = copy_images_neighbors
    # Transform RGB and LIME mask images
    elif method == 'normal':
        copy_images = copy_images_normal
    else:
        raise ValueError('Invalid method')
    
    # Run method to transform images
    copy_images(maxx, maxy, image_cropped, image_transformed, image_lime, lime_transformed, points_transformed, points,
                    triangle_list)

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


def copy_images_normal(maxx, maxy, image, image_result, mask, mask_result, points_mask, points, triangle_list):
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


def copy_images_neighbors(maxx, maxy, image, image_result, mask, mask_result, points_mask, points, triangle_list):
    """ Function that transforms an image to normalize. Instead of directly iterating through triangle_list,
        neighbor point triangles will be checked first to find out which triangle the point belongs to in a more
        efficient way.

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

    memory = np.full((maxx, maxy), -1, dtype=int)

    for x in range(0, maxx):
        for y in range(0, maxy):
            pt = np.array([x, y])
            if y > 0 and memory[x, y - 1] != -1:
                trian = triangle_list[memory[x, y - 1]]
                v1n = points_mask[trian[0], :]
                v2n = points_mask[trian[1], :]
                v3n = points_mask[trian[2], :]
                if point_in_triangle(pt, v1n, v2n, v3n):
                    memory[x, y] = memory[x, y - 1]
                else:
                    for tri_num, tri in enumerate(triangle_list):
                        v1 = points_mask[tri[0], :]
                        v2 = points_mask[tri[1], :]
                        v3 = points_mask[tri[2], :]
                        if point_in_triangle(pt, v1, v2, v3):
                            memory[x, y] = tri_num
                            break
            elif x > 0 and memory[x - 1, y] != -1:
                trian = triangle_list[memory[x - 1, y]]
                v1n = points_mask[trian[0], :]
                v2n = points_mask[trian[1], :]
                v3n = points_mask[trian[2], :]
                if point_in_triangle(pt, v1n, v2n, v3n):
                    memory[x, y] = memory[x - 1, y]
                else:
                    for tri_num, tri in enumerate(triangle_list):
                        v1 = points_mask[tri[0], :]
                        v2 = points_mask[tri[1], :]
                        v3 = points_mask[tri[2], :]
                        if point_in_triangle(pt, v1, v2, v3):
                            memory[x, y] = tri_num
                            break
            else:
                for tri_num, tri in enumerate(triangle_list):
                    v1 = points_mask[tri[0], :]
                    v2 = points_mask[tri[1], :]
                    v3 = points_mask[tri[2], :]
                    if point_in_triangle(pt, v1, v2, v3):
                        memory[x, y] = tri_num
                        break
            if memory[x, y] != -1:
                def_tri = triangle_list[memory[x, y]]
                v1 = points_mask[def_tri[0], :]
                v2 = points_mask[def_tri[1], :]
                v3 = points_mask[def_tri[2], :]
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

                v1o = points[def_tri[0], :]
                v2o = points[def_tri[1], :]
                v3o = points[def_tri[2], :]
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


def copy_images_warpaffine(image, image_result, mask, mask_result, points_mask, points, triangle_list):
    """ Function that transforms an image to normalize using WarpAffine method.

    Args:
        maxx: Maximum width to process in the resulting image.
        maxy: Maximum height to process in the resulting image.
        image: Source image.
        image_result: Resulting image.
        points_mask: Normalized mask points.
        points: Points from the image to normalize.
        triangle_list: List of triangles used for transformation, defined according to points and points_mask.
    """

    for tri in triangle_list:
        v1 = points_mask[tri[0], :]
        v2 = points_mask[tri[1], :]
        v3 = points_mask[tri[2], :]

        v1o = points[tri[0], :]
        v2o = points[tri[1], :]
        v3o = points[tri[2], :]

        # Define input and output triangles
        tri1 = np.float32([[v1o, v2o, v3o]])
        tri2 = np.float32([[v1, v2, v3]])

        # Find bounding box.
        r1 = cv2.boundingRect(tri1)
        r2 = cv2.boundingRect(tri2)

        # Offset points by left top corner of the respective rectangles
        tri1Cropped = []
        tri2Cropped = []

        for i in range(0, 3):
            tri1Cropped.append(((tri1[0][i][0] - r1[0]), (tri1[0][i][1] - r1[1])))
            tri2Cropped.append(((tri2[0][i][0] - r2[0]), (tri2[0][i][1] - r2[1])))

        # Crop input image
        x, y, w, h = r1
        x1 = max(0, x)
        y1 = max(0, y)
        img1Cropped = image[y1:y1 + h, x1:x1 + w]
        # Crop input mask
        maskcropped = mask[y1:y1 + h, x1:x1 + w]

        if img1Cropped.shape[0] == 0 or img1Cropped.shape[1] == 0:
            continue

        # Given a pair of triangles, find the affine transform.
        warpMat = cv2.getAffineTransform(np.float32(tri1Cropped), np.float32(tri2Cropped))

        # Apply the Affine Transform just found to the src image
        img2Cropped = cv2.warpAffine(img1Cropped, warpMat, (
            image_result[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]].shape[1],
            image_result[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]].shape[0]), None, flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REFLECT_101)

        # Apply the Affine Transform to the src mask
        maskresultcropped = cv2.warpAffine(maskcropped, warpMat, (
            mask_result[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]].shape[1],
            mask_result[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]].shape[0]), None, flags=cv2.INTER_LINEAR,
                                           borderMode=cv2.BORDER_REFLECT_101)

        # Get mask by filling triangle
        tri_mask = np.zeros_like(image_result[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]], dtype=np.float32)
        cv2.fillConvexPoly(tri_mask, np.int32(tri2Cropped), (1.0, 1.0, 1.0), 16, 0)

        img2Cropped = img2Cropped * tri_mask

        # Copy triangular region of the rectangular patch to the output image
        image_result[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = image_result[r2[1]:r2[1] + r2[3],
                                                                 r2[0]:r2[0] + r2[2]] * ((1.0, 1.0, 1.0) - tri_mask)

        image_result[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = image_result[r2[1]:r2[1] + r2[3],
                                                                 r2[0]:r2[0] + r2[2]] + img2Cropped

        # Get mask by filling triangle
        tri_mask2 = np.zeros_like(mask_result[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]], dtype=np.float32)
        cv2.fillConvexPoly(tri_mask2, np.int32(tri2Cropped), (1.0, 1.0, 1.0), 16, 0)

        maskresultcropped = maskresultcropped * tri_mask2

        # Copy triangular region of the rectangular patch to the output mask
        mask_result[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = mask_result[r2[1]:r2[1] + r2[3],
                                                                 r2[0]:r2[0] + r2[2]] * ((1.0, 1.0, 1.0) - tri_mask2)

        mask_result[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = mask_result[r2[1]:r2[1] + r2[3],
                                                                 r2[0]:r2[0] + r2[2]] + maskresultcropped
