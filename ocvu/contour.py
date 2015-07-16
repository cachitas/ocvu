import numpy as np
import cv2


class Contour(object):
    """Provides a user-friendly object defining a contour in OpenCV.
    Adapted from:
    http://answers.opencv.org/question/6043/segmentation-and-contours/
    """

    def __init__(self, contour):
        self.contour = contour
        self.size = len(contour)

    @property
    def position(self):
        """Top left coordinate of the contour bounding box."""
        x, y, _, _ = self.bounding_box
        return (x, y)

    @property
    def area(self):
        """Contour.area - Area bounded by the contour region"""
        return cv2.contourArea(self.contour)

    @property
    def perimeter(self):
        """Lorem"""
        return cv2.arcLength(self.contour, closed=True)

    @property
    def approx(self):
        """Lorem"""
        return cv2.approxPolyDP(self.contour, 0.02 * self.perimeter, True)

    @property
    def hull(self):
        """Lorem"""
        return cv2.convexHull(self.contour)

    @property
    def moments(self):
        """Lorem"""
        return cv2.moments(self.contour)

    @property
    def bounding_box(self):
        """Lorem"""
        return cv2.boundingRect(self.contour)

    @property
    def centroid(self):
        if self.moments['m00'] != 0.0:
            cx = self.moments['m10'] / self.moments['m00']
            cy = self.moments['m01'] / self.moments['m00']
            return (cx, cy)
        else:
            raise ValueError("Region has zero area")

    @property
    def ellipse(self):
        """Fits an ellipse and returns the rotated rectangle in which
        the ellipse is inscribed.
        The returned value is the tuple:
            ((x, y), (major_axis, minor_axis), angle)
        """
        return cv2.fitEllipse(self.contour)

    @property
    def diameter(self):
        """EquivDiameter: diameter of circle with same area as region"""
        return np.sqrt(4 * self.moments['m00'] / np.pi)

    def draw(self, image, **kwargs):
        """Draw contour.
        `kwargs` are passed to `cv2.rectangle` method.
        Default color is green.
        """
        color = kwargs.pop('color', (0, 255, 0))
        cv2.drawContours(image, [self.contour], 0, color, **kwargs)

    def draw_approx(self, image, **kwargs):
        """Draw approximated contour.
        `kwargs` are passed to `cv2.rectangle` method.
        Default color is red.
        """
        color = kwargs.pop('color', (0, 0, 255))
        cv2.drawContours(image, [self.approx], 0, color, **kwargs)

    def draw_hull(self, image, **kwargs):
        """Draw convex hull.
        `kwargs` are passed to `cv2.rectangle` method.
        Default color is blue.
        """
        color = kwargs.pop('color', (255, 0, 0))
        cv2.drawContours(image, [self.hull], 0, color, **kwargs)

    def draw_bounding_box(self, image, **kwargs):
        """Draw contour's bounding box on image.
        `kwargs` are passed to `cv2.rectangle` method.
        Default color is white.
        """
        x, y, w, h = self.bounding_box
        pt1 = (x, y)
        pt2 = (x + w, y + h)
        color = kwargs.pop('color', (255, 255, 255))
        cv2.rectangle(image, pt1, pt2, color, **kwargs)

    def draw_ellipse(self, image, **kwargs):
        """Draw contour's ellipse.
        `kwargs` are passed to `cv2.circle` method.
        """
        color = kwargs.pop('color', (0, 204, 204))
        thickness = kwargs.pop('thickness', 2)
        cv2.ellipse(image, self.ellipse, color, thickness, **kwargs)

    def draw_centroid(self, image, radius=3, **kwargs):
        """Draw contour's centroid.
        `kwargs` are passed to `cv2.circle` method.
        """
        color = kwargs.pop('color', (0, 255, 0))
        thickness = kwargs.pop('thickness', -1)
        center = tuple(map(int, self.centroid))
        cv2.circle(image, center, radius, color, thickness, **kwargs)
