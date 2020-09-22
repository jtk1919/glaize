import cv2
import numpy as np
import os
import sys
import scipy.cluster.hierarchy as hcluster


class IMAGE(object):
    """
    Image object with narrow AI methods for determining scale assuming the visibility of
    the vertical edges of the card
    """

    def __init__(self, path_to_image, name_of_image, save_images=False, card_width_mm=85.60):
        """

        :param path_to_image: str
        :param name_of_image: str
        :return: None if name isn't valid
        """
        self.valid_image = self.check_valid_extension(name_of_image)
        if self.valid_image != True:
            print("'{}' is not an image or does not have valid image type".format(name_of_image))
            return None
        # Instantiate path and name of file
        self.read_path = str(path_to_image)
        self.name = str(name_of_image)
        self.save_images = save_images
        self.line_distance_mm = card_width_mm
        self.good_quality = True
        """
        Read RGB image and record dimensionality
        Make grayscale copy makes edges detection copy from selected parameters
        """
        self.image = cv2.imread(str(path_to_image) + '/' + str(name_of_image))
        self.n_H, self.n_W, self.n_C = self.image.shape

        self.image_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.image_edges = cv2.Canny(self.image_gray, 50, 100, apertureSize=3)
        if self.save_images:
            cv2.imwrite('edges.jpg', self.image_edges)

    """
    Validation of filename methods
    """
    def check_valid_extension(self, name_of_image):
        """Returns boolean if the extension is valid

        :param name_of_image:
        :return: bool True if name of extension is valid
        """
        self.name = str(name_of_image[:-4])
        self.extension = str(name_of_image[-4:])
        extension_types_list = self.define_extension_types()
        if self.extension in extension_types_list:
            return True
        else:
            return False

    def define_extension_types(self):
        """Returns list of valid extensions define in this method

        :return: list of valid types
        """
        extension_types = ['.png', '.jpg']
        return extension_types

    """
    Methods to determine scale
    """
    def hough_lines(self, threshold=100):
        """

        :param threshold: number of votes required to accept line
        :return: original image with horizonatal lines drawn,
                the complete set of lines,
                the y-position of the horizontal lines
        """
        minLineLength = 100
        maxLineGap = 10
        lines = cv2.HoughLines(self.image_edges, 1, np.pi / 180, threshold, minLineLength, maxLineGap)
        n_lines = len(lines)

        widths = []
        n_ver_lines = 0
        image = self.image
        for i in range(n_lines):
            rho, theta = lines[i][0][0], lines[i][0][1]

            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            if (theta < 0.05) or (np.pi - 0.05 < theta):
                n_ver_lines += 1
                cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                widths.append([x0, 0])
        if len(widths) < 2:
            self.good_quality = False
        return image, lines, widths

    def cluster_lines(self, positions, threshold=20):
        """

        :param positions: list of positions [y,x] for points on each line
        :param threshold: max distance allowed for distance between points in a cluster
        :return: list of labels of which cluster each line belongs to
        """
        clusters = hcluster.fclusterdata(np.array(positions), threshold, criterion="distance")
        n_clusters = len(set(clusters))
        print("Number of clusters: {}".format(n_clusters))
        if n_clusters < 2:
            self.good_quality = False
        return clusters

    def cluster_averages(self, clusters, height_cart):
        """

        :param clusters: list of cluster positional arguments ((n_lines)
        :param height_cart: list of heights in cartesian coordinates (n_lines)
        :return: list of all line heights (n_lines),
                 list of averages of each cluster sorted in ascending order (n_clusters)
        """
        _, idx, counts = np.unique(clusters, return_inverse=True, return_counts=True)
        heights = []
        for point in height_cart:
            heights.append(point[0])
        sum_clusters = np.bincount(idx, weights=heights)
        avg_clusters = np.sort(sum_clusters / counts)
        return heights, avg_clusters

    def max_distance(self, sorted_list):
        """

        :param sorted_list: list values sorted in ascending order (n)
        :return: distance: (np_array) distance between leftmost and rightmost
        """
        distance = sorted_list[-1] - sorted_list[0]
        return distance

    def point_of_scale(self):
        """

        :return: scalar, median distance in pixels between lines
        """
        image_with_lines, lines, heights_cart = self.hough_lines(threshold=100)
        if self.is_good_quality == False:
            return None
        clusters = self.cluster_lines(heights_cart, threshold=20)
        if self.is_good_quality == False:
            return None
        heights, avg_clusters = self.cluster_averages(clusters, heights_cart)
        distance = self.max_distance(avg_clusters)
        if self.save_images:
            cv2.imwrite('houghlines.jpg', image_with_lines)
        """
        We will assume that the median value in distances will
        most likely be correct, as the error from one or two spurious 
        clusters will significantly effect the mean
        """
        self.line_distance_pixels = distance
        self.pixels_per_mm = self.line_distance_pixels/self.line_distance_mm
        return self.line_distance_pixels

    def is_good_quality(self):
        """

        :return: bool
        """
        if self.good_quality:
            return True
        else:
            print("\nThe image is bad quality.\n")
            return False
