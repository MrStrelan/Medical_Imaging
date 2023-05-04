import numpy as np
import matplotlib.pyplot as plt
from skimage import transform, morphology
import cv2
import os

# A class for processing mole images
class Mole:
    def __init__(self, image_id):
        self.id = image_id
        # Load and prepare image and mask
        self.img, self.mask = self.prepare_im()
        # Find maximum height and rotated mask
        self.mask, self.high = self.max_height()
        # Calculate the mole's perimeter
        self.perim = self.perimeter()
        # Calculate the mole's symmetry
        self.symmetry = self.symmetry_detection()
        # Fuse the mask and the original picture
      #  self.seg = self.mask_segm(self.img,self.mask,image_id)          UNCOMMENT THIS LINE TO RUN THE SEGMENTATION
                                                                        #THIS IS A TEMPORARY FIX       
      #  self.seg = self.mask_segm()

    # Method that loads and prepares image and mask for further processing
    # Input: image id
    # Output: image and mask
    def prepare_im(self):
        # Set path to image and mask directories
        path = '.\\Medical_Imaging'
        # Load image and scale it down by a factor of 4
        im = plt.imread(path + "\\Images\\" + self.id + '.png')
        im = transform.resize(im, (im.shape[0] // 4, im.shape[1] // 4), anti_aliasing=True)
        # Load mask and scale it down by a factor of 4
        gt = plt.imread(path + '\\Masks_png\\' + "mask_"+ self.id + '.png')
        gt = transform.resize(gt, (gt.shape[0] // 4, gt.shape[1] // 4), anti_aliasing=False) #Setting it to True creates values that are not 0 or 1
        return im, gt

    # Method that finds the maximum height of the mole and rotates the mask to the correct orientation
    # Input: mask of the image
    # Output: rotated mask and maximum height of the mole
    def max_height(self):
        # Sum the number of pixels in each column
        pixels_in_col = np.sum(self.mask, axis=0)
        # Find the column with the largest number of pixels
        max_pixels_in_col = np.max(pixels_in_col)
        # Rotate the mask by 45 degrees until the largest width is found
        for i in range(1,8):
            rot_mask = transform.rotate(self.mask, 45*i)
            height = np.max(np.sum(rot_mask, axis=0))
            if height > max_pixels_in_col:
                max_pixels_in_col = height
        return rot_mask, max_pixels_in_col

    # Method that calculates the perimeter of the mole
    # Input: mask of the image
    # Output: perimeter of the mole
    def perimeter(self):
        #brush saves this shape:
        #[[0 0 1 0 0]
         #[0 1 1 1 0]
         #[1 1 1 1 1]
         #[0 1 1 1 0]
         #[0 0 1 0 0]]
        brush = morphology.disk(2)

        # Erode the mask to remove small details
        mask_cleaned = morphology.binary_erosion(self.mask, brush)
        # Calculate the perimeter by subtracting the cleaned mask from the original mask
        perimeter_im = self.mask - mask_cleaned
        return perimeter_im
    
    # Method that detects symmetry in the mole
    def symmetry_detection(self):
        # Save the perimeter image
        plt.imsave("perimeter.png", self.perim, format='png', cmap='gray')
        # Load the grayscale image
        img_gray = cv2.imread("perimeter.png", cv2.IMREAD_GRAYSCALE)

        # Threshold the image to create a binary image
        ret, img_binary = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY)

        # Find the contours in the binary image
        contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        symmetry_values = []
        for contour in contours:
            # Calculate the centroid of the object
            M = cv2.moments(contour)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            # Find the distance between each point in the contour and the centroid
            distances = []
            for point in contour:
                px, py = point[0]
                distance = np.sqrt((px - cx)**2 + (py - cy)**2)
                distances.append(distance)
            mean_distance = sum(distances) / len(distances)

            # Find the corresponding points on the other side of the centroid
            corresponding_points = []
            for point in contour:
                px, py = point[0]
                distance = np.sqrt((px - cx)**2 + (py - cy)**2)
                dx = int(cx + (cx - px) / distance * mean_distance)
                dy = int(cy + (cy - py) / distance * mean_distance)
                corresponding_points.append((dx, dy))

            # Calculate the distances between the pairs of points
            pair_distances = []
            for i in range(len(contour)):
                px, py = contour[i][0]
                qx, qy = corresponding_points[i]
                distance = np.sqrt((px - qx)**2 + (py - qy)**2)
                pair_distances.append(distance)

            # Calculate the mean and standard deviation of the distances
            mean_distance = np.mean(pair_distances)
            std_distance = np.std(pair_distances)

            # Calculate symmetry value for this object
            symmetry_value = std_distance
            symmetry_values.append(symmetry_value)

        return symmetry_values


  #  def mask_segm(self):
"""
    def mask_segm(self,im, gt):
        # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(5, 3))
        # axes[0].imshow(im)
        # axes[1].imshow(gt, cmap='gray')
        #fig.tight_layout()
        im2 = self.img.copy()
        im2[self.mask == 0] = 0
        im2 = im.copy()
        im2[mask==0] = 0                                                    #UNCOMMENT. THIS IS A TEMPORARY FIX
        # Save the resulting image in a folder called "results"
        path = '.'
        plt.imsave(path + '\\Fused_Images\\' + self.id + '_segm.png',im2)
        return im2

    """

  #  ---------------------------------- Print functions ----------------------------------
"""


    # Function prints out symmetry
    def symmetric(self):
        # Check if the object is symmetric
        if self.symmetry < 10:
            print("Object is symmetric:", self.symmetry)
        else:
            print("Object is not symmetric:", self.symmetry)

    # Method that displays the calculated perimeter
    def show_per(self):
        
        plt.imshow(self.perim, cmap='gray')
        plt.show()

    def show_seg_mask(self):

        # Display 
        plt.imshow(self.seg)
        plt.show()

   """
"""
    # Load the image
    image = Image.open("example_image.png")

    # Convert the image to black and white
    image = image.convert("1")

    # Get the pixel values as a matrix of 0s and 1s
    matrix = list(image.getdata())
    matrix = [matrix[i:i+image.width] for i in range(0, len(matrix), image.width)]

    # Print the matrix
    for row in matrix:
        print(row)

"""