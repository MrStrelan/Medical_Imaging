import numpy as np
import matplotlib.pyplot as plt
from skimage import transform, morphology

# A class for processing mole images
class Mole:
    def __init__(self, image_id):
        # Load and prepare image and mask
        self.img, self.mask = self.prepare_im(image_id)
        # Find maximum height and rotated mask
        self.mask, self.high = self.max_height(self.mask)
        # Calculate the mole's perimeter
        self.perim = self.perimeter(self.mask)

    # Method that loads and prepares image and mask for further processing
    # Input: image id
    # Output: image and mask
    def prepare_im(self,im_id):
        # Set path to image and mask directories
        path = '.'
        # Load image and scale it down by a factor of 4
        im = plt.imread(path + "\\Images\\" + im_id + '.png')
        im = transform.resize(im, (im.shape[0] // 4, im.shape[1] // 4), anti_aliasing=True)
        # Load mask and scale it down by a factor of 4
        gt = plt.imread(path + '\\Masks_png\\' + im_id + '.png')
        gt = transform.resize(gt, (gt.shape[0] // 4, gt.shape[1] // 4), anti_aliasing=False) #Setting it to True creates values that are not 0 or 1
        return im, gt

    # Method that finds the maximum height of the mole and rotates the mask to the correct orientation
    # Input: mask of the image
    # Output: rotated mask and maximum height of the mole
    def max_height(self, img_msk):
        # Sum the number of pixels in each column
        pixels_in_col = np.sum(img_msk, axis=0)
        # Find the column with the largest number of pixels
        max_pixels_in_col = np.max(pixels_in_col)
        # Rotate the mask by 45 degrees until the largest width is found
        for i in range(1,8):
            rot_mask = transform.rotate(img_msk, 45*i)
            height = np.max(np.sum(rot_mask, axis=0))
            if height > max_pixels_in_col:
                max_pixels_in_col = height
                final_msk = rot_mask
        return final_msk, max_pixels_in_col

    # Method that calculates the perimeter of the mole
    # Input: mask of the image
    # Output: perimeter of the mole
    def perimeter(self, img_msk):
        # Define a brush shape for erosion
        brush = morphology.disk(2)
        # Erode the mask to remove small details
        mask_cleaned = morphology.binary_erosion(img_msk, brush)
        # Calculate the perimeter by subtracting the cleaned mask from the original mask
        perimeter_im = img_msk - mask_cleaned
        return perimeter_im

    # Method that displays the calculated perimeter
    def show_per(self):
        plt.imshow(self.perim, cmap='gray')
        plt.show()```
