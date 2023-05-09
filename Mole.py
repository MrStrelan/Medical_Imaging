from cmath import pi
import numpy as np
import matplotlib.pyplot as plt
from skimage import transform, morphology
import cv2

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
        self.symmetry = self.symmetry()
        # Fuse the mask and the original picture
        self.seg = self.overlay_segm()
        # Calculate compactness
        self.comp = self.compactness_calc()

    # Method that loads and prepares image and mask for further processing
    # Input: image id
    # Output: image and mask
    def prepare_im(self):
        # Set path to image and mask directories
        path = '.'
        # Load image and scale it down by a factor of 4
        im = plt.imread(path + "\\Images\\" + self.id + '.png')
        im = transform.resize(im, (im.shape[0] // 4, im.shape[1] // 4), anti_aliasing=True)
        # Load mask and scale it down by a factor of 4
        mask = plt.imread(path + '\\Masks_png\\' + "mask_"+ self.id + '.png')
        mask = transform.resize(mask, (mask.shape[0] // 4, mask.shape[1] // 4), anti_aliasing=False) #Setting it to True creates values that are not 0 or 1
        return im, mask

    # Method that finds the maximum height of the mole and rotates the mask to the correct orientation
    # Input: mask of the image
    # Output: rotated mask and maximum height of the mole
    def max_height(self):
        # Sum the number of pixels in each column
        pixels_in_col = np.sum(self.mask, axis=0)
        # Find the column with the largest number of pixels
        max_pixels_in_col = np.max(pixels_in_col)
        # Rotate the mask by 45 degrees until the largest width is found
        max_height_mask = self.mask  
        for i in range(1,24):
            rot_mask = transform.rotate(self.mask, 15*i)
            height = np.max(np.sum(rot_mask, axis=0))
            if height > max_pixels_in_col:
                max_pixels_in_col = height
                max_height_mask = rot_mask  
                
        return max_height_mask, max_pixels_in_col

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
    
    #counts symmetry proportionally to moles size
    def symmetry(self):

        # find the indices of the non-zero elements
        nonzero_rows, nonzero_cols = np.nonzero(self.mask)
    
        # Find the minimum and maximum row and column indices
        min_row, max_row = np.min(nonzero_rows), np.max(nonzero_rows)
        min_col, max_col = np.min(nonzero_cols), np.max(nonzero_cols)

        row_index=1
        if (max_row-min_row)%2==0:
            row_index = 0
        col_index=1
        if (max_col-min_col)%2==0:
            col_index = 0
        # Extract the subarray containing the non-zero elements
        cut_mask = self.mask[min_row:max_row+row_index, min_col:max_col+col_index]
        
        #Flipping image by x and y axes
        x_flipped = np.flip(cut_mask, axis=0)
        y_flipped = np.flip(cut_mask, axis=1)

        
        left_area = cut_mask[:,:cut_mask.shape[1]//2]
        #flipped right half
        right_area = y_flipped[:,:y_flipped.shape[1]//2]

        upper_area = cut_mask[:cut_mask.shape[0]//2,:]
        #flipped bottom half
        bottom_area = x_flipped[:x_flipped.shape[0]//2,:]
        
        #Uncamment and to see how compared halfs look like
        #plt.imshow(upper_area, cmap='gray')
        #plt.show()
        #plt.imshow(bottom_area, cmap='gray')
        #plt.show()
        
        #actually it asymmetry
        x_symmetry = upper_area-bottom_area
        #We make all values positive and equeal because we don't care about intensity of color but only shape
        x_symmetry[x_symmetry != 0] = 1
        y_symmetry = left_area-right_area
        y_symmetry[y_symmetry != 0] = 1
        
        #We make all values positive and equeal because we don't care about intensity of color but only shape
        area_mask = self.mask.copy()
        area_mask[area_mask != 0] = 1
        
        #deviding unsemetric areas by area of mole for proportionality
        symmetry_factor = (np.sum(y_symmetry)+np.sum(x_symmetry))/np.sum(area_mask)

        return symmetry_factor

    # Returns picture where eberything besides mask shown as black
    def overlay_segm(self):
        # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(5, 3))
        # axes[0].imshow(im)
        # axes[1].imshow(gt, cmap='gray')
        #fig.tight_layout()

        im2 = self.img.copy()
        im2[self.mask == 0] = 0
        
        # Save the resulting image in a folder called "results"
        path = '.'
        plt.imsave(path + '\\Fused_Images\\' + self.id + '_segm.png',im2)
        return im2

    #Calculate compactness from area an perimeter
    def compactness_calc(self):
        compactness = (np.sum(self.perim)*np.sum(self.perim))/4*pi*np.sum(self.mask)
        return compactness


    """
    ---------------------------------- Print functions ----------------------------------
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

    
