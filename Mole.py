import numpy as np
import matplotlib.pyplot as plt
from skimage import transform, morphology
from skimage.segmentation import slic, mark_boundaries

class Mole:
    def __init__(self, image_id):
        self.img, self.mask = self.prepare_im(image_id)
        self.mask, self.high = self.max_height(self.mask)
        self.perim = self.perimeter(self.mask)

    #prepare_im takes image id and returns image and drawn mask
    def prepare_im(self,im_id):

      path = '.'

      im = plt.imread(path + "\\Images\\" + im_id + '.png')
      im = transform.resize(im, (im.shape[0] // 4, im.shape[1] // 4), anti_aliasing=True)
 
      gt = plt.imread(path + '\\Masks_png\\' + im_id + '.png')
      gt = transform.resize(gt, (gt.shape[0] // 4, gt.shape[1] // 4), anti_aliasing=False) #Setting it to True creates values that are not 0 or 1

      return im, gt

    #max_height Takes mask of the image, return maximum height and rotated mask
    def max_height(self, img_msk):
  
      pixels_in_col = np.sum(img_msk, axis=0)

      #finding largest width
      max_pixels_in_col = np.max(pixels_in_col)

      #Rotates mask by 45 degrees until finds largest width
      for i in range(1,8):

          rot_mask = transform.rotate(img_msk, 45*i)
          height = np.max(np.sum(rot_mask, axis=0))
          if height > max_pixels_in_col:
              max_pixels_in_col = height
              final_msk = rot_mask

      return final_msk, max_pixels_in_col
    
    #perimeter takes image mask and returns moles perimeter
    def perimeter(self, img_msk):
        #brush saves this shape:
        #[[0 0 1 0 0]
         #[0 1 1 1 0]
         #[1 1 1 1 1]
         #[0 1 1 1 0]
         #[0 0 1 0 0]]
        brush = morphology.disk(2)

        mask_cleaned = morphology.binary_erosion(img_msk, brush)

        perimeter_im = img_msk - mask_cleaned

        return perimeter_im

    #show_perimeter - example of function that just shows smthing
    def show_per(self):
        

        plt.imshow(self.perim, cmap='gray')
        plt.show()
    

    def color_regions(self):
        segments_slic = slic(self.img, n_segments=10, compactness=3, sigma=3, start_label=1)
        fig, ax = plt.subplots(1, 2, figsize=(10, 10), sharex=True, sharey=True)

        ax[0].imshow(self.img)
        ax[0].set_title("Original")

        ax[1].imshow(mark_boundaries(self.img, segments_slic))
        ax[1].set_title('SLIC')

        print("here1")
        plt.tight_layout()
        plt.show()
        print("here2")