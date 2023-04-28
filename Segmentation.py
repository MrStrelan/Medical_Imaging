# Get repo, prepare some images

#if False:      
#  !rm -rf fyp2022-imaging
#  !git clone https://github.com/vcheplygina/fyp2022-imaging.git


import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
import os
#print(os.listdir('Images'))
# Function to get us some example images and their masks, and resize them 

def prepare_im(im_id):

  path = '.\\Medical_Imaging'

  im = plt.imread(path + "\\Images\\" + im_id + '.png')
  im = resize(im, (im.shape[0] // 4, im.shape[1] // 4), anti_aliasing=True)
 
  gt = plt.imread(path + '.\\Masks_png\\' + im_id + '.png')
  gt = resize(gt, (gt.shape[0] // 4, gt.shape[1] // 4), anti_aliasing=False) #Setting it to True creates values that are not 0 or 1


  return im, gt


# I do not call the masks ground truth here, because you can also measure features based on your own masks
def notmaskcreation(im):
  im1, mask1 = prepare_im(im)
#  im2, mask2 = prepare_im('PAT_92_141_551')

  plt.imshow(mask1, cmap='gray')
  #plt.imshow(mask2, cmap='gray')

  #Total size of the image
  total = mask1.shape[0] * mask1.shape[1] 

  #Size of mask only
  area = np.sum(mask1)

  #As percentage
  print("Cancer percentage is" , area/total*100)

  # Measure diameter of the lesion: measure height or width of the mask

  #How many 1's in each column of the image (sum over axis 0, i.e. rows)
  pixels_in_col = np.sum(mask1, axis=0)
  #print(mask1.shape)
  #print(pixels_in_col.shape)


  #Without this there are some non zeros and ones still because of the resizing
  pixels2 = pixels_in_col > 0
  pixels2 = pixels2.astype(np.int8)

  #print(pixels2)

  max_pixels_in_col = np.max(pixels_in_col)
  print('height is', max_pixels_in_col)

  # pixels_in_row = 


  # Measuring the diameter at an angle

  # General rule of thumb - create a simpler image first, then do the measurement

  from skimage import transform
  rot_im = transform.rotate(mask1, 45)
  plt.imshow(rot_im, cmap='gray')

  from skimage import morphology

  #Structural element, that we will use as a "brush" on our mask
  struct_el = morphology.disk(3)

  # Use this "brush" to erode the image - eat away at the borders
  mask_eroded = morphology.binary_erosion(mask1, struct_el)

  # Show side by side (depending on brush size, you might not see a difference visually)
  fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(5, 3))
  axes[0].imshow(mask1, cmap='gray')
  axes[1].imshow(mask_eroded, cmap='gray') #This one has accidental pixels removed. Consider using it
  fig.tight_layout()

  # Verify the new mask is smaller
  #print(area)
  #print(np.sum(mask_eroded))

  # Now we can find the pixels that have value 1 in the original mask but not in the eroded mask

  perimeter_im = mask1 - mask_eroded

  plt.imshow(perimeter_im, cmap='gray')
  print("Pixels of perimiter:", np.sum(perimeter_im))



# Now we can measure as before by counting pixels in rows/columns...
  return

if __name__ == "__main__":
    im2mask('PAT_92_141_551')
    plt.show()
    print("Done!")