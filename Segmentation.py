# Get repo, prepare some images

#if False:      
#  !rm -rf fyp2022-imaging
#  !git clone https://github.com/vcheplygina/fyp2022-imaging.git


import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize

# Function to get us some example images and their masks, and resize them 
def prepare_im(im_id):

  path = 'Images/'

  im = plt.imread(path + 'example_image/' + im_id + '.jpg')
  im = resize(im, (im.shape[0] // 4, im.shape[1] // 4), anti_aliasing=True)
 
  gt = plt.imread(path + 'example_segmentation/' + im_id + '_segmentation.png')
  gt = resize(gt, (gt.shape[0] // 4, gt.shape[1] // 4), anti_aliasing=False) #Setting it to True creates values that are not 0 or 1


  return im, gt


# I do not call the masks ground truth here, because you can also measure features based on your own masks

im1, mask1 = prepare_im('ISIC_0001871')
im2, mask2 = prepare_im('ISIC_0012151')

plt.imshow(mask1, cmap='gray')


plt.imshow(mask1, cmap='gray')

#Total size of the image
total = mask1.shape[0] * mask1.shape[1] 

#Size of mask only
area = np.sum(mask1)

#As percentage
print(area/total*100)

# Measure diameter of the lesion: measure height or width of the mask

#How many 1's in each column of the image (sum over axis 0, i.e. rows)
pixels_in_col = np.sum(mask1, axis=0)
print(mask1.shape)
print(pixels_in_col.shape)


#Without this there are some non zeros and ones still because of the resizing
pixels2 = pixels_in_col > 0
pixels2 = pixels2.astype(np.int8)

print(pixels2)

max_pixels_in_col = np.max(pixels_in_col)
print('height is', max_pixels_in_col)

# pixels_in_row = 


# Measuring the diameter at an angle

# General rule of thumb - create a simpler image first, then do the measurement

from skimage import transform
rot_im = transform.rotate(mask1, 45)
plt.imshow(rot_im, cmap='gray')

# Now we can measure as before by counting pixels in rows/columns...