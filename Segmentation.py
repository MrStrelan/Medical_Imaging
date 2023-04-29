import numpy as np
import matplotlib.pyplot as plt
from skimage import transform, morphology


def prepare_im(im_id):

  path = '.'

  im = plt.imread(path + "\\Images\\" + im_id + '.png')
  im = transform.resize(im, (im.shape[0] // 4, im.shape[1] // 4), anti_aliasing=True)
 
  gt = plt.imread(path + '\\Masks_png\\' + im_id + '.png')
  gt = transform.resize(gt, (gt.shape[0] // 4, gt.shape[1] // 4), anti_aliasing=False) #Setting it to True creates values that are not 0 or 1


  return im, gt


# I do not call the masks ground truth here, because you can also measure features based on your own masks
def notmaskcreation(img):
  my_img, img_msk = prepare_im(img)


  plt.imshow(img_msk, cmap='gray')

  #Total size of the image
  area_img = img_msk.shape[0] * img_msk.shape[1] 

  #Size of mask only
  area = np.sum(img_msk)

  #Area of mole as percentage
  print("Cancer percentage:", area/area_img*100)
  
  img_msk, height = max_height(img_msk)
  print("Max height:", height)

  #Making mask without perimeter
  print(perimeter(img_msk))

  return


#max_height Takes mask of the image, return maximum height and rotated mask
def max_height(img_msk):
  
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
    
def perimeter(img_msk):
    #brush saves this shape:
    #[[0 0 0 1 0 0 0]
    # [0 1 1 1 1 1 0]
    # [0 1 1 1 1 1 0]
    # [1 1 1 1 1 1 1]
    # [0 1 1 1 1 1 0]
    # [0 1 1 1 1 1 0]
    # [0 0 0 1 0 0 0]]
    brush = morphology.disk(1)

    mask_cleaned = morphology.binary_erosion(img_msk, brush)

    perimeter_im = img_msk - mask_cleaned

    return perimeter_im

if __name__ == "__main__":
  notmaskcreation('PAT_92_141_551')
  plt.show()
  print("Done!")