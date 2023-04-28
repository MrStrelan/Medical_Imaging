from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import matplotlib
matplotlib.use("QtAgg")

from skimage.filters import gaussian
from skimage.segmentation import active_contour




"""

THIS FILE IS FOR TESTING PURPOSES ONLY

"""




filename = "PAT_90_219_648"

png = Image.open(".\\Medical_Imaging\\Images\\" + filename + ".png")
png.load() # required for png.split

background = Image.new("RGB", png.size, (255, 255, 255))
background.paste(png, mask=png.split()[3]) # 3 is the alpha channel 

background.save('.\\Medical_Imaging\\ImagesJPG\\' + filename + '.jpg', 'JPEG', quality=80)

im = rgb2gray(Image.open(".\\Medical_Imaging\\ImagesJPG\\" + filename + ".jpg"))

def snaking(im):
    # Resize for speed
    im = gaussian(im, 3)
    
    # Find the darkest spot in the image
    r, c = np.unravel_index(np.argmin(im), im.shape)
    
    # Create the initial snake as a circle centered on the darkest spot
    s = np.linspace(0, 2*np.pi, 200) # Number of points on the circle
    radius = 50
    r_circle = r + radius*np.sin(s)
    c_circle = c + radius*np.cos(s)
    init = np.array([r_circle, c_circle]).T
    
    # Run active contour segmentation, the snake will be an array of the same shape as init
    snake = active_contour(im, init, alpha=0.015, beta=10, gamma=0.001, w_line=-5)
    
    # Show
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(im, cmap=plt.cm.gray)
    ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3)
    plt.show()
    
    return im, snake



def masking():
    im = rgb2gray(Image.open(".\\Medical_Imaging\\ImagesJPG\\" + filename + ".jpg"))
    im, snake = snaking(im)

    from skimage.draw import polygon

    # Create an empty image to add the mask to
    mask = np.zeros_like(im)

    # Find coordinates inside the polygon defined by the snake
    rr, cc = polygon(snake[:, 0], snake[:, 1], im.shape)

    # This is the foreground class
    mask[rr, cc] = 1

    plt.imshow(mask, cmap="gray")
    plt.show()

    return



masking()
