from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import matplotlib
matplotlib.use("QtAgg")
import os
import random

from skimage.filters import gaussian
from skimage.segmentation import active_contour


folderPNG = ".\\Medical_Imaging\\Images\\" #folder containing the png images
folderJPG = ".\\Medical_Imaging\\ImagesJPG\\" #folder containing the jpg images

def pngtojpeg(): #converts all pngs in a folder to jpgs

    for filename in os.listdir(folderPNG):
        png = Image.open(folderPNG + filename)
        png.load() # required for png.split

        background = Image.new("RGB", png.size, (255, 255, 255))
        background.paste(png, mask=png.split()[3]) # 3 is the alpha channel 

        filename = filename[:-4] #removes the .png from the filename
        background.save(folderJPG + filename + '.jpg', 'JPEG', quality=80)

    return print("All pngs in the folder have been converted to jpgs")


def randpicture(folder): #for the sake of testing other photos with ease
    filename = random.choice(os.listdir(folder))
    return filename


im = rgb2gray(Image.open(".\\Medical_Imaging\\ImagesJPG\\" + randpicture(folderJPG)))

def snaking(im):
    # Resize for speed and bloor to blend the collor
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

    from skimage.draw import polygon

    im = rgb2gray(Image.open(".\\Medical_Imaging\\ImagesJPG\\" + randpicture(folderJPG)))
    im, snake = snaking(im)

  

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
