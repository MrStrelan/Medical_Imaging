#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[12]:





# In[82]:


#Currently working on 
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import colorsys

def mask_segm(img, mask):
    # Overlay the mask on the original image
    im2 = img.copy()
    im2[mask == 0] = 0
    return im2

img_path = "/Users/tobiasmichelsen/Downloads/imgs_part_1/PAT_9_17_80.png"
mask_path = "/Users/tobiasmichelsen/Desktop/Masks_Folder1/mask_PAT_9_17_80.png"

    # Load image and mask files as NumPy arrays
img = cv2.imread(img_path)
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Load the mask as a grayscale image

# Convert the image from BGR (OpenCV default) to RGB (Matplotlib default)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Call the mask_segm function and save the result to a variable
overlayed_img = mask_segm(img, mask)

# Display the overlayed image using Matplotlib
plt.imshow(overlayed_img)
plt.show()

def plot_color_histogram(image, mask=None):
    color_channels = ('r', 'g', 'b')
    for i, color in enumerate(color_channels):
        histogram = cv2.calcHist([image], [i], mask, [256], [0, 256])
        plt.plot(histogram, color=color)
        plt.xlim([0, 256])
        plt.xlabel('Color intensity')
        plt.ylabel('Frequency')
        plt.show()

# Create a mask for the non-black pixels in the overlayed_img
non_black_mask = cv2.inRange(overlayed_img, (1, 1, 1), (255, 255, 255))

# Call the modified plot_color_histogram function with the non_black_mask
plot_color_histogram(overlayed_img, non_black_mask)

#Now we want to find the corresponding HSV values as they mimic the way humans perceive color.

def find_hsv(r, g, b):
    r /= 255.0
    g /= 255.0
    b /= 255.0
    hsv = colorsys.rgb_to_hsv(r, g, b)
    return hsv
    
def extract_rgb_values(image, mask):
    # Get the indices of the non-black pixels in the mask
    non_black_indices = np.where(mask == 255)

    # Extract the RGB values using the non_black_indices
    rgb_values = image[non_black_indices]

    # Convert the extracted RGB values to HSV
    hsv_values = np.array([find_hsv(r, g, b) for r, g, b in rgb_values])

    print("These are the corresponding hsv values to the pixels' rgb values:", hsv_values)
    return hsv_values

# Extract the HSV values of the non-black pixels in the overlayed_img
hsv_values = extract_rgb_values(overlayed_img, non_black_mask)
# Initialize counters for different colors
red_count = 0
yellow_count = 0
green_count = 0
cyan_count = 0
blue_count = 0
magenta_count = 0
white_count = 0

plt.imshow(non_black_mask)
plt.imshow(overlayed_img)

print(len(hsv_values))
# Loop through all the HSV values and update the counters based on the H value
for hsv in hsv_values:
    h, s, v = hsv

    if 0.005 < h <= 0.060:
        red_count += 1
    elif 0.061 <= h <= 0.120:
        yellow_count += 1
    elif 0.121 <= h <= 0.180:
        green_count += 1
    elif 0.181 <= h <= 0.240:
        cyan_count += 1
    elif 0.241 <= h <= 0.300:
        blue_count += 0
    elif 0.301 <= h <= 0.360:
        magenta_count += 0

    if v >= 0.9:
        white_count += 1

print("Red:",red_count,"Yellow:",yellow_count,"Green:",green_count,"Cyan:",cyan_count,"Blue:",blue_count,"Magenta:", magenta_count,"White:",white_count)


# In[50]:


def find_dominant_colors(hsv_values, n_clusters=5):
    # Fit a KMeans clustering model to the HSV values
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(hsv_values)

    # Get the cluster centers (dominant colors)
    dominant_colors = kmeans.cluster_centers_

    return dominant_colors


def hsv_intervals(dominant_colors, hsv_values):
    intervals = []

    for color in dominant_colors:
        min_h, max_h = color[0], color[0]
        min_s, max_s = color[1], color[1]
        min_v, max_v = color[2], color[2]

        for hsv in hsv_values:
            min_h = min(min_h, hsv[0])
            max_h = max(max_h, hsv[0])

            min_s = min(min_s, hsv[1])
            max_s = max(max_s, hsv[1])

            min_v = min(min_v, hsv[2])
            max_v = max(max_v, hsv[2])

        intervals.append(((min_h, max_h), (min_s, max_s), (min_v, max_v)))

    return intervals


# Find the dominant colors
dominant_colors = find_dominant_colors(hsv_values)

# Calculate the HSV intervals for the dominant colors
dominant_color_intervals = hsv_intervals(dominant_colors, hsv_values)

print("Dominant Colors HSV Intervals:")
for h, s, v in dominant_color_intervals:
    print(f"H: {h}, S: {s}, V: {v}")


# In[17]:


#CURRENT WORKING 


import cv2
import matplotlib.pyplot as plt
from skimage.segmentation import slic, mark_boundaries

def mask_segm(img, mask):
    # Extract the masked region from the original image
    im2 = img.copy()
    im2[mask == 0] = 0
    return im2

img_path = "/Users/tobiasmichelsen/Downloads/imgs_part_1/PAT_9_17_80.png"
mask_path = "/Users/tobiasmichelsen/Desktop/Masks_Folder1/mask_PAT_9_17_80.png"

# Load image and mask files as NumPy arrays
img = cv2.imread(img_path)
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

# Convert the image from BGR (OpenCV default) to RGB (Matplotlib default)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Call the mask_segm function and save the result to a variable
masked_region = mask_segm(img, mask)


# Apply SLIC segmentation to the masked region
segments_slic = slic(masked_region, n_segments=10, compactness=1, sigma=1, start_label=1)

# Show the results
fig, ax = plt.subplots(1, 2, figsize=(10, 10), sharex=True, sharey=True)

ax[0].imshow(img)
ax[0].set_title("Original")

ax[1].imshow(mark_boundaries(masked_region, segments_slic))
ax[1].set_title('SLIC')

plt.tight_layout()
plt.show()


#Pseudocode: Find area of mask, then use SLIC on that area only


# In[16]:


import cv2
import matplotlib.pyplot as plt
from skimage.segmentation import slic, mark_boundaries

def mask_segm(img, mask):
    # Overlay the mask on the original image
    im2 = img.copy()
    im2[mask == 0] = 0
    return im2

img_path = "/Users/tobiasmichelsen/Downloads/imgs_part_1/PAT_9_17_80.png"
mask_path = "/Users/tobiasmichelsen/Desktop/Masks_Folder1/mask_PAT_9_17_80.png"

# Load image and mask files as NumPy arrays
img = cv2.imread(img_path)
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Load the mask as a grayscale image

# Convert the image from BGR (OpenCV default) to RGB (Matplotlib default)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Call the mask_segm function and save the result to a variable
overlayed_img = mask_segm(img, mask)

# Apply SLIC segmentation to the masked colored area
segments_slic = slic(overlayed_img, n_segments=10, compactness=1, sigma=1, start_label=1)

# Create an image with the masked region and boundaries marked
boundary_marked_img = mark_boundaries(overlayed_img, segments_slic)

# Combine the marked boundaries with the original image, but only within the mask
result = np.zeros_like(img)
result[mask > 0] = boundary_marked_img[mask > 0]

# Show the results
fig, ax = plt.subplots(1, 2, figsize=(10, 10), sharex=True, sharey=True)

ax[0].imshow(img)
ax[0].set_title("Original")

ax[1].imshow(result)
ax[1].set_title('SLIC')

plt.tight_layout()
plt.show()

