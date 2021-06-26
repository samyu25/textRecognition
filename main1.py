import cv2 # For OpenCV modules (Morphology and Contour Finding)
import numpy as np # For general purpose array manipulation
from PIL import Image
from pdf2image import convert_from_path

images = convert_from_path('img.pdf')

for i, image in enumerate(images):
    fname = "image" + str(i) + ".jpeg"

    image.save(fname, "JPEG")

# Load in image
i = Image.open('image0.jpeg')
images = i.resize((400,400))
print(images.size)
images.save('images.jpeg')
img = cv2.imread('images.jpeg',0)

# Create a new image that pads each side by 20 pixels
# This will allow the outlining of the text to be tighter
# Create an image of all white
imgPad = 255*np.ones((img.shape[0] + 40, img.shape[1] + 40),
    dtype='uint8')

# Place the original image in the middle
imgPad[20:imgPad.shape[0]-20, 20:imgPad.shape[1]-20] = img

# Invert image
# White becomes black and black becomes white
imgBW = imgPad < 128
imgBW = 255*imgBW.astype('uint8')

# Find all of the contours in the image
contours,hierarchy = cv2.findContours(imgBW.copy(), cv2.RETR_LIST,
    cv2.CHAIN_APPROX_NONE)

# New image that places square blocks over all letters
imgBlocks = np.zeros(imgBW.shape, dtype='uint8')

# For each contour...
for idx in range(len(contours)):
    # Reshape each contour into a 2D array
    # First co-ordinate is the column, second is the row
    cnt = np.reshape(contours[idx], (contours[idx].shape[0],
        contours[idx].shape[2]))
    # Transpose to allow for max and min calls
    cnt = cnt.T

    # Find the max and min of each contour
    maxCol = np.max(cnt[0])
    minCol = np.min(cnt[0])
    maxRow = np.max(cnt[1])
    minRow = np.min(cnt[1])

    # Use the previous to fill in a minimum spanning bounding
    # box around each contour
    for row in np.arange(minRow, maxRow+1):
        for col in np.arange(minCol, maxCol+1):
            imgBlocks[row,col] = 255

# Morphological closing on the image with a 20 x 20 structuring element
structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (20,20))
imgClose = cv2.morphologyEx(imgBlocks.copy(), cv2.MORPH_CLOSE, structuringElement)

# Find the contour of this structure
contoursFinal,hierarchyFinal = cv2.findContours(imgClose.copy(), cv2.RETR_LIST,
    cv2.CHAIN_APPROX_NONE)

# Take the padded image, and draw a red boundary around the font
# First, create a colour image
imgPadColour = np.dstack((imgPad, imgPad, imgPad))

# Reshape the contour points like we did before
cnt = np.reshape(contoursFinal[0], (contoursFinal[0].shape[0],
    contoursFinal[0].shape[2]))

# Careful - pixels are packed in BGR format
# As such, for each point in the outer shape, set to red
for (col,row) in cnt:
    imgPadColour[row][col][0] = 0
    imgPadColour[row][col][1] = 0
    imgPadColour[row][col][2] = 255

# Crop out image for final one
imgFinal = imgPadColour[20:imgPad.shape[0]-20, 20:imgPad.shape[1]-20]

# Show both images
cv2.imshow('Original Image', img)
cv2.imshow('Image with red outline over text', imgFinal)
cv2.waitKey(0)
cv2.destroyAllWindows()
