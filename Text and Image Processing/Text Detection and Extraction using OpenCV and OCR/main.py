#1 - import libraries
import cv2
import matplotlib.pyplot as plt
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

import os
os.makedirs('outputs', exist_ok = 'True')

#2 - Loading Image
image_path = 'input_image.jpg'
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#3 - converting image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#4 - displayin original image and grayscale
plt.imshow(image_rgb)
plt.title('Original Image')
plt.axis('off')
plt.savefig('outputs/1_original_image.jpg', dpi = 300, bbox_inches = 'tight')
plt.show()

plt.imshow(gray)
plt.title('Grayscale Image')
plt.axis('off')
plt.savefig('outputs/2_grayscale_image.jpg', dpi = 300, bbox_inches = 'tight')
plt.show()

#5 - extract text from image
extracted_text = pytesseract.image_to_string(image_rgb)
extracted_text = ' '.join(extracted_text.split())
print('Extracted text:',extracted_text)


#6 - drawing bounding boxes around text
data = pytesseract.image_to_data(image_rgb, output_type = pytesseract.Output.DICT)
n_boxes = len(data['level'])
for i in range(n_boxes):
    (x,y,w,h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
    cv2.rectangle(image_rgb, (x, y), (x + w, y + h), (255, 0, 0), 2)

#7 - showing image with bounding boxes
plt.imshow(image_rgb)
plt.title('Image with text bounding boxes')
plt.axis('off')
plt.savefig('outputs/3_bounding_boxes_text.jpg', dpi = 300, bbox_inches = 'tight')
plt.show()

