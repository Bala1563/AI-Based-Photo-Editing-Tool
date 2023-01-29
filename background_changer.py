

import pixellib
from pixellib.tune_bg import alter_bg
import cv2
import matplotlib.pyplot as plt
from PIL import Image

change_bg = alter_bg()
change_bg.load_pascalvoc_model("deeplabv3_xception_tf_dim_ordering_tf_kernels.h5")

i=int(input("choose the 1 for background change to image and 2 for background colour change:"))
if(i==1):
    input1=input("Enter the back ground image number:")
    input1=input1+'.jpg'
    output = change_bg.change_bg_img(f_image_path = "photo.jpeg",b_image_path = input1)
    cv2.imwrite("img.jpg", output)
if(i==2):
    output = change_bg.color_bg("group.jpg", colors = (255, 255, 255))
    cv2.imwrite("img.jpg", output)
    
plt.figure(figsize=(80,80))
plt.subplot(1,1,1)
img=cv2.imread("img.jpg")
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.axis("off")

