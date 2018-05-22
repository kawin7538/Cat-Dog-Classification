# -*- coding: utf-8 -*-
"""
Created on Sun May 20 09:32:46 2018

@author: Kawin-PC
"""

#Part 3 makeing new prediction
from keras.preprocessing import image
import numpy as np
from keras.models import load_model
import cv2

img_size=150;
target_size=(img_size,img_size)

model=load_model('model.h5')
cv2.namedWindow("image",cv2.WINDOW_AUTOSIZE)
cv2.moveWindow("image",0,0)
cv2.namedWindow("text",cv2.WINDOW_AUTOSIZE)
cv2.moveWindow("text",500,100)
for i in range(141,151):
    test_image=image.load_img('test/'+str(i)+'.jpg',target_size=target_size)
    test_image=image.img_to_array(test_image)
    test_image=np.expand_dims(test_image,axis=0)
    result=model.predict(test_image)
    result_text=""
    if result[0][0]==1:
        result_text="dog"
    else:
        result_text="cat"
    image_cv=cv2.imread('test/'+str(i)+'.jpg')
    blank_image=np.zeros((500,500,3),np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(blank_image,result_text,(150,250), font, 4,(255,255,255),2,cv2.LINE_AA)
    cv2.imshow("image",image_cv)
    cv2.imshow("text",blank_image)
    cv2.waitKey(0)
    test_image=""
cv2.destroyAllWindows()

    


