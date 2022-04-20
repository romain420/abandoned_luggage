import cv2
import numpy as np


def meancolor(img,x_min,y_min,x_max,y_max):
    x =img.shape[0]
    y = img.shape[1]
    if (x_min <= 0):
        x_min = 0
    if (x_max >= x):
        x_max = x-1
    if (y_min <= 0):
        y_min = 0
    if (y_max <= y):
        y_max = y-1
    img_rect = img[y_min:y_max,x_min:x_max]
    try:
        img_rect = cv2.blur(img_rect,(20,20))
    except cv2.error:
        pass
    G,B,R = cv2.split(img_rect)
    meanG = np.mean(G)
    meanR = np.mean(R)
    meanB = np.mean(B)
    return meanG,meanB,meanR

def color_score(img,x_min,y_min,x_max,y_max):
    meanG,meanB,meanR = meancolor(img,x_min,y_min,x_max,y_max)
    scoreG = np.sqrt(meanG**2 + meanG**2)/255
    scoreR = np.sqrt(meanR**2 + meanR**2)/255
    scoreB = np.sqrt(meanB**2 + meanB**2)/255
    score = (scoreG+scoreB+scoreR)/3
    return score


# def main():
#     cap=cv2.VideoCapture(0)
#     cv2.namedWindow('Camera')
#     color=90
#     S=50
#     V=50
#     rangecolor = 15
#     while True:
#         ret, img=cap.read()
#         hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
       
#         # lower bound and upper bound 
#         lower_bound=np.array([color-rangecolor, S, V])
#         upper_bound=np.array([color+rangecolor, 255,255])
        
#         # find the colors within the boundaries
#         mask = cv2.inRange(hsv, lower_bound, upper_bound)
#         mask=cv2.erode(mask, None, iterations=2)
#         mask=cv2.dilate(mask, None, iterations=5)
#         #define kernel size  
#         kernel = np.ones((7,7),np.uint8)

#         # Remove unnecessary noise from mask
#         mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
#         mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

#         # Segment only the detected region
#         segmented_img = cv2.bitwise_and(img, img, mask=mask)
                
#         # Find contours from the mask
#         contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         if len(contours) > 0:
#             c=max(contours, key=cv2.contourArea)
#             ((x, y), radius)=cv2.minEnclosingCircle(c)
#             meancolor(img,int(x-radius),int(x+radius),int(y-radius),int(y+radius))
#         output = cv2.drawContours(np.copy(img), contours, -1, (0, 0, 255), 3)

#         # Showing the output
#         cv2.putText(output, "[Souris]Couleur: {:d}    [o|l] S:{:d}    [p|m] V{:d}".format(color, S, V), (5, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255,0, 0), 1, cv2.LINE_AA)
#         cv2.imshow('Camera', img)
#         cv2.imshow('mask',segmented_img)
#         cv2.imshow("Output", output)
#         if cv2.waitKey(1)&0xFF==ord('q'):
#             break
     
#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ =='__main__':
#     main()