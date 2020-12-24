import cv2
import tensorflow as tf 

CATEGORIES = ["10.000","20.000"]

model = tf.keras.models.load_model("MoneyDetect2.0.model")

webCam = cv2.VideoCapture(0)
webCam.set(10,100)

while True:
    success_frame_read, frame = webCam.read()

    gray_scale = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    img_resize = cv2.resize(gray_scale, (300, 160), 1)
    detect_img = img_resize.reshape(-1, 300, 160, 1)

    prediction = model.predict(detect_img)
    # coor = model.detectMultiScaleI(detect_img)

    name = CATEGORIES[int(prediction[0][0])]
    
    # for (x, y, w, h) in coor:
    #     cv2.rectangle(frame,(x, y),(x+w, y+h),(0,255,0), 2)
    scv2.putText(gray_scale,name,(10,20),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,0),2)

    cv2.imshow("Video",gray_scale)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()

    