from flask import Flask, render_template, flash, request, session
import warnings
import os
import mysql.connector

app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'


@app.route("/")
def homepage():
    return render_template('index.html')


@app.route("/Prediction")
def Prediction():
    return render_template('Prediction.html')


@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        import tensorflow as tf
        import numpy as np
        import cv2
        from keras.preprocessing import image
        file = request.files['file']
        file.save('static/upload/Test.jpg')
        org = 'static/upload/Test.jpg'

        img1 = cv2.imread('static/upload/Test.jpg')

        dst = cv2.fastNlMeansDenoisingColored(img1, None, 10, 10, 7, 21)
        noi = 'static/upload/noi.jpg'
        cv2.imwrite(noi, dst)

        classifierLoad = tf.keras.models.load_model('Vggmodel.h5')
        test_image = image.load_img('static/upload/Test.jpg', target_size=(100, 100))
        test_image = np.expand_dims(test_image, axis=0)
        result = classifierLoad.predict(test_image)
        print(result)

        result = classifierLoad.predict(test_image)
        print(result)
        result = np.argmax(result, axis=1)

        print(result)

        out = ''
        pre = ''
        if result[0] == 0:
            print("cardboard")
            out = "cardboard"
            pre = "Degradable"
        elif result[0] == 1:
            print("glass")
            out = "glass"
            pre = "Non-Degradable"
        elif result[0] == 2:
            print("metal")
            out = "metal"
            pre = "Non-Degradable"
        elif result[0] == 3:
            print("paper")
            out = "paper"
            pre = "Degradable"

        elif result[0] == 4:
            print("plastic")
            out = "plastic"
            pre = "Non-Degradable"
        elif result[0] == 5:
            print("trash")
            out = "trash"
            pre = "Degradable"

        sendmsg("9486365535",'Prediction Result : ' + out + ' ' + pre)

        return render_template('Result.html', res=out, pre=pre, org=org, noi=noi)


def sendmsg(targetno, message):
    import requests
    requests.post(
        "http://sms.creativepoint.in/api/push.json?apikey=6555c521622c1&route=transsms&sender=FSSMSS&mobileno=" + str(
            targetno) + "&text=Dear customer your msg is " + message + "  Sent By FSMSG FSSMSS")


@app.route("/Camera")
def Camera():
    import warnings
    import cv2
    import numpy as np
    import os
    import time
    args = {"confidence": 0.5, "threshold": 0.3}
    flag = False

    labelsPath = "./yolo-coco/coco.names"
    LABELS = open(labelsPath).read().strip().split("\n")
    final_classes = ['bottle', 'wine glass', 'cup', 'cell phone', 'book']

    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
                               dtype="uint8")

    weightsPath = os.path.abspath("./yolo-coco/yolov3-tiny.weights")
    configPath = os.path.abspath("./yolo-coco/yolov3-tiny.cfg")

    # print(configPath, "\n", weightsPath)

    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    vs = cv2.VideoCapture(0)
    writer = None
    (W, H) = (None, None)

    flag = True

    flagg = 0

    while True:
        # read the next frame from the file
        (grabbed, frame) = vs.read()

        # if the frame was not grabbed, then we have reached the end
        # of the stream
        if not grabbed:
            break

        # if the frame dimensions are empty, grab them
        if W is None or H is None:
            (H, W) = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                     swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        layerOutputs = net.forward(ln)
        end = time.time()

        # initialize our lists of detected bounding boxes, confidences,
        # and class IDs, respectively
        boxes = []
        confidences = []
        classIDs = []

        # loop over each of the layer outputs
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability)
                # of the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > args["confidence"]:
                    # scale the bounding box coordinates back relative to
                    # the size of the image, keeping in mind that YOLO
                    # actually returns the center (x, y)-coordinates of
                    # the bounding box followed by the boxes' width and
                    # height
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    # use the center (x, y)-coordinates to derive the top
                    # and and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # update our list of bounding box coordinates,
                    # confidences, and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        # apply non-maxima suppression to suppress weak, overlapping
        # bounding boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
                                args["threshold"])

        # ensure at least one detection exists
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                if (LABELS[classIDs[i]] in final_classes):

                    flagg += 1
                    # print(flag)
                    # sendmsg("9486365535", " Animal Detected " + LABELS[classIDs[i]])

                    color = [int(c) for c in COLORS[classIDs[i]]]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    text = "{}: {:.4f}".format(LABELS[classIDs[i]],
                                               confidences[i])
                    cv2.putText(frame, text, (x, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    if (flagg == 40):
                        flagg = 0

                        out = LABELS[classIDs[i]]
                        pre = ''
                        if out=="bottle":
                            pre = "Non-Degradable"
                        elif out=="wine glass":
                            pre = "Non-Degradable"
                        elif out=="cup":
                            pre = "Degradable"
                        elif out == "cell phone":
                            pre = "Non-Degradable"
                        elif out == "book":
                            pre = "Degradable"

                        sendmsg("9486365535",'Prediction Result : ' + out + ' ' + pre)





        else:
            flag = True

        cv2.imshow("Output", frame)

        if cv2.waitKey(1) == ord('q'):
            break

    # release the webcam and destroy all active windows
    vs.release()
    cv2.destroyAllWindows()


    return render_template('index.html')


if __name__ == '__main__':
    # app.run(host='0.0.0.0',debug = True, port = 5000)
    app.run(debug=True, use_reloader=True)
