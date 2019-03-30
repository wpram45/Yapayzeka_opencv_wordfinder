# Import required modules
import cv2 as cv
import math
import argparse
import pytesseract
import time






parser = argparse.ArgumentParser(description='Bu yazılım Opencv ile  yazıları tanımlar.')
# Giriş argümanı
parser.add_argument('--input', help='Kameradan görüntü almak için bu parametreyi göndermeyin.')


parser.add_argument("--ciz", default=None,help="kelimeleri cevresine dikdörtgen çizilmesi için true gönderin")

# Model argümanı
parser.add_argument('--model', default="frozen_east_text_detection.pb",
                    help='pb dosyası ile ağırlık-weight değiştirerek öğrenir'
                    )
# Genişlik argümanı
parser.add_argument('--width', type=int, default=128,
                    help='Ön resim işleme için Genişlik belirtmek için. 32 nin olmalıdır.'
                   )
# Yükseklik argümanı
parser.add_argument('--height',type=int, default=128,
                    help='Ön resim işleme için Yükseklik belirle 32 nin katı olmalıdır.'
                   )
# Güven Eşiği
parser.add_argument('--thr',type=float, default=0.5,
                    help='Güven Eşiği.'
                   )
# Maksimum olmayan bastırma eşiği
parser.add_argument('--nms',type=float, default=0.4,
                    help='Maksimum olmayan bastırma eşiği.'
                   )

args = parser.parse_args()


############ Utility functions ############
def decode(scores, geometry, scoreThresh):
    
    detections = []
    confidences = []

    ############ CHECK DIMENSIONS AND SHAPES OF geometry AND scores ############
    assert len(scores.shape) == 4, "Incorrect dimensions of scores"
    assert len(geometry.shape) == 4, "Incorrect dimensions of geometry"
    assert scores.shape[0] == 1, "Invalid dimensions of scores"
    assert geometry.shape[0] == 1, "Invalid dimensions of geometry"
    assert scores.shape[1] == 1, "Invalid dimensions of scores"
    assert geometry.shape[1] == 5, "Invalid dimensions of geometry"
    assert scores.shape[2] == geometry.shape[2], "Invalid dimensions of scores and geometry"
    assert scores.shape[3] == geometry.shape[3], "Invalid dimensions of scores and geometry"
    height = scores.shape[2]
    width = scores.shape[3]
    for y in range(0, height):

        # Extract data from scores
        scoresData = scores[0][0][y]
        x0_data = geometry[0][0][y]
        x1_data = geometry[0][1][y]
        x2_data = geometry[0][2][y]
        x3_data = geometry[0][3][y]
        anglesData = geometry[0][4][y]
        for x in range(0, width):
            score = scoresData[x]

            # If score is lower than threshold score, move to next x
            if(score < scoreThresh):
                continue

            # Calculate offset
            offsetX = x * 4.0
            offsetY = y * 4.0
            angle = anglesData[x]

            # Calculate cos and sin of angle
            cosA = math.cos(angle)
            sinA = math.sin(angle)
            h = x0_data[x] + x2_data[x]
            w = x1_data[x] + x3_data[x]

            # Calculate offset
            offset = ([offsetX + cosA * x1_data[x] + sinA * x2_data[x], offsetY - sinA * x1_data[x] + cosA * x2_data[x]])

            # Find points for rectangle
            p1 = (-sinA * h + offset[0], -cosA * h + offset[1])
            p3 = (-cosA * w + offset[0],  sinA * w + offset[1])
            center = (0.5*(p1[0]+p3[0]), 0.5*(p1[1]+p3[1]))
            detections.append((center, (w,h), -1*angle * 180.0 / math.pi))
            confidences.append(float(score))

    # Return detections and confidences
    return [detections, confidences]

if __name__ == "__main__":
    config = ('-l eng --oem 1 --psm 3')
    #kelimeler sözlüğü
    kelimeler={}
    #buluan kelimenin rect olarak image nesnesi - kesilmiş resim:
    new_img=None
    x1=0
    x2=0
    y1=0
    y2=0


    # Read and store arguments
    confThreshold = args.thr
    nmsThreshold = args.nms
    inpWidth = args.width
    inpHeight = args.height
    model = args.model
    #kelimelerin etrafına çizim için
    cizim=False
    cizim=args.ciz

    completed=False
   # print(cizim)
    # Load network
    if cizim!=None:
        net = cv.dnn.readNet(model)

    # Create a new named window
    kWinName = "Yapay zeka destekli görüntüden kelime bulucu"
    cv.namedWindow(kWinName, cv.WINDOW_NORMAL)
    outputLayers = []
    outputLayers.append("feature_fusion/Conv_7/Sigmoid")
    outputLayers.append("feature_fusion/concat_3")

    # Open a video file or an image file or a camera stream
    cap = cv.VideoCapture(args.input if args.input else 0)

    while completed==False and cv.waitKey(1) < 0:

        #kelimeler sözlügü uzunlugu 250 gectiyse tamamla

        if len(kelimeler)>250:
            completed=True
            break

        # Read frame
        hasFrame, frame = cap.read()
        if not hasFrame:
            cv.waitKey()
            break
        if cizim!=None:
        # Get frame height and width
            height_ = frame.shape[0]
            width_ = frame.shape[1]
            rW = width_ / float(inpWidth)
            rH = height_ / float(inpHeight)

        # Create a 4D blob from frame.
            blob = cv.dnn.blobFromImage(frame, 1.0, (inpWidth, inpHeight), (123.68, 116.78, 103.94), True, False)

        # Run the model
            net.setInput(blob)
            output = net.forward(outputLayers)
            t, _ = net.getPerfProfile()
        #label = 'Anlam cikarma zamani: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())

        # Get scores and geometry
            scores = output[0]
            geometry = output[1]


            [boxes, confidences] = decode(scores, geometry, confThreshold)
        # Apply NMS
            indices = cv.dnn.NMSBoxesRotated(boxes, confidences, confThreshold,nmsThreshold)
        # print(indices)
            for i in indices:
            # get 4 corners of the rotated rect
                vertices = cv.boxPoints(boxes[i[0]])
            # scale the bounding box coordinates based on the respective ratios
                for j in range(4):
                    vertices[j][0] *= rW
                    vertices[j][1] *= rH
                for j in range(4):
                    p1 = (vertices[j][0], vertices[j][1])
                    p2 = (vertices[(j + 1) % 4][0], vertices[(j + 1) % 4][1])
                    cv.line(frame, p1, p2, (0, 255, 0), 2, cv.LINE_AA);





               # print(vertices)

                    x1=int(p1[0])
                    x2=int(p1[1])
                    y1=int(p2[0])
                    y2=int(p2[1])
             #   print(x1,x2,y1,y2)
            #   #cv.putText(frame, "{:.3f}".format(confidences[i[0]]), (vertices[0][0], vertices[0][1]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv.LINE_AA)

        # Put efficiency information


        #anlam cıkarma zamanı
      #  cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))


        if x1<=0 or x2<=0 or y1<=0 or y2<=0:

            crop_img=frame
        else:
           # rect_img=frame[x1:x2,y2:y1]
            #adjust camera

              crop_img=frame[50:250+abs(x1),50:250+abs(y1)]
             #crop_img=frame[abs(x1):+abs(x1)+abs(x2)+50,abs(y1):abs(y2)+abs(y1)+10]


                  #diske anlık resmi yaz
        cv.imwrite("anlık_resim.png",crop_img)

                   #diskten anlık resmi oku
        img = cv.imread('anlık_resim.png',0)

                #text olarak resimden kelimeleri çek
        text = pytesseract.image_to_string(img, config=config)
        if text!="":
            for word in text.split():
                if word not in kelimeler:
                    kelimeler[word.lower()]=1
                else:
                    kelimeler[word.lower()]+=1
           # cv.imwrite("anlık_resim.png",rect_img)
        else:
            crop_img=frame
        cv.imshow(kWinName,crop_img)

        print(kelimeler)
    aranan_kelime=input("Aranılacak kelimeyi giriniz :")

    if aranan_kelime in kelimeler:
        print("kelime bulundu kelime",kelimeler[aranan_kelime],"defa vardır")
    else:
        print("Kelime bulunamadı")






















