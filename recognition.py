from flask import Flask, render_template, Response, request
from flask_socketio import SocketIO
import cv2
import numpy as np
import face_recognition
import os
import math
import time


app = Flask(__name__)
socketioApp = SocketIO(app)
path="Images"
images=[]
classNames=[]
files = os.listdir(path)
for file in files:
    curImg=cv2.imread(f'{path}/{file}')
    images.append(curImg)
    classNames.append(os.path.splitext(file)[0]) 

def face_distance_to_conf(distance, face_match_threshold=0.6):
    if distance > face_match_threshold:
        range = (1.0 - face_match_threshold)
        linear_val = (1.0 - distance) / (range * 2.0)
        return linear_val
    else:
        range = face_match_threshold
        linear_val = 1.0 - (distance / (range * 2.0))
        return linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))

def findEncodings(images):
    encodeList =[]
    for img in images:
        img= cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeList= findEncodings(images)

cap = cv2.VideoCapture(0)

def frames(frame_skip):
    img_count = 0
    start_time = time.time()
    skip_count = 0
    pre_faces = []
    text = ""
    while True:
        success,img=cap.read()
        skip_count +=1
        if skip_count % frame_skip == 0:
            imgSmall=cv2.resize(img,(0,0),None,0.25,0.25)
            imgSmall= cv2.cvtColor(imgSmall,cv2.COLOR_BGR2RGB)

            facesInCurrentFrame = face_recognition.face_locations(imgSmall)
            encodingsCurrentFrame=face_recognition.face_encodings(imgSmall,facesInCurrentFrame)
            pre_faces = []
            for encodeFace,faceLocation in zip(encodingsCurrentFrame,facesInCurrentFrame):
                matches = face_recognition.compare_faces(encodeList,encodeFace)
                faceDist = face_recognition.face_distance(encodeList,encodeFace)
                matchIndex = np.argmin(faceDist)
                pre_faces.append(faceLocation)

                if matches[matchIndex]:
                    name=classNames[matchIndex]
                    matchPerc= round(face_distance_to_conf(faceDist[matchIndex])*100)
                    y1,x2,y2,x1=faceLocation
                    y1,x2,y2,x1=y1*4,x2*4,y2*4,x1*4
                    cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
                    cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
                    text = name+" "+ str(matchPerc)+"%"
                    cv2.putText(img,name+" "+ str(matchPerc)+"%",(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
                else:
                    y1,x2,y2,x1=faceLocation
                    y1,x2,y2,x1=y1*4,x2*4,y2*4,x1*4
                    cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
                    cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
                    text = "Unknown"
                    cv2.putText(img,"Unknown",(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
        else:
            for faceLocation in pre_faces:
                y1,x2,y2,x1=faceLocation
                y1,x2,y2,x1=y1*4,x2*4,y2*4,x1*4
                cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
                cv2.putText(img,text,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
        img_count += 1
        elapsed_time = time.time() - start_time
        fps = img_count / elapsed_time
        cv2.putText(img, f"FPS: {fps:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', img)
        img = buffer.tobytes()

        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n') 
        if cv2.waitKey(1) == ord('q'): 
            break

    cap.release() 
    cv2.destroyAllWindows() 
                    
@app.route('/video')
def video():
    frame_skip = request.args.get('skipFrames', default=2, type=int)
    return Response(frames(frame_skip), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

def run():
    socketioApp.run(app)

if __name__ == '__main__':
    socketioApp.run(app)