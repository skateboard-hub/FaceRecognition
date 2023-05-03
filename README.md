# Assignment - Live Video Streaming Service with Facial Recognition.

Name: Shunyi Xu

## Setup requirements.
+ pip3 install opencv-python
+ pip3 install Flask
+ pip3 install numpy
+ pip3 install socketio
+ pip3 install Flask-SocketIO
+ pip3 install eventlet
+ pip3 install waitress
+ pip3 install cmake
+ pip3 install dlib
+ pip3 install face_recognition
+ Dwonload C++ complier

## Modification
+ Compress the frames (to about 25% of the original size) 
+ Skip some frames. Do the detection/recognition part only on fixed interval.

## Technology
+ Primary programming language: Python 3.9 
+ Web application framework: Flask 
+ Real-time communication between front-end and back-end: Flask-SocketIO and SocketIO 
+ Web server for Flask-SocketIO: Eventlet 
+ Mathematical computing: NumPy 
+ Image processing and computer vision tasks: OpenCV 
+ Web server for production environment: Waitress 
+ Face recognition: face_recognition library