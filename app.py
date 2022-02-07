#Import necessary libraries
from flask import Flask, render_template, Response
#Initialize the Flask app
app = Flask(__name__)

import pandas as pd
import numpy as np
from cv2 import KeyPoint, cv2
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Dropout
from tensorflow.keras.callbacks import TensorBoard
import mediapipe as mp
import sys
import os
import os.path
import time
from moviepy.editor import VideoFileClip, concatenate_videoclips 
from flask import Flask,render_template,url_for,request
import speech_recognition as sr
import threading
import re
from os.path import isfile, join
from zipfile import ZipFile
from os import listdir
from stos.sign_to_speech.model_prepare import download_file
from stos.sign_to_speech.sign_to_speech import SignToSpeech
from stos.speech_to_sign.speech_to_sign import SpeechToSign

os.environ["TOKENIZERS_PARALLELISM"] = "false"

#------------------------------------------------------------------------------------------------
#download signs videos
#------------------------------------------------------------------------------------------------
def download_videos():
    path=r'static\dataset'
    download_file('https://drive.google.com/u/0/uc?id=12ERh3zdqjX3kAXJVXiIII3d8P-J544oW&export=download',os.path.join(path, 'videos_with_40_frames.zip'))
    download_file('https://drive.google.com/u/0/uc?id=1Ry1Ra7XAuNjwSqInZfhrkPVKuDuCZ11F&export=download',os.path.join(path, 'videos with words.zip'))
    #print('2 files downloaded')
    with ZipFile(os.path.join(path, 'videos_with_40_frames.zip'), 'r') as videos:
        videos.extractall(path)
        #print('File videos_with_40_frames unziped')
    with ZipFile(os.path.join(path, 'videos with words.zip'), 'r') as videos:
        videos.extractall(path)
        #print('File videos with words unziped')
    os.remove(path+r'\videos_with_40_frames.zip')
    os.remove(path+r'\videos with words.zip')
    #print('zip files deleted')

#------------------------------------------------------------------------------------------------
#Sign to Speech
#------------------------------------------------------------------------------------------------
def start_stream():

    sts = SignToSpeech(0, 20, os.path.join('model', 'cv_model.hdf5'), os.path.join('model', 'names'),
                       display_keypoint=True, display_window=False)
    for word, frame in sts.start_pipeline():
        ret, buffer = cv2.imencode('.jpg', frame)
        im = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + im + b'\r\n')
		
        


#------------------------------------------------------------------------------------------------
#Speech to Sign (Text Input)
#------------------------------------------------------------------------------------------------

def add_words_to_videos(name):
    
    video = cv2.VideoCapture(r"static\dataset\videos_with_40_frames/{}.mp4".format(name))

    # We need to check if camera
    # is opened previously or not
    if (video.isOpened() == False): 
        print("Error reading video file")

    # We need to set resolutions.
    # so, convert them from float to integer.
    frame_width = int(video.get(3))
    frame_height = int(video.get(4))

    size = (frame_width, frame_height)

    # Below VideoWriter object will create
    # a frame of above defined The output 
    # is stored in 'filename.mp4' file.
    result = cv2.VideoWriter(r'static\dataset\videos with words\{}.mp4'.format(name), 
                             cv2.VideoWriter_fourcc(*'MJPG'),
                             10, size)

    while(True):
        ret, frame = video.read()

        font = cv2.FONT_HERSHEY_SIMPLEX

        cv2.putText(frame, 
            name, 
            (450, 50), 
            font, 1, 
            (0, 0, 0), 
            2, 
            cv2.LINE_4)

        if ret == True: 

            # Write the frame into the
            # file 'filename.avi'
            result.write(frame)

        # Break the loop
        else:
            break

    video.release()
    result.release()

    # Closes all the frames
    cv2.destroyAllWindows()

    print("The video was successfully saved")
    
    return r'static\dataset\videos with words\{}.mp4'.format(name)


def showing_videos(text):
    matched_words = []  
    videos_links = []
    videos_with_words = []
    all_words = ['age', 'book', 'call', 'car', 'day', 'egypt', 'English', 'Enjoy', 'every', 'Excuse', 'football', 'Forget', 'fun', 'Good', 'hate', 'have', 'hello', 'help', 'holiday', 'Iam', 'love', 'meet', 'month', 'morning', 'my', 'NA', 'name', 'Nice', 'no', 'not', 'number', 'okay', 'picture', 'play', 'read', 'ride', 'run', 'sorry', 'speak', 'sport', 'take', 'thanks', 'time', 'today', 'understand', 'what', 'when', 'where', 'year', 'yes', 'you', 'your','i']
    all_words = [x.lower() for x in all_words]
    for i in text.lower().split():
        if i in all_words:
            matched_words.append(i)
            
    for i in matched_words:
    
        one_video = add_words_to_videos(i)
        videos_with_words.append(one_video)
        
    for i in videos_with_words:
        clip = VideoFileClip(i) 
        videos_links.append(clip)

    if len(videos_links)> 0:
        final_clip = concatenate_videoclips(videos_links) 
        final_clip.write_videofile(r"static\dataset\videos with words\my_concatenation.mp4") 
    else:
        final_clip = VideoFileClip(r"static\dataset\videos with words\idk\0.mp4") 
        final_clip.write_videofile(r"static\dataset\videos with words\my_concatenation.mp4") 
        
        
        
        
    cap = cv2.VideoCapture(r"static\dataset\videos with words\my_concatenation.mp4")

    while(cap.isOpened()):

        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
        
            # Display the resulting frame
            ret, buffer = cv2.imencode('.jpg', frame)
            im = buffer.tobytes()
            time.sleep(0.02)
            
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + im + b'\r\n') 
                




#------------------------------------------------------------------------------------------------
#Speech to Sign (Audio Input)
#------------------------------------------------------------------------------------------------
def speach_to_sign():
    sts = SpeechToSign(10)
    for frame in sts.start_pipeline():
        ret, buffer = cv2.imencode('.jpg', frame)
        im = buffer.tobytes()
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + im + b'\r\n')	





#------------------------------------------------------------------------------------------------
#Web App Starting 
#------------------------------------------------------------------------------------------------

@app.route('/')
def index():
    return render_template('0 - Start.html')

# SIGN TO SPEECH
@app.route('/Sign_to_Speech',methods =['POST'])
def Sign_to_Speech():
    return render_template('1 - Sign to Speech.html')

@app.route('/video_feed')
def video_feed():
    return Response(start_stream(), 
            mimetype='multipart/x-mixed-replace; boundary=frame')
            



# SPEECH TO SIGN -- text
@app.route('/Speech_to_Sign',methods =['POST'])
def Speech_to_Sign():
    return render_template('2 - Text or Speech.html')


@app.route('/enter_text',methods =['POST'])
def enter_text():
    return render_template('3 - Submit Text.html')


@app.route('/submit_to_testing', methods =['POST'])
def submit_to_testing():
    global text
        #  check if the current request from a user was performed using the HTTP "POST" method.
    if request.method == 'POST':
        #get text from textbox and store it in message variable.
        text = request.form['message']
        
    return render_template('4 - Text to Sign.html',submit_to_testing = text)


@app.route('/video_feed2')
def video_feed2():
    global text
    return Response(showing_videos(text), 
            mimetype='multipart/x-mixed-replace; boundary=frame')


# SPEECH TO SIGN -- audio
@app.route('/enter_audio',methods =['POST'])
def enter_audio():
    return render_template('5 - Speech to Sign.html')


@app.route('/return_text',methods =['POST'])
def return_text():
    global text 
    return text
 
@app.route('/app3_testing')
def app3_testing():
    global text 
    return Response(speach_to_sign(), 
            mimetype='multipart/x-mixed-replace; boundary=frame')
    
    
download_videos()    
if __name__ == "__main__":
    app.run(debug=False)
    

