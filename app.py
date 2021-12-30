# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 15:34:02 2021

@author: muhammad
"""
#Import necessary libraries
from flask import Flask, render_template, Response
#import cv2
#Initialize the Flask app
app = Flask(__name__)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from cv2 import KeyPoint, cv2
import tensorflow as tf
#from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Dropout
from tensorflow.keras.callbacks import TensorBoard
import mediapipe as mp
import sys
import os
import pyttsx3
import time
from moviepy.editor import VideoFileClip, concatenate_videoclips 
from flask import Flask,render_template,url_for,request
import speech_recognition as sr
import threading
import re


#------------------------------------------------------------------------------------------------
#Sign to Speech
#------------------------------------------------------------------------------------------------

def TTS_using_pyttsx3(text,gender='male',speed=175,volume=1):
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    if gender.lower() == 'male':
        engine.setProperty('voice', voices[0].id)  #changing index, changes voices. o for male
    else:
        engine.setProperty('voice', voices[1].id)   #changing index, changes voices. 1 for female
    engine.setProperty('rate', speed) ## speach speed 
    engine.setProperty('volume',1.0)  # setting up volume level  between 0 and 1 (min=0 and max=1)
    engine.say(text)    #put our text to process it 
    # engine.save_to_file(text, 'output.mp3')   # incase want to save file
    engine.runAndWait()  #speaking

def read_labels(path):
    names_list = os.listdir(path)
    equ_num_list = list(range(len(names_list)))
    num_of_videos_list = []
    
    for i in range(len(names_list)):
        sub_path = os.path.join(path, names_list[i])
        nov = len(os.listdir(sub_path))
        num_of_videos_list.append(nov)
    print(names_list)   
    df = pd.DataFrame(list(zip(names_list, equ_num_list, num_of_videos_list)), columns=['names', 'values', 'num_of_videos'])
    return df.set_index('names').iloc[:, 0].to_dict(), df.set_index('names').iloc[:, 1].to_dict()



def detect_keypoints(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def draw_landmarks(mp_drawing, mp_holistic_model, image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic_model.FACEMESH_CONTOURS,
                             mp_drawing.DrawingSpec(color=(10, 194, 80), thickness=1, circle_radius=1),
                             mp_drawing.DrawingSpec(color=(214, 200, 80), thickness=1, circle_radius=1))
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic_model.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(90, 194, 80), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(230, 200, 80), thickness=2, circle_radius=4))
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic_model.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(20, 194, 80), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(190, 200, 80), thickness=2, circle_radius=4))
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic_model.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(20, 194, 80), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(190, 200, 80), thickness=2, circle_radius=4))

def extract_keypoints(results):
    face_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark]).flatten()if results.face_landmarks else np.zeros(468*3)
    pose_landmarks = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    righ_hand_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    left_hand_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose_landmarks, face_landmarks, left_hand_landmarks, righ_hand_landmarks])


def start_stream(mp_holistic_model, holistic_model, mp_drawing, model, actions):

    sequence = []
    sentence = []
    predictions = []
    threshold = 0.75

    # read the video frames and save it in a list
    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    #cap.set(3, 600)
    #cap.set(4, 600)
    #cap.set(10, 0)
    #success, frame = cap.read()
    res = []
    while True: 
        success, frame = cap.read()
        
        if not success:
            break
        else:
            #width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            #height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
            image, results = detect_keypoints(frame, holistic_model)
            draw_landmarks(mp_drawing, mp_holistic_model, image, results)
    
            KeyPoints = extract_keypoints(results)
    
            sequence.append(KeyPoints)
            sequence = sequence[-40:]
    
    
            if len(sequence) == 40:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                predictions.append(np.argmax(res))
                #print(actions[np.argmax(res)])
    
            display = frame
            if True:
                display = image
            if True :
                ret, buffer = cv2.imencode('.jpg', display)
                im = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + im + b'\r\n')
    
            if len(predictions)> 0 and np.unique(predictions[-15:])[0] == np.argmax(res):
                if res[np.argmax(res)] > threshold:
                    if len(sentence) > 0:
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                            #TTS_using_pyttsx3(sentence[-1],'1',190)
                            
                            sentence_joined = ' '.join(x for x in sentence)
                            t1 = threading.Thread(target=TTS_using_pyttsx3, args=(sentence_joined,'1',190))
                            t1.daemon = True
                            t1.start()
                            
                    else:
                        sentence.append(actions[np.argmax(res)])
            print(sentence)

            #TTS_using_pyttsx3(sentence,'1',190)
            #t1 = threading.Thread(target=TTS_using_pyttsx3, args=(sentence,'1',190,))
            #t2 = threading.Thread(video_open_stream, args=(image,))
            
            #t1.start()
            #t2.start()

        
        
model = tf.keras.models.load_model('cv_model.h5')
sequence_length = 40
#actions = ['age', 'book', 'call', 'car', 'day', 'egypt', 'English', 'Enjoy', 'every', 'Excuse', 'football', 'Forget', 'fun', 'Good', 'hate', 'have', 'hello', 'help', 'holiday', 'Iam', 'love', 'meet', 'month', 'morning', 'my', 'NA', 'name', 'Nice', 'no', 'not', 'number', 'okay', 'picture', 'play', 'read', 'ride', 'run', 'sorry', 'speak', 'sport', 'take', 'thanks', 'time', 'today', 'understand', 'what', 'when', 'where', 'year', 'yes', 'you', 'your']
actions_map, num_of_videos = read_labels(r'C:\Users\muhammad\Desktop\slt_web\static\videos_with_40_frames')
actions = list(actions_map.keys())

mp_holistic_model = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
holistic_model = mp_holistic_model.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.5)
    

#------------------------------------------------------------------------------------------------
#Speech to Sign
#------------------------------------------------------------------------------------------------



#data_path =  r"C:\Users\muhammad\Desktop\slt_web\static\videos_with_40_frames"

#output_video_path = r"C:\Users\muhammad\Desktop"
def add_words_to_videos(name):
    
    video = cv2.VideoCapture(r"C:\Users\muhammad\Desktop\slt_web\static\videos_with_40_frames\{}\0.mp4".format(name))

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
    # is stored in 'filename.avi' file.
    result = cv2.VideoWriter(r'C:\Users\muhammad\Desktop\slt_web\static\videos with words\{}.mp4'.format(name), 
                             cv2.VideoWriter_fourcc(*'MJPG'),
                             10, size)

    while(True):
        ret, frame = video.read()

        font = cv2.FONT_HERSHEY_SIMPLEX

        cv2.putText(frame, 
            name, 
            (520, 50), 
            font, 1, 
            (0, 255, 255), 
            2, 
            cv2.LINE_4)

        if ret == True: 

            # Write the frame into the
            # file 'filename.avi'
            result.write(frame)

            # Display the frame
            # saved in the file
            #cv2.imshow('Frame', frame)

            # Press S on keyboard 
            # to stop the process
            if cv2.waitKey(1) & 0xFF == ord('s'):
                break

        # Break the loop
        else:
            break

    # When everything done, release 
    # the video capture and video 
    # write objects
    video.release()
    result.release()

    # Closes all the frames
    cv2.destroyAllWindows()

    print("The video was successfully saved")
    
    return r'C:\Users\muhammad\Desktop\slt_web\static\videos with words\{}.mp4'.format(name)




def showing_videos(text):
    listtttt = []  
    videos_links = []
    videos_with_words = []
    all_words = ['age', 'book', 'call', 'car', 'day', 'egypt', 'English', 'Enjoy', 'every', 'Excuse', 'football', 'Forget', 'fun', 'Good', 'hate', 'have', 'hello', 'help', 'holiday', 'Iam', 'love', 'meet', 'month', 'morning', 'my', 'NA', 'name', 'Nice', 'no', 'not', 'number', 'okay', 'picture', 'play', 'read', 'ride', 'run', 'sorry', 'speak', 'sport', 'take', 'thanks', 'time', 'today', 'understand', 'what', 'when', 'where', 'year', 'yes', 'you', 'your']
    all_words = [x.lower() for x in all_words]
    for i in text.lower().split():
        if i in all_words:
            listtttt.append(i)
            
    for i in listtttt:
    
        one_video = add_words_to_videos(i)
        videos_with_words.append(one_video)
        
    for i in videos_with_words:
        clip = VideoFileClip(i) 
        videos_links.append(clip)

    if len(videos_links)> 0:
        final_clip = concatenate_videoclips(videos_links) 
        final_clip.write_videofile(r"C:\Users\muhammad\Desktop\my_concatenation.mp4") 
    else:
        final_clip = VideoFileClip(r"C:\Users\muhammad\Desktop\slt_web\static\videos_with_40_frames\idk\0.mp4") 
        final_clip.write_videofile(r"C:\Users\muhammad\Desktop\my_concatenation.mp4") 
        
        
        
        #cap = cv2.VideoCapture(r"C:\Users\muhammad\Desktop\sign_language_translator-data_preparation\data_preparation\create_dataset_videos\videos_with_40_frames\{}\0.mp4".format(i))
        
    cap = cv2.VideoCapture(r"C:\Users\muhammad\Desktop\my_concatenation.mp4")
    #cap = cv2.VideoCapture(r"C:\Users\muhammad\Desktop\new_video.mp4")
    while(cap.isOpened()):

        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
        
            # Display the resulting frame
            #cv2.imshow('Frame', frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            im = buffer.tobytes()
            time.sleep(0.02)
            
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + im + b'\r\n') 
                




#------------------------------------------------------------------------------------------------
#Speech Identification
#------------------------------------------------------------------------------------------------

def STT_from_file(filename): #Still buggy but kinda works
    r = sr.Recognizer()
    with sr.AudioFile(filename) as source:
        r.adjust_for_ambient_noise(source)
        print("Converting Audio To Text ..... ")
        audio = r.listen(source)
    try:
        print("Converted Audio Is : \n" + r.recognize_google(audio))
    except Exception as e:
        print("Error {} : ".format(e) )
        
        
def STT_from_mic():
    global text
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Recording in : 3")
        time.sleep(2)
        print("Recording in : 2")
        time.sleep(2)
        print("Recording in : 1")
        time.sleep(2)
        print("Say Anything :")
        audio = r.listen(source)
    try:
        text = r.recognize_google(audio)
        print(text)
        return showing_videos(text)

    except:
        text = 'Sorry could not recognize what you said'
        print('Sorry could not recognize what you said')
        return showing_videos('idk')









#------------------------------------------------------------------------------------------------
#Web App Starting 
#------------------------------------------------------------------------------------------------

@app.route('/')
def index():
    return render_template('start.html')

# SIGN TO SPEECH
@app.route('/Sign_to_Speech',methods =['POST'])
def Sign_to_Speech():
    return render_template('Sign to Speech.html')

@app.route('/video_feed')
def video_feed():
    return Response(start_stream(mp_holistic_model
                                 , holistic_model
                                 , mp_drawing, 
                                 model, 
                                 actions), 
            mimetype='multipart/x-mixed-replace; boundary=frame')
            



# SPEECH TO SIGN -- text
@app.route('/Speech_to_Sign',methods =['POST'])
def Speech_to_Sign():
    return render_template('text or speech.html')


@app.route('/enter_text',methods =['POST'])
def enter_text():
    return render_template('submit text.html')


@app.route('/submit_to_testing', methods =['POST'])
def submit_to_testing():
    global text
        #  check if the current request from a user was performed using the HTTP "POST" method.
    if request.method == 'POST':
        #get text from textbox and store it in message variable.
        text = request.form['message']
        
    return render_template('testing.html',submit_to_testing = text)


@app.route('/video_feed2')
def video_feed2():
    global text
    return Response(showing_videos(text), 
            mimetype='multipart/x-mixed-replace; boundary=frame')


# SPEECH TO SIGN -- audio
@app.route('/enter_audio',methods =['POST'])
def enter_audio():
    return render_template('app_3.html')


@app.route('/return_text',methods =['POST'])
def return_text():
    global text 
    return text
 
@app.route('/app3_testing')
def app3_testing():
    global text 
    return Response(STT_from_mic(), 
            mimetype='multipart/x-mixed-replace; boundary=frame')
    
    
    
    
    
    
    
if __name__ == "__main__":
    app.run(debug=False)
    
    

