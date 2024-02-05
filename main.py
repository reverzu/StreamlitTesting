import cv2
import streamlit as st
import streamlit.components.v1 as components
import mediapipe as mp
import numpy as np
import staticData
import random
import os
import findFiles
import dataCollector
import facialEmotion
from keras.models import load_model

if 'emotion' not in st.session_state:
    st.session_state['emotion'] = None
if 'play_something_else' not in st.session_state:
    st.session_state['play_something_else'] = None


try:
    # Loading Model
    model = load_model("Models/model.h5")
    # Loading Labels
    labels = np.load("Models/labels.npy")
except:
    model = None

st.set_page_config(page_title="🎶 AI Music Therapy", layout='wide')
st.markdown(""" <style>
                .block-container{
                padding-top:1.5rem;
                }
                .container {
                display: inline-block;
                }
                div.stButton > button:first-child {
                height:2.5em;
                width:100%;
                }
                div.stButton > button:first-child:hover{
                border:1 px solid gray;
                }
                .typed-out{
                overflow: hidden;
                border-right: .15em solid orange;
                white-space: nowrap;
                font-size: 1.1rem;
                margin: 0 auto; /* Gives that scrolling effect as the typing happens */
                letter-spacing: .15em; /* Adjust as needed */
                width: 0;
                animation: typing 1s forwards,
                blink-caret .75s step-end infinite;
                font-weight: 550;
                }		
                .typewriter{
                overflow: hidden; /* Ensures the content is not revealed until the animation */
                border-right: .12em solid orange; /* The typwriter cursor */
                white-space: nowrap; /* Keeps the content on a single line */
                font-size: 1.6rem;
                margin: 0 auto; /* Gives that scrolling effect as the typing happens */
                letter-spacing: .15em; /* Adjust as needed */
                animation: 
                typing 3.5s forwards,
                blink-caret .75s step-end infinite;
                font-size:1rem;
                display: inline-block;
                width: 0;
                }

                /* The typing effect */
                @keyframes typing {
                from { width: 0 }
                to { width: 100% }
                }

                /* The typewriter cursor effect */
                @keyframes blink-caret {
                from, to { border-color: transparent }
                50% { border-color: orange; }
                }
                </style>""",
                unsafe_allow_html=True)


st.title("🎶 AI Music Therapy")
st.markdown(f'<div class="container"><div class="typed-out">"{staticData.music_quotes[random.randint(0, len(staticData.music_quotes) - 1)]}</div></div><br><br>', unsafe_allow_html=True)


if model is None:
    spacel, middle, spacer = st.columns([2, 8, 2])
    with middle:
        st.markdown("<h2 style='text-align: center;'>Train the Model to your Facial Expressions</h2>",
                    unsafe_allow_html=True)
        st.markdown(
            f'''<h4 style='text-align: center;'>Show different Faces for the Emotions</h4>''', unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 1, 1])
        col11, col12 = st.columns([1, 1])

        with col1:
            happy_collect = st.button("Happy 😊", use_container_width=True)
        with col2:
            neutral_collect = st.button("Netural 😐", use_container_width=True)
        with col3:
            sad_collect = st.button("Sad 😢", use_container_width=True)

        if happy_collect:
            st.markdown(
                "<h4 style='text-align: center;'>Show us your Happy 😊 Face</h4>", unsafe_allow_html=True)
            dataCollector.startCollectingData('Happy')
        if neutral_collect:
            st.markdown(
                "<h4 style='text-align: center;'>Show us your Neutral 😐 Face</h4>", unsafe_allow_html=True)
            dataCollector.startCollectingData('Neutral')
        if sad_collect:
            st.markdown(
                "<h4 style='text-align: center;'>Show us your Sad 😢 Face</h4>", unsafe_allow_html=True)
            dataCollector.startCollectingData('Sad')
        
        with col12:
            happy_file = findFiles.findFiles('Happy.npy', './Models/')
            sad_file = findFiles.findFiles('Sad.npy', './Models/')
            neutral_file = findFiles.findFiles('Neutral.npy', './Models/')
            if happy_file and neutral_file and sad_file:
                train_model_button = st.button(label="Train the Model ➡", type="primary", use_container_width=True)
                if train_model_button:
                    with col11:
                        creating_progress = st.spinner("Training the Model... (Please be patient, it may take a few minutes)")
                        with creating_progress:
                            import modelTrainer
                            modelTrainer.train()
            else:
                st.markdown(
                    "<h4 style='text-align: center;'>Emotion Values are Missing...</h4>", unsafe_allow_html=True)


else:
    tab1, tab2, tab3 = st.tabs(['Home', 'Chat Companion', "AI Music Generator"])

    with tab1:
        col_main1, col_main2 = st.columns([8, 4])
        with col_main1:
            key, value = random.choice(list(staticData.articles_dict.items()))
            st.subheader(key)
            st.image("./images/slides/" +
                 random.choice(os.listdir("./images/slides/")))
            st.markdown(value[0])
            st.markdown(value[1])
            st.markdown('---')
            mainHeaderPlaceholder = st.empty()
            mainHeader = mainHeaderPlaceholder.markdown(
                "<h1 style='text-align: center;'>Hello there, How are we feeling today?</h1><br>", unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns([1, 1, 1])

            with col1:
                happy = st.button("Play Something Happy 😊", use_container_width=True)
            with col2:
                neutral = st.button("Play Something Netural 😐", use_container_width=True)
            with col3:
                sad = st.button("Play Something Sad 😢", use_container_width=True)

            st.markdown("<br>", unsafe_allow_html=True)

            if happy:
                st.image(findFiles.imageOnEmotion('Happy'))
                findFiles.autoplay_audio('Happy')
            elif sad:
                st.image(findFiles.imageOnEmotion('Sad'))
                findFiles.autoplay_audio('Sad')
            elif neutral:
                st.image(findFiles.imageOnEmotion('Neutral'))
                findFiles.autoplay_audio('Neutral')

            st.markdown('<hr>', unsafe_allow_html=True)
            st.markdown(
                "<h1 style='text-align: center;'>Capture Emotion from your webcam?</h1><br>", unsafe_allow_html=True)

            colb1, colb2 = st.columns([1, 1])
            with colb1:
                emotion_capture_button = st.button(
                    "Capture My Facial Emotion 📷", use_container_width=True)
            with colb2:
                Play_something_else_button = st.button(
                    "Play Something Else 🎵", use_container_width=True)
                
            if emotion_capture_button:
                facialEmotion.startFacialEmotionRecognition()
            if st.session_state['emotion'] is not None:
                emotion_capture_result = st.empty()
                emotion_capture_result.markdown(
                f"<h2 style='text-align: center;'>It Looks like you're {st.session_state['emotion']}</h1>", unsafe_allow_html=True)
                st.image(findFiles.imageOnEmotion(st.session_state['emotion']))
                findFiles.autoplay_audio(st.session_state['emotion'])
                st.session_state['emotion'] = None
            else:
                pass

            if Play_something_else_button and st.session_state['play_something_else'] is not None:
                emotion_capture_play_something_else_result = st.empty()
                emotion_capture_play_something_else_result.markdown(
                    f"<h2 style='text-align: center;'>It Looks like you're {st.session_state['play_something_else']}</h1>", unsafe_allow_html=True)
                st.image(findFiles.imageOnEmotion(
                    st.session_state['play_something_else']))
                findFiles.autoplay_audio(
                    st.session_state['play_something_else'])
            else:
                pass

            components.html('''<script>
                        const audio_files = document.getElementsByTagName('audio');
                        console.log(audio_files);
                        </script>''')


    with tab2:
        pass
    with tab3:
        pass