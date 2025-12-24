🖐️ Real-Time Sign Language Detection System
📌 Project Overview

This project is a Real-Time Sign Language Detection System developed using Python, OpenCV, and MediaPipe.
It aims to help bridge the communication gap between hearing/speech-impaired individuals and normal users by recognizing hand gestures and converting them into readable text.

🎯 Objectives

Detect hand gestures in real time using a webcam

Extract hand landmarks using MediaPipe

Classify gestures using a Machine Learning model

Display detected signs as text on the screen

🛠️ Technology Stack

Programming Language: Python

Libraries:

OpenCV

MediaPipe

NumPy

Scikit-learn

IDE: VS Code / PyCharm

Hardware: Webcam

Operating System: Windows

⚙️ System Workflow

Webcam captures live video frames

OpenCV processes the frames

MediaPipe detects 21 hand landmarks

Features are extracted from landmarks

ML model classifies the gesture

Output is displayed as text

📂 Project Structure
Sign-Language-Detection/
│
├── data/                 # Dataset folders (gesture images)
├── model/                # Trained ML model
├── create_data.py        # Dataset creation script
├── train_model.py        # Model training script
├── inference_classifier.py  # Real-time prediction
├── requirements.txt      # Required libraries
└── README.md             # Project documentation

▶️ How to Run the Project
1️⃣ Install Required Libraries
pip install opencv-python mediapipe numpy scikit-learn

2️⃣ Create Dataset
python create_data.py

3️⃣ Train the Model
python train_model.py

4️⃣ Run Real-Time Detection
python inference_classifier.py

✅ Feasibility

Uses open-source tools

No high-end hardware required

Easy to use and cost-effective

📈 Expected Outcomes

Accurate real-time sign detection

Improved communication for hearing-impaired users

Foundation for future enhancements like speech output

🚀 Future Scope

Add speech synthesis

Support full sign language sentences

Mobile application version

Improve accuracy using deep learning

📚 References

OpenCV Official Documentation

MediaPipe Documentation

IEEE Research Papers on Hand Gesture Recognition
