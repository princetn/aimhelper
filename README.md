# Aimhelper

This repo was created with a goal in mind to use AI object-detection for finding and tracking enemy players and to move the mouse pointer to target 
the enemy head closest in field of view. The code can be used in First-person shooter (FPS) video games to provide an aim assist and give you an edge
over the enemy players in an FPS game.

### Structure of this repo: 
There are mainly 3 independent separate python scripts:
* main.py uses TensorFlow object-detection API and is an aim-helper program. A frozen graph can be generated from TensorFlow Object-detection model zoo and 
 get it trained on a specific game like Call of Duty operators and then convert it to a frozen inference graph
that can be consumed by main.py
* yolo.py uses Yolov8 (The Yolov8 detection code was inspired by the following url: https://www.geeksforgeeks.org/object-detection-using-yolov8/ ).
* tensorflow_api_converter_frozen_graph.py: this code is not an aim helper. It is mainly for converting a TensorFlow trained saved model to a frozen model
that can be deployed on with main.py 
* Please note that this work is purely experimental and requires further grooming for better results.

## Disclaimer: 
This code is provided as is and was developed for entertainment purposes only and does not encourage cheating in games. The code is free to use and I do not assume any responsibility for how you will use it. 

## Known Issues
* Most games use anti-cheat and will prevent a virtual mouse movement Event from orienting the player to target enemy. 
Currently the code uses PyAutogui library for moving the mouse. Maybe finding a better library will make the aimhelper 
work on games. Modern video games do not use Windows API that pyautogui relies on. 
Windows API messages pump will actually flag the virtual mouse input as Injected which probably a game anti-cheat can ignore. 
* Second issue both Neural Net models used in the Machine learning based detection are not trained proprely on game.
So for that reason they may not be as accurate. But if you manage to generate a training data and train those models 
you will see much better results.
* Yolov8 currently is much better at detecting Call of Duty operators.

## How to run
1) First you need python 3.9 or later. code was tested on 3.9
2) You will need to install all required modules listed in requirements.txt
3) For aimhelper using Yolo V8: run  in a terminal ``` py -3.9 yolo.py ```
4) For aimhlper using TensorFlow: run in a terminal ``` py -3.9 main.py```