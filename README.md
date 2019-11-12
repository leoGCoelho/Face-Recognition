# Face-Recognition
### Resume
A face recognition system using supervised learning (OpenCV + NumPy), analizing images set and the recorded on the webcam. 
### Getting Started
  - Clone Repository;
  - Include (if necessary) more pictures for trainning on the **images** folder, tagging the element folder;
  - At first run, type the command `python3 setup.py` to install/update all the dependences, train the system and open the camera;
  - With the system trainned, just type `python3 faces.py` on terminal to start the camera;
  - To train the system:
    - Use `python3 faces_train.py` to train with all elements folders;
    - Or use `python3 faces_train_specify.py "NameFolder1 NameFolder2 ... NameFolderX"` to train with specific element folders;
