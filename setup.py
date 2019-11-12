import os

print('\nVeryfing and installing dependences...\n')
os.system("pip3 install numpy")
os.system("pip3 install opencv-python")
os.system("pip3 install Pillow")
os.system("clear")
print('All dependences updated!\n')

print('\nTrainning system...')
os.system("python3 faces_train.py")

print('\nStarting cam...\n')
os.system("python3 faces.py")