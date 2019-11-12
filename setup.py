import os

print('\nVeryfing and installing dependences...\n')
os.system("pip3 install numpy")
os.system("pip3 install opencv-python")
os.system("pip3 install Pillow")
os.system("clear")
print('All dependences updated!\n')

op = input("(1) Trainning with all elements\n(2) Trainning with specific elements\n\n->")
print('\nTrainning system...')

if op == '1':
    os.system("python3 faces_train.py")
else:
    if op == '2':
        elem = input("\nElements name (without ','): ")
        os.system("python3 faces_train_specific.py '" + elem + "'")
    else:
        print("\nInvalid input!\n")
        exit()

print('\nStarting cam...\n')
os.system("python3 faces.py")