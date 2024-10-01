import cv2
import numpy as np

# Load pre-trained Haar cascade model for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the video capture (0 for webcam)
cap = cv2.VideoCapture(0)

# Check if the capture is open
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# List of filter image paths
filter_images = [
    r'C:\Users\hp\OneDrive\Desktop\cvproject\m1.png',  # Filter 1
    r'C:\Users\hp\OneDrive\Desktop\cvproject\m2.png',  # Filter 2
    r'C:\Users\hp\OneDrive\Desktop\cvproject\m3.png'   # Filter 3 (Add your own)
]

# Load the filters
filters = [cv2.imread(filter_path, cv2.IMREAD_UNCHANGED) for filter_path in filter_images]

# Check if images are loaded successfully
if any(f is None for f in filters):
    print("Error: Failed to load AR filter images.")
    exit()

# Prompt the user to select a filter
print("Select a filter to use:")
for i, img_path in enumerate(filter_images):
    print(f"{i + 1}: {img_path}")


while True:
    try:
        user_input = int(input("Enter the number of the filter you want to use (1-3): "))
        if 1 <= user_input <= len(filters):
            current_filter_index = user_input - 1
            break
        else:
            print(f"Please enter a number between 1 and {len(filters)}.")
    except ValueError:
        print("Invalid input. Please enter a number.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Iterate through detected faces
    for (x, y, w, h) in faces:
        # Get the current filter
        current_filter = filters[current_filter_index]
        filter_resized = cv2.resize(current_filter, (w, h), interpolation=cv2.INTER_AREA)

        # Overlay filter image (with alpha blending if it has an alpha channel)
        if filter_resized.shape[2] == 4:  # Check for alpha channel
            alpha_filter = filter_resized[:, :, 3] / 255.0
            for c in range(0, 3):
                frame[y:y + filter_resized.shape[0], x:x + filter_resized.shape[1], c] = \
                    (1 - alpha_filter) * frame[y:y + filter_resized.shape[0], x:x + filter_resized.shape[1], c] + \
                    alpha_filter * filter_resized[:, :, c]
        else:
            frame[y:y + filter_resized.shape[0], x:x + filter_resized.shape[1]] = filter_resized

    # Display the augmented reality video stream
    cv2.imshow('AR Filters', frame)

    # Check for key presses to change filters or stop the application
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Quit
        print("Quitting the application...")
        break
    elif key == ord('n'):  # Next filter
        current_filter_index = (current_filter_index + 1) % len(filters)
    elif key == ord('s'):  # Stop application
        print("Stopping the application...")
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
