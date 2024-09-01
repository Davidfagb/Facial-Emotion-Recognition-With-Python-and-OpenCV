import cv2
from deepface import DeepFace
from datetime import datetime

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Define the desired output window size
output_width = 640  # Width of the window frame
output_height = 480  # Height of the window frame

# Define a confidence threshold for emotion detection
confidence_threshold = 0.5

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Analyze the frame for emotion using higher resolution for accuracy
    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, align=True)
    except Exception as e:
        print(f"Error: {e}")
        continue

    if result:
        for face in result:
            emotion = face['dominant_emotion']
            score = face['emotion'][emotion]

            if score >= confidence_threshold:  # Only display emotions with high confidence
                x, y, w, h = face['region']['x'], face['region']['y'], face['region']['w'], face['region']['h']

                # Draw rectangle around the face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                # Display the emotion label with confidence score
                cv2.putText(frame, f"{emotion}: {score:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Add date and time to the frame
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cv2.putText(frame, timestamp, (10, output_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Resize the frame to the desired output size
    resized_frame = cv2.resize(frame, (output_width, output_height))

    # Display the resulting frame
    cv2.imshow('Facial Emotion Recognition', resized_frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
