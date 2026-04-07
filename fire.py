from ultralytics import YOLO
import cvzone
import cv2
import math
from playsound import playsound
import threading
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import ssl

# Email configuration
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465  # Changed to SSL port
SENDER_EMAIL = "amanpuff1@gmail.com"
# Replace this with your 16-character App Password from Google
SENDER_PASSWORD = "jawy snfv zjjk vcuz"
RECEIVER_EMAIL = "amanyaduvanshi.7492@gmail.com"

model_path = '/Users/amanyadav/Desktop/Fire_Detector_model/fire.pt'  # if in Downloads

#already trained this in google callab 

model = YOLO(model_path)

cap = cv2.VideoCapture(0)

# Function to send email notification
def send_email_notification():
    try:
        # Create message
        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = RECEIVER_EMAIL
        msg['Subject'] = "Fire Detection Alert!"
        
        # Email body
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        body = f"Fire has been detected!\n\nTime of detection: {current_time}\n\nThis is an automated alert from your Fire Detection System."
        msg.attach(MIMEText(body, 'plain'))
        
        # Create SSL context
        context = ssl.create_default_context()
        
        # Create SMTP session with SSL
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT, context=context) as server:
            # Login to the server
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            
            # Send email
            text = msg.as_string()
            server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, text)
            
        print("Email notification sent successfully!")
    except smtplib.SMTPAuthenticationError:
        print("Error: Email authentication failed. Please check your email and app password.")
    except smtplib.SMTPException as e:
        print(f"Error sending email: {str(e)}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")

# Function to play alarm sound
def play_alarm():
    playsound('alarm.mp3')

alarm_thread = None
email_thread = None
last_fire_time = 0
last_email_time = 0

classnames = ['fire']

confidence_threshold = 0.75  # Increased confidence threshold

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Unable to read frame from webcam.")
        break

    frame = cv2.resize(frame, (640, 480))

    result = model(frame, stream=True)

    fire_detected = False
    current_time = time.time()

    for info in result:
        boxes = info.boxes
        for box in boxes:
            confidence = box.conf[0]  # Get raw confidence value
            Class = int(box.cls[0])

            if confidence > confidence_threshold:  # Use the threshold directly
                fire_detected = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
                confidence_percentage = math.ceil(confidence * 100)
                cvzone.putTextRect(frame, f'{classnames[Class]} {confidence_percentage}%', [x1 + 8, y1 + 100],
                                   scale=1.5, thickness=2)

    # Handle alarm and email notifications
    if fire_detected:
        # Play alarm if enough time has passed
        if (current_time - last_fire_time) > 2.0:
            if alarm_thread is None or not alarm_thread.is_alive():
                alarm_thread = threading.Thread(target=play_alarm, daemon=True)
                alarm_thread.start()
                last_fire_time = current_time
        
        # Send email if enough time has passed (e.g., every 5 minutes)
        if (current_time - last_email_time) > 300:  # 300 seconds = 5 minutes
            if email_thread is None or not email_thread.is_alive():
                email_thread = threading.Thread(target=send_email_notification, daemon=True)
                email_thread.start()
                last_email_time = current_time

    cv2.imshow('frame', frame)
    
    if cv2.waitKey(1) == ord('s'):
        break

cap.release()
cv2.destroyAllWindows()

















