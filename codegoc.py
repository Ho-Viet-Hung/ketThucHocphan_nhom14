import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import numpy as np
import cv2
from keras.models import load_model

model = load_model('Model.h5')

classes = {0: 'Thưa sếp đây là con mèo ạ!', 1: 'Thưa sếp đây là con chó ạ!', }

top = tk.Tk()
top.geometry('800x600')
top.title('Phân loại động vật')
top.configure(background='#CDCDCD')
label = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
sign_image = Label(top)
img = None  # Khai báo biến img là biến toàn cục
from datetime import datetime

# Phân loại động vật từ ảnh và lưu kết quả vào file txt
def classify_image(image):
    global img  # Sử dụng biến toàn cục img
    image = image.resize((128, 128))
    image = np.expand_dims(image, axis=0)
    image = np.array(image)
    image = image / 255.0
    pred_probabilities = model.predict([image])
    pred_class = np.argmax(pred_probabilities, axis=-1)[0]
    sign = classes[pred_class]
    label.configure(foreground='#011638', text=sign)

    # Lưu kết quả vào file txt
    save_result_to_txt(sign)

def save_result_to_txt(result):
    try:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f'classification_result_{timestamp}.txt'
        with open(filename, 'w') as file:
            file.write(result + '\n')
        label.configure(text=f'Kết quả đã được lưu vào {filename}')
    except Exception as e:
        print(e)

# Phân loại động vật từ webcam
def classify_webcam_frame():
    global img  # Sử dụng biến toàn cục img
    ret, frame = cap.read()

    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Phát hiện khuôn mặt
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        if len(faces) == 0:
            # Không có khuôn mặt được phát hiện, tiếp tục với việc phân loại động vật
            img = Image.fromarray(frame)
            img.thumbnail(((top.winfo_width() / 2.25), (top.winfo_height() / 2.25)))
            im = ImageTk.PhotoImage(img)
            sign_image.configure(image=im)
            sign_image.image = im
            classify_image(img)
        else:
            # Khuôn mặt được phát hiện, hiển thị thông báo hoặc thực hiện hành động phù hợp
            label.configure(foreground='#011638', text='Phát hiện khuôn mặt. Bỏ qua phân loại động vật.')

    top.after(10, classify_webcam_frame)

# Tải ảnh từ file
def upload_image():
    try:
        global img  # Sử dụng biến toàn cục img
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width() / 2.25), (top.winfo_height() / 2.25)))
        im = ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image = im
        classify_b = Button(top, text="Phân loại", command=lambda: classify_image(uploaded), padx=10, pady=5)
        classify_b.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))
        classify_b.place(relx=0.79, rely=0.46)
        label.configure(text='')
    except Exception as e:
        print(e)

# Nút sử dụng webcam
webcam_button = Button(top, text="Sử dụng Webcam", command=classify_webcam_frame, padx=10, pady=5)
webcam_button.configure(background='#364156', foreground='black', font=('arial', 10, 'bold'))
webcam_button.pack(side=BOTTOM, pady=10)

# Nút tải ảnh lên
upload = Button(top, text="Tải ảnh lên", command=upload_image, padx=10, pady=5)
upload.configure(background='#364156', foreground='black', font=('arial', 10, 'bold'))
upload.pack(side=BOTTOM, pady=10)

sign_image.pack(side=BOTTOM, expand=True)
label.pack(side=BOTTOM, expand=True)
heading = Label(top, text="Phân loại động vật", pady=20, font=('arial', 20, 'bold'))
heading.configure(background='#CDCDCD', foreground='black')
heading.pack()

# Khởi tạo webcam
cap = cv2.VideoCapture(0)

# Khởi tạo bộ phân lớp khuôn mặt Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

top.mainloop()
