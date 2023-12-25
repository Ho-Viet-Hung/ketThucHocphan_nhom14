import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import numpy as np
import cv2
from keras.models import load_model
from datetime import datetime

# Tải mô hình Keras đã được huấn luyện để phân loại động vật
model = load_model('keras_model.h5')

# Định nghĩa các lớp cho nhãn động vật
classes = {1: 'Thưa sếp đây là con mèo ạ!',
           0: 'Thưa sếp đây là con chó ạ!',
           2: 'Thưa sếp đây là con bò ạ!', }

# Tạo cửa sổ chính của tkinter
top = tk.Tk()
top.geometry('800x600')
top.title('Phân loại động vật')
top.configure(background='#CDCDCD')

# Các nhãn để hiển thị kết quả phân loại và mức độ tin cậy
label = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
confidence_label = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))

# Nhãn để hiển thị hình ảnh
sign_image = Label(top)
img_list = []
total_dogs = 0
total_cats = 0
total_cows = 0  # Đã thêm để đếm số lượng bò

# Hàm để phân loại hình ảnh đã tải lên
def classify_image(image):
    global total_dogs, total_cats, total_cows
    # Thay đổi kích thước ảnh để phù hợp với kích thước đầu vào mong đợi của mô hình
    image = image.resize((224, 224))
    image = np.expand_dims(image, axis=0)
    image = np.array(image)
    image = image / 255.0
    pred_probabilities = model.predict([image])
    pred_class = np.argmax(pred_probabilities, axis=-1)[0]
    sign = classes[pred_class]
    label.configure(foreground='#011638', text=sign)

    save_result_to_txt(sign)

    confidence_level = pred_probabilities[0][pred_class] * 100
    confidence_label.configure(foreground='#011638', text=f'Phần trăm xác định: {confidence_level:.2f}%')

    if 'chó' in sign:
        total_dogs += 1
    elif 'mèo' in sign:
        total_cats += 1
    elif 'bò' in sign:
        total_cows += 1

# Hàm để hiển thị tổng số con chó, mèo và bò
def show_totals():
    label.configure(foreground='#011638', text=f'Tổng số con chó: {total_dogs}\nTổng số con mèo: {total_cats}\nTổng số con bò: {total_cows}')

# Hàm để lưu kết quả phân loại vào một tệp văn bản
def save_result_to_txt(result):
    try:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f'ketqua_{timestamp}.txt'
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(result + '\n')

    except Exception as e:
        print(e)

# Hàm để liên tục phân loại các khung hình từ webcam
def classify_webcam_frame():
    ret, frame = cap.read()

    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        if len(faces) == 0:
            img = Image.fromarray(frame)
            img.thumbnail(((top.winfo_width() / 2.25), (top.winfo_height() / 2.25)))
            im = ImageTk.PhotoImage(img)
            sign_image.configure(image=im)
            sign_image.image = im
            classify_image(img)
        else:
            label.configure(foreground='#011638', text='Phát hiện khuôn mặt. Bỏ qua phân loại động vật.')

    top.after(10, classify_webcam_frame)

# Hàm để tải lên một ảnh từ tệp
def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        if file_path:
            if not file_path.lower().endswith(('.jpg', '.jpeg')):
                label.configure(foreground='#FF0000', text='Vui lòng chọn ảnh có đuôi .jpg hoặc .jpeg.')
                return

            uploaded = Image.open(file_path)
            uploaded.thumbnail(((top.winfo_width() / 2.25), (top.winfo_height() / 2.25)))
            im = ImageTk.PhotoImage(uploaded)
            sign_image.configure(image=im)
            sign_image.image = im
            img_list.append(uploaded)

            # Hiển thị nút "Phân loại" cho ảnh hiện tại
            classify_b = Button(top, text="Phân loại", command=lambda img=uploaded: classify_image(img), padx=10,
                                pady=5)
            classify_b.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))
            classify_b.place(relx=0.79, rely=0.46)

            label.configure(text='')

    except Exception as e:
        print(e)

# Nút để kích hoạt chế độ phân loại từ webcam
webcam_button = Button(top, text="Sử dụng Webcam", command=classify_webcam_frame, padx=10, pady=5)
webcam_button.configure(background='#364156', foreground='black', font=('arial', 10, 'bold'))
webcam_button.pack(side=BOTTOM, pady=10)

# Nút để tải lên một ảnh từ tệp
upload = Button(top, text="Tải ảnh lên", command=upload_image, padx=10, pady=5)
upload.configure(background='#364156', foreground='black', font=('arial', 10, 'bold'))
upload.pack(side=BOTTOM, pady=10)

# Nút để hiển thị tổng số
total_button = Button(top, text="Tổng số con chó và con mèo", command=show_totals, padx=10, pady=5)
total_button.configure(background='#364156', foreground='black', font=('arial', 10, 'bold'))
total_button.pack(side=BOTTOM, pady=10)

# Nhãn để hiển thị hình ảnh
sign_image.pack(side=BOTTOM, expand=True)
label.pack(side=BOTTOM, expand=True)
heading = Label(top, text="Phân loại động vật", pady=20, font=('arial', 20, 'bold'))
heading.configure(background='#CDCDCD', foreground='black')
heading.pack()

# Thiết lập webcam và bộ phân loại khuôn mặt
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Thêm một Nhãn để hiển thị mức độ tin cậy
confidence_label.pack(side=BOTTOM, expand=True)

# Bắt đầu vòng lặp chính của tkinter
top.mainloop()
