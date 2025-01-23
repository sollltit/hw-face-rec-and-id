import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile

# загрузка модели детекции
model_d = YOLO(r'runs\detect\variant_three\weights\best.pt')


def detection_face(model, f_path):
    photo = cv2.imread(f_path)
    
    # результат детекции
    res = model.predict(photo, conf = 0.45, iou = 0.1)
    if len(res[0].boxes) != 0:
        for r in res:
            for box in r.boxes:
                # получаем координаты результата детекции
                x1, y1, x2, y2 = map(int, box.xyxy[0])
    
                # выделяем область детекции на фото
                cv2.rectangle(photo, (x1, y1), (x2, y2), (0, 176, 65), 2)
        
        return cv2.cvtColor(photo, cv2.COLOR_BGR2RGB)
    else:
        photo_non = cv2.imread('nono.jpg')
        return cv2.cvtColor(photo_non, cv2.COLOR_BGR2RGB)

# текст
st.title('Детекция лиц')
st.markdown('_________________________________________')
st.html("<p><tt><span style = 'font_size: 30px;'>Загрузите изображение для детекции</span></tt></p>")

# загрузка изображения
upload_file = st.file_uploader('Выбрать файл', accept_multiple_files=False, type = ['jpg', 'png'])

if upload_file is not None: # если фото загружено, для него создаётся временный путь
    with tempfile.NamedTemporaryFile(delete = False, suffix = 'jpg') as tmp_f:
        tmp_f.write(upload_file.getbuffer())
        tmp_f = tmp_f.name
    resultat = detection_face(model_d, tmp_f) 
    st.image(resultat, caption='Результат')
    
