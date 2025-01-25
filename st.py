import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import ast

# загрузка модели детекции
model_d = YOLO(r'runs\detect\variant_three\weights\best.pt')
model_8s = YOLO(r'runs\detect\v_3_s\weights\best.pt')

traindf = pd.read_csv('emb_db.csv', sep = ',')
traindf['embedding'] = traindf['embedding'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

def identity_face(img, model, df):
    res = []
    photo = cv2.imread(img)
    no_img_id = cv2.imread(r'no_id.jpg')
    no_d = cv2.imread(r'no_det.jpg')
    res_detect = model.predict(photo)
    if (res_detect[0].boxes) != 0:
        if len(res_detect[0].boxes) == 1:
            x1, y1, x2, y2 = map(int, res_detect[0].boxes.xyxy[0])
            crop_ph = photo[y1:y2, x1:x2]
            try:
                embedding_input = DeepFace.represent(img_path = crop_ph, model_name='VGG-Face')[0]['embedding']
                max_prob = 0
                best_res = None
                for i, embedding_db in enumerate(df['embedding']):
                    similarity = cosine_similarity([embedding_input], [embedding_db])[0][0]
                    if similarity > max_prob:
                        max_prob = similarity
                        best_res = df.iloc[i]

                if best_res is not None:
                    res = {'Name': best_res['Name'], 'similarity': max_prob}

                    st.write(f'Имя: {res["Name"]}\nСовпадение: {res["similarity"]}')
                    cv2.rectangle(photo, (x1, y1), (x2, y2), (195, 0, 130), 3)
                    return cv2.cvtColor(photo, cv2.COLOR_BGR2RGB)
                else:
                    st.write('Не удалось идентифицировать лицо')
                    return cv2.cvtColor(no_img_id, cv2.COLOR_BGR2RGB)
                    
            except:
                st.write('Не удалось идентифицировать лицо')
                return cv2.cvtColor(no_img_id, cv2.COLOR_BGR2RGB)
        else:
            no_d = cv2.imread(r'no_det.jpg')
            st.write('Ошибка! Обнаружено более одного лица. Пожалуйста, выберите фотографию с одним лицом.')
            return cv2.cvtColor(no_d, cv2.COLOR_BGR2RGB)
    else:
        no_d = cv2.imread(r'no_det.jpg')
        st.write('Ошибка! Лицо не обнаружено. Пожалуйста, выберите другую фотографию.')
        return cv2.cvtColor(no_d, cv2.COLOR_BGR2RGB)


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
    resultat = identity_face(tmp_f, model_8s, traindf) 
    st.image(resultat, caption='Результат')
    
