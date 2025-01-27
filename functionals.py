import matplotlib.pyplot as plt
import cv2
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity


def get_photo(img_path, model):

    if not (img_path.lower().endswith('.jpg') or img_path.lower().endswith('.png')):
        return False
    else:
        img = cv2.imread(img_path)
        res_detect = model.predict(img)
        if len(res_detect[0].boxes) != 0:
            if len(res_detect[0].boxes) == 1:
                x1, y1, x2, y2 = map(int, res_detect[0].boxes.xyxy[0].numpy())
                crop_ph = img[y1:y2, x1:x2]
                return(crop_ph, img_path)
            else:
                return 'Обнаружено более одного лица'
        else:
            return None
        
            

def get_recog_result(crop_array, f_path, df):
    plt.imshow(crop_array)
    try:
        input_embedding = DeepFace.represent(img_path = crop_array, model_name='VGG-Face')[0]['embedding']
        max_prob = 0.4
        best_res = None
        for i, embeddings in enumerate(df['embedding']):
            similar = cosine_similarity([input_embedding], [embeddings])[0][0]
            if similar > max_prob:
                max_prob = similar
                best_res = df.iloc[i]
                if best_res is not None:
                    res = {'Name': best_res['Name'], 'similarity': max_prob, 'f_path': best_res['f_path']}
                    print(f'Имя: {res["Name"]}\nСовпадение: {res["similarity"]}')
                    return (res, f_path)
                else:
                    print('Человек не опознан')
                    return None
    except:
        print('Не удалось вычислить эмбеддинг')
        return None
