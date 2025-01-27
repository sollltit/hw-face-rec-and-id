from functionals import get_photo, get_recog_result
from ultralytics import YOLO
import pandas as pd
import numpy as np
import ast
import unittest
import warnings
warnings.filterwarnings('ignore')
import os

# инициализация модели
model = YOLO(r'runs\detect\v_3_s\weights\best.pt')
# инициализация датафрейма (база данных, эмбеддинги)
df_ut = pd.read_csv('emb_db.csv', sep = ',')
# в колонке 'embedding' все значения приводим к списку (до этого тип данных был str, а со строкой работать далее не получится)
df_ut['embedding'] = df_ut['embedding'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# тесты для функции get_photo
class TestFunctionGetPhoto(unittest.TestCase):
    # проверка расширения файла
    def test_invalid_file(self):
        val_img_path = r'id_data\train\Ann_Veneman\Ann_Veneman_0002.txt' # файл с неподходящим расширением
        model_ut = model
        self.assertFalse(get_photo(val_img_path, model_ut), 'файл должен быть с расширением .jpg или .png')
    
    # проверка на вывод корректного результата (не None)
    def test_val_result(self):
        corr_img = r'id_data\test\Paul_Bremer\Paul_Bremer_0009.jpg'
        res = get_photo(corr_img, model) # результат работы функции
        self.assertIsNotNone(res, 'функция не должна возвращать None')


# тесты для функции get_recog_result
class TestFunctionGetRecogResult(unittest.TestCase):
    # проверка на вывод не None
    def test_none_result(self):
        incorr_img = r'id_data\test\Winona_Ryder\Winona_Ryder_0009.jpg'
        # тк параметры для функции get_recog_result выявляются через функцию get_photo, сначала вызываем её
        crop_array = get_photo(incorr_img, model)[0]
        pathh = get_photo(incorr_img, model)[1]
        res = get_recog_result(crop_array, pathh, df_ut) # результат работы функции get_recog_result
        self.assertIsNotNone(res, 'функция не должна возвращать None')


    # проверка на то, что в качестве параметра crop_array передаётся как numpy массив
    def test_crop_array_check(self):
        incorr_img = r'id_data\test\Winona_Ryder\Winona_Ryder_0003.jpg'
        crop_array = get_photo(incorr_img, model)[0] # берём только crop_array
        self.assertIsInstance(crop_array, np.ndarray, 'crop_array должен быть numpy массивом') # проверяем тип данных у crop_array

    # проверка существования файла по пути (параметр f_path)
    def tect_check_f_path(self):
        incorr_img = r'id_data\test\Nicole_Kidman\Nicole_Kidman_0018.jpg'
        f_path = get_photo(incorr_img, model)[1] # берём только f_path (путь к файлу)
        self.assertTrue(os.path.exists(f_path), 'неверный путь: файл не найден :(')


# вызов юнит-тестов
if __name__ == '__main__':
    unittest.main()