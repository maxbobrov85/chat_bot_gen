# Отчет. Домашнее Задание 2.
*Генеративный чат-бот*
### Датасет и обработка данных
Датасет взят на Kaggle.com
https://www.kaggle.com/datasets/pierremegret/dialogue-lines-of-the-simpsons
Он представляет собой диалоги из мультфильма Симпсоны. 
Обработка данных выполнена в ноутбуке [Preparing_dataset_Homer](https://github.com/maxbobrov85/chat_bot_gen/blob/main/Preparing_dataset_Homer.ipynb). Для датасета были разобраны диалоги, были выделены вопросы, ближайшие к этому вопросу реплики участников, ответы (хорошие и плохие) подобраные с помощью TF-IDF и т.д. После обработки сформирован окончательный датасет [final_dataset_homer.csv](https://github.com/maxbobrov85/chat_bot/blob/main/final_dataset_homer.csv)
### Модель
Модель состоит из BI (для генерации ответов) и Cross (для учета контекста) энкодеров. 
### BI энкодер
В этом же ноутбуке обучили BI энкодер на основе модели 'distilbert-base-uncased'. Обучение производилось на платформе Google Colab (GPU T4). График обучения модели (training_loss) представлен на рисунке

![Без имени](https://github.com/maxbobrov85/chat_bot/assets/114837957/7bf08a65-16a0-4a7f-b94c-2042c7d53489)
### Cross энкодер
Обучение кросс энкодера и формирование окончательной модели описано в ноутбуке [HW_1_cross_encoder_homer.ipynb](https://github.com/maxbobrov85/chat_bot/blob/main/HW_1_cross_encoder_homer.ipynb). График обучения модели (training_loss) представлен на рисунке

![Без имени](https://github.com/maxbobrov85/chat_bot/assets/114837957/ed01fad8-63fb-4ab7-9491-ec86a1c4940f)
Модели bi-encoder и cross-encoder были сохранены на huggingface для передачи в приложение flask
### Инференс
Создан из 2 файлов. Загрузка моделей (для ускорения модели грузятся в режиме float16) и генерация ответов прописаны в [answgen.py](https://github.com/maxbobrov85/chat_bot/blob/main/answgen.py). Описание flask приложения и хранение контекста вопросов реализовано в [app.py](https://github.com/maxbobrov85/chat_bot/blob/main/app.py):

![изображение](https://github.com/maxbobrov85/chat_bot/assets/114837957/27004573-9664-48da-a630-d080b205251d)
