import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import torch
import torch.nn as nn
import tempfile
import os
from collections import Counter
import re

# ----- Конфиг -----
ALPHABET = "0123456789ABCEHKMOPTXY"
NUM_CLASSES = len(ALPHABET) + 1
IMG_HEIGHT = 32
IMG_WIDTH = 128

# ----- CRNN модель -----
class CRNN(nn.Module):
    def __init__(self, num_classes):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1))
        )
        self.rnn = nn.LSTM(512 * 2, 256, bidirectional=True, num_layers=2, batch_first=True)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        b, c, h, w = x.size()
        x = x.reshape(b, c * h, w)
        x = x.permute(0, 2, 1)
        x, _ = self.rnn(x)
        x = self.classifier(x)
        x = x.permute(1, 0, 2)
        x = nn.functional.log_softmax(x, dim=2)
        return x

# ----- Декодер -----
def ctc_decode(preds):
    preds = preds.permute(1, 0, 2)
    preds = torch.argmax(preds, dim=2)
    int_to_char = {i + 1: char for i, char in enumerate(ALPHABET)}
    decoded_texts = []
    for pred in preds:
        decoded_seq = []
        last_char_idx = 0
        for char_idx in pred:
            char_idx = char_idx.item()
            if char_idx != 0 and char_idx != last_char_idx:
                decoded_seq.append(int_to_char.get(char_idx, ''))
            last_char_idx = char_idx
        decoded_texts.append("".join(decoded_seq))
    return decoded_texts

# ----- Валидация российских номеров -----
def validate_russian_plate(text):
    """
    Проверка формата российского номера:
    - Буква + 3 цифры + 2 буквы + 2-3 цифры региона
    - Только допустимые буквы: A, B, C, E, H, K, M, O, P, T, X, Y
    """
    if not text or len(text) < 8:
        return False
    
    # Регулярное выражение для российских номеров
    pattern = r'^[ABEKMHOPCTXY]\d{3}[ABEKMHOPCTXY]{2}\d{2,3}$'
    
    if re.match(pattern, text):
        return True
    return False

# ----- Препроцессинг ROI -----
def preprocess_roi(roi):
    img = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT)).astype(np.float32)/255.0
    img = (img - 0.5) / 0.5
    img = torch.tensor(img).unsqueeze(0).unsqueeze(0)
    return img

# ----- Загрузка моделей -----
@st.cache_resource
def load_ocr_model():
    model = CRNN(NUM_CLASSES)
    model.load_state_dict(torch.load('crnn_ocr_model_best.pth', map_location='cpu'))
    model.eval()
    return model

@st.cache_resource
def load_yolo_model():
    return YOLO('best.pt')

ocr_model = load_ocr_model()
det_model = load_yolo_model()

def recognize_crnn(roi, ocr_model):
    input_tensor = preprocess_roi(roi)
    with torch.no_grad():
        output = ocr_model(input_tensor)
        text = ctc_decode(output)[0]
    return text.strip()

def process_frame(image, conf_threshold=0.25):
    results = det_model.predict(image, conf=conf_threshold, device='cpu', verbose=False)
    detected_plates = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            roi = image[y1:y2, x1:x2]
            if roi.size > 0:
                text = recognize_crnn(roi, ocr_model)
                text = text.replace('\n', '').replace(' ', '')
                
                # Валидация номера
                is_valid = validate_russian_plate(text)
                
                # Цвет рамки: зеленый если валидный, оранжевый если нет
                color = (0, 255, 0) if is_valid else (255, 165, 0)
                
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                label = f"{text} ({confidence:.2f})"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(image, (x1, y1 - 25), (x1 + w, y1), color, -1)
                cv2.putText(image, label, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                
                detected_plates.append({
                    'text': text,
                    'confidence': confidence,
                    'bbox': (x1, y1, x2, y2),
                    'valid': is_valid
                })
    return image, detected_plates

# ----- Фильтрация номеров по частоте появления -----
def filter_plates_by_frequency(plate_counter, min_occurrences=5):
    """
    Оставляет только номера, которые появились минимум min_occurrences раз
    и прошли валидацию формата
    """
    filtered = {}
    for plate, count in plate_counter.items():
        if count >= min_occurrences and validate_russian_plate(plate):
            filtered[plate] = count
    return filtered

# ---------- Streamlit UI ----------
st.set_page_config(
    page_title="Распознавание автомобильных номеров",
    page_icon="🚗",
    layout="wide"
)
st.title("🚗 Система распознавания российских номеров: YOLOv8 + CRNN-OCR")
st.markdown("**Система распознавания российских автомобильных номеров**")

st.sidebar.header("⚙️ Настройки")
conf_threshold = st.sidebar.slider(
    "Порог уверенности детекции", min_value=0.1, max_value=1.0, value=0.25, step=0.05
)

# Настройка фильтрации для видео
min_occurrences = st.sidebar.slider(
    "Минимум появлений номера на видео", 
    min_value=3, max_value=20, value=5, step=1,
    help="Номера, которые появились меньше раз, будут отфильтрованы как ошибочные"
)

input_type = st.sidebar.radio("Тип входных данных:", ["📷 Изображение", "🎥 Видео"])

if input_type == "📷 Изображение":
    uploaded_file = st.file_uploader(
        "Загрузите изображение с автомобилем",
        type=['jpg', 'jpeg', 'png']
    )
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📥 Исходное изображение")
            st.image(image)
        with st.spinner("🔍 Обработка изображения..."):
            processed_image, detected_plates = process_frame(
                image_np.copy(), conf_threshold
            )
        with col2:
            st.subheader("📤 Результат распознавания")
            st.image(processed_image)
        st.markdown("---")
        st.subheader("🎯 Обнаруженные номера")
        if detected_plates:
            for i, plate in enumerate(detected_plates, 1):
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    if plate['valid']:
                        st.success(f"**✓ Номер {i}:** {plate['text']}")
                    else:
                        st.warning(f"**⚠ Номер {i}:** {plate['text']} (не прошёл валидацию)")
                with col2:
                    st.metric("Уверенность", f"{plate['confidence']:.2%}")
                with col3:
                    st.info(f"bbox: {plate['bbox']}")
        else:
            st.warning("⚠️ Номера не обнаружены. Попробуйте изменить порог уверенности.")

elif input_type == "🎥 Видео":
    uploaded_video = st.file_uploader(
        "Загрузите видео с автомобилями", type=['mp4', 'avi', 'mov']
    )
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_video.read())
        video_path = tfile.name
        st.video(video_path)
        process_video_btn = st.button("🎬 Обработать видео")
        if process_video_btn:
            cap = cv2.VideoCapture(video_path)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            output_path = tempfile.NamedTemporaryFile(delete=False, suffix='_output.mp4').name
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Счетчик для каждого номера
            plate_counter = Counter()
            frame_count = 0
            skip_frames = 2
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1
                if frame_count % skip_frames == 0:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    processed_frame, detected_plates = process_frame(
                        frame_rgb, conf_threshold
                    )
                    frame = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)
                    
                    # Подсчет только валидных номеров
                    for plate in detected_plates:
                        if plate['valid'] and plate['text']:
                            plate_counter[plate['text']] += 1
                
                out.write(frame)
                progress = int((frame_count / total_frames) * 100)
                progress_bar.progress(progress)
                status_text.text(f"Обработано кадров: {frame_count}/{total_frames}")
            
            cap.release()
            out.release()
            
            st.success("✅ Видео обработано!")
            st.subheader("📹 Обработанное видео")
            st.video(output_path)
            
            # Фильтрация номеров
            filtered_plates = filter_plates_by_frequency(plate_counter, min_occurrences)
            
            st.markdown("---")
            st.subheader("📊 Статистика распознавания")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Обработано кадров", total_frames)
            with col2:
                st.metric("Всего распознаваний", sum(plate_counter.values()))
            with col3:
                st.metric("Уникальных номеров", len(filtered_plates))
            
            if filtered_plates:
                st.subheader("🎯 Распознанные номера (отфильтрованные)")
                st.info(f"Показаны только номера, которые появились минимум {min_occurrences} раз")
                
                # Сортировка по частоте появления
                sorted_plates = sorted(filtered_plates.items(), key=lambda x: x[1], reverse=True)
                
                for plate, count in sorted_plates:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.success(f"✓ **{plate}**")
                    with col2:
                        st.metric("Появлений", count)
            else:
                st.warning(f"⚠️ Не найдено номеров, появившихся минимум {min_occurrences} раз")
            
            with open(output_path, 'rb') as f:
                st.download_button(
                    label="⬇️ Скачать обработанное видео",
                    data=f,
                    file_name="processed_video.mp4",
                    mime="video/mp4"
                )
            
            # Безопасное удаление временных файлов
            try:
                cap.release()
                out.release()
                os.unlink(video_path)
            except (PermissionError, OSError):
                pass  # Файл используется, будет удален позже
            
            try:
                os.unlink(output_path)
            except (PermissionError, OSError):
                pass

with st.expander("ℹ️ Как использовать"):
    st.markdown("""
    **Для изображений:**
    1. Выберите режим "📷 Изображение"
    2. Загрузите фото с автомобилем
    3. Просмотрите результаты распознавания
    
    **Для видео:**
    1. Выберите режим "🎥 Видео"
    2. Загрузите видеофайл
    3. Настройте минимальное количество появлений номера (для фильтрации ошибок)
    4. Нажмите "🎬 Обработать видео"
    5. Дождитесь завершения и просмотрите результаты
    
    **Система валидации:**
    - ✓ Зеленая рамка — номер прошел валидацию формата
    - ⚠ Оранжевая рамка — номер не соответствует формату российских номеров
    - Формат: 1 буква + 3 цифры + 2 буквы + 2-3 цифры региона
    - Допустимые буквы: A, B, E, K, M, H, O, P, C, T, X, Y
    
    **Фильтрация на видео:**
    - Номера, появившиеся менее N раз, считаются ошибочными и не выводятся
    - Настройте параметр "Минимум появлений" в зависимости от длины видео
    - Для коротких видео (< 100 кадров) — 3-5 появлений
    - Для длинных видео (> 500 кадров) — 10-15 появлений
    """)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 Информация")
st.sidebar.info(
    "**Архитектура:**\n"
    "- YOLOv8n для детекции\n"
    "- CRNN для распознавания\n"
    "- Валидация формата номеров\n"
    "- Фильтрация по частоте\n\n"
)
