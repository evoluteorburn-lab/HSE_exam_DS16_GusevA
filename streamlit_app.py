import warnings
warnings.filterwarnings("ignore", message="Thread 'MainThread': missing ScriptRunContext!")
warnings.filterwarnings("ignore", message="No runtime found, using MemoryCacheStorageManager")

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import requests
from io import BytesIO

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

try:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler, OneHotEncoder
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

st.set_page_config(
    page_title="Анализ рынка недвижимости - Полный анализ",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data(ttl=3600)
def load_data_from_github():
    github_url = "https://github.com/evoluteorburn-lab/HSE_exam_DS16_GusevA/raw/5a932eac450c1cc46fd032db7deb0fc9dd86843b/Cian.xlsx"
    try:
        response = requests.get(github_url)
        response.raise_for_status()
        file_content = BytesIO(response.content)
        df = pd.read_excel(file_content)
        if 'Номер квартиры' not in df.columns:
            df['Номер квартиры'] = range(1, len(df) + 1)
        return df
    except Exception as e:
        st.error(f"Ошибка при загрузке данных: {e}")
        return pd.DataFrame()

data = load_data_from_github()

st.title("🏠 Комплексный анализ рынка недвижимости")
st.write(f"Данные загружены: {len(data)} строк, {len(data.columns)} колонок")

if not SKLEARN_AVAILABLE:
    st.error("⚠️ Библиотека scikit-learn не установлена. Функция прогнозирования будет недоступна.")

if not MATPLOTLIB_AVAILABLE:
    st.warning("⚠️ Библиотека matplotlib не установлена. Некоторые графики будут недоступны.")

st.sidebar.title("Навигация")
section_options = ["Поиск квартир"]
if SKLEARN_AVAILABLE:
    section_options.append("Прогнозирование")

section = st.sidebar.radio("Выберите раздел:", section_options)

if 'filtered_apartments' not in st.session_state:
    st.session_state.filtered_apartments = None
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'model_features' not in st.session_state:
    st.session_state.model_features = None
if 'target_column' not in st.session_state:
    st.session_state.target_column = None

def show_apartment_search():
    st.header("🔍 Поиск квартир по параметрам")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Основные параметры")

        def get_unique_values(column_name, default_options=None):
            if column_name in data.columns and not data[column_name].empty:
                unique_vals = data[column_name].dropna().unique().tolist()
                return sorted([x for x in unique_vals if x is not None and x != ''])
            return default_options if default_options else []

        class_options = get_unique_values('Класс К....', ['Эконом', 'Комфорт', 'Бизнес', 'Премиум'])
        class_input = st.selectbox('Класс квартиры', options=[None] + class_options)

        area_min = st.number_input('Площадь от (м²)', min_value=0.0, value=0.0)
        area_max = st.number_input('Площадь до (м²)', min_value=0.0, value=0.0)

    with col2:
        st.subheader("Дополнительные параметры")

        rooms_options = get_unique_values('Комнат', [1, 2, 3, 4, 5])
        rooms_input = st.selectbox('Комнат', options=[None] + rooms_options)

        floor_min = st.number_input('Этаж от', min_value=0, value=0)
        floor_max = st.number_input('Этаж до', min_value=0, value=0)

        district_options = get_unique_values('Район Город', ['ЦАО', 'САО', 'ЮАО', 'ЗАО', 'СВАО', 'ЮЗАО', 'ВАО'])
        district_input = st.selectbox('Район', options=[None] + district_options)

        builder_options = get_unique_values('Застройщик', ['ПИК', 'Самолет', 'Эталон'])
        builder_input = st.selectbox('Застройщик', options=[None] + builder_options)

    st.subheader("🏗️ Инфраструктура")
    infra_cols = st.columns(6)
    infrastructure_options = {}
    infra_columns = ['Школа/Детский Сад', 'Парк/Зона отдыха', 'Спорт', 'Парковка', 'Рестораны', 'Метро']

    for i, col_name in enumerate(infra_columns):
        if col_name in data.columns:
            options = get_unique_values(col_name, [])
            with infra_cols[i]:
                infrastructure_options[col_name] = st.selectbox(
                    col_name,
                    options=[None] + options,
                    key=f"infra_{col_name}"
                )

    if st.button('Найти квартиры', type='primary'):
        filtered_df = data.copy()

        if class_input and 'Класс К....' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['Класс К....'] == class_input]

        if area_min > 0:
            filtered_df = filtered_df[filtered_df['Площадь'] >= area_min]
        if area_max > 0:
            filtered_df = filtered_df[filtered_df['Площадь'] <= area_max]

        if rooms_input:
            filtered_df = filtered_df[filtered_df['Комнат'] == rooms_input]

        if floor_min > 0:
            filtered_df = filtered_df[filtered_df['Этаж'] >= floor_min]
        if floor_max > 0:
            filtered_df = filtered_df[filtered_df['Этаж'] <= floor_max]

        if district_input:
            filtered_df = filtered_df[filtered_df['Район Город'] == district_input]

        if builder_input:
            filtered_df = filtered_df[filtered_df['Застройщик'] == builder_input]

        for col_name, value in infrastructure_options.items():
            if value and col_name in filtered_df.columns:
                filtered_df = filtered_df[filtered_df[col_name] == value]

        if len(filtered_df) == 0:
            st.warning("Не найдено объектов по указанным параметрам")
            st.session_state.filtered_apartments = None
        else:
            st.success(f"Найдено {len(filtered_df)} объектов")
            st.session_state.filtered_apartments = filtered_df

            if 'Цена кв м' in filtered_df.columns and 'Цена' in filtered_df.columns:
                col1, col2, col3, col4, col5, col6 = st.columns(6)
                with col1:
                    st.metric("Средняя цена за м²", f"{filtered_df['Цена кв м'].mean():,.0f} руб.")
                with col2:
                    st.metric("Минимальная цена за м²", f"{filtered_df['Цена кв м'].min():,.0f} руб.")
                with col3:
                    st.metric("Максимальная цена за м²", f"{filtered_df['Цена кв м'].max():,.0f} руб.")
                with col4:
                    st.metric("Средняя цена квартиры", f"{filtered_df['Цена'].mean():,.0f} руб.")
                with col5:
                    st.metric("Минимальная цена квартиры", f"{filtered_df['Цена'].min():,.0f} руб.")
                with col6:
                    st.metric("Максимальная цена квартиры", f"{filtered_df['Цена'].max():,.0f} руб.")

            display_columns = ['Номер квартиры', 'Площадь', 'Комнат', 'Этаж', 'Район Город', 'Цена кв м', 'Класс К....']
            display_columns.extend([col for col in infra_columns if col in filtered_df.columns])
            available_columns = [col for col in display_columns if col in filtered_df.columns]

            display_df = filtered_df[available_columns].copy()
            display_df.rename(columns={
                'Класс К....': 'Класс',
                'Район Город': 'Район',
                'Цена кв м': 'Цена за м²',
                'Номер квартиры': '№ Квартиры',
                'Метро': 'Метро'
            }, inplace=True)

            st.dataframe(
                display_df.style.format({
                    'Цена за м²': '{:,.0f} руб.',
                    'Площадь': '{:.1f} м²',
                    '№ Квартиры': '{:.0f}'
                }),
                height=400
            )

            if SKLEARN_AVAILABLE:
                if st.button("Перейти к прогнозированию ➡️", type="secondary"):
                    st.session_state.goto_forecast = True

def show_polynomial_regression():
    if not SKLEARN_AVAILABLE:
        st.error("Функция прогнозирования недоступна. Установите scikit-learn.")
        return

    st.header("📈 Прогнозирование цен на 2026 год")

    use_filtered = st.checkbox(
        "Использовать отфильтрованные квартиры из поиска",
        value=st.session_state.filtered_apartments is not None,
        disabled=st.session_state.filtered_apartments is None
    )

    if use_filtered and st.session_state.filtered_apartments is not None:
        analysis_data = st.session_state.filtered_apartments
        st.info(f"Используются отфильтрованные данные: {len(analysis_data)} квартир")
    else:
        analysis_data = data
        st.info("Используются все данные")

    numeric_cols = analysis_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = analysis_data.select_dtypes(include=['object']).columns.tolist()

    target_col = st.selectbox("Целевая переменная (y):", options=['Цена кв м', 'Цена'], index=0)

    if target_col not in analysis_data.columns:
        st.error(f"Колонка '{target_col}' не найдена в данных!")
        return

    available_features = [col for col in numeric_cols + categorical_cols if col != target_col and col != 'Номер квартиры']
    exclude = ['Цена кв м', 'Цена', 'Цена со скидкой', 'Изменение цены последнее', 'Изменение цены',
               'ID ЖК','ID Корпуса','ЖК рус','ЖК англ','Корпус','кр Корпус','Регион','ID кв','Дата актуализации',
               'Номер на этаже','Номер в корпусе','Номер секции','Адрес корп','lat','lng','Округ Направление',
               'АТД','Источник','Тип корпуса','Тип кв/ап','Тип помещения','Отделка помещения','Отделка К','Договор К',
               'Сдача К','Зона','Отделка текст','Старт продаж К','Изменение цены последнее','Экспозиция','Изменение цены']
    available_features = [col for col in available_features if col not in exclude and col in analysis_data.columns]

    default_feats = [c for c in ['Площадь', 'Комнат', 'Этаж'] if c in available_features]
    if 'selected_features' not in st.session_state:
        st.session_state.selected_features = default_feats

    col1, col2 = st.columns([3, 1])

    with col2:
        st.write(""); st.write("")
        if st.button("Выбрать все доступные", key="btn_select_all"):
            st.session_state.selected_features = [
                f for f in available_features if f in analysis_data.columns and analysis_data[f].notna().any()
            ]

    with col1:
        selected_features = st.multiselect(
            "Признаки для модели (X):",
            options=available_features,
            key="selected_features"
        )

    temp_data = analysis_data[selected_features + [target_col]].copy()
    temp_data = temp_data.dropna()
    if temp_data.empty:
        st.error("❌ После обработки данные оказались пустыми, проверьте признаки")
