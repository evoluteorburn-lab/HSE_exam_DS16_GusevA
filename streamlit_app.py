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
    st.warning("Matplotlib не установлен. Некоторые графики будут недоступны.")

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
    st.error("Библиотека scikit-learn не установлена. Функция прогнозирования будет недоступна.")

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
        return None

data = load_data_from_github()

if data is None:
    st.error("Не удалось загрузить данные. Пожалуйста, проверьте подключение к интернету.")
    st.stop()

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

query_params = st.query_params
initial_section = query_params.get("section", ["Поиск квартир"])[0]
if initial_section not in section_options:
    initial_section = "Поиск квартир"

section = st.sidebar.radio("Выберите раздел:", section_options, index=section_options.index(initial_section))

if 'filtered_apartments' not in st.session_state:
    st.session_state.filtered_apartments = None
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'model_features' not in st.session_state:
    st.session_state.model_features = None
if 'target_column' not in st.session_state:
    st.session_state.target_column = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

def show_apartment_search():
    st.header("🔍 Поиск квартир по параметрам")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Основные параметры")
        
        def get_unique_values(column_name):
            if column_name in data.columns and not data[column_name].empty:
                unique_vals = data[column_name].dropna().unique().tolist()
                return sorted([str(x) for x in unique_vals if x is not None and x != ''])
            return []
        
        class_options = get_unique_values('Класс К....')
        class_input = st.selectbox('Класс квартиры', options=[None] + class_options)
        
        area_min = st.number_input('Площадь от (м²)', min_value=0.0, value=0.0, step=1.0)
        area_max = st.number_input('Площадь до (м²)', min_value=0.0, value=0.0, step=1.0)
        if area_max > 0 and area_min > area_max:
            st.error("Максимальная площадь не может быть меньше минимальной")
    
    with col2:
        st.subheader("Дополнительные параметры")
        
        rooms_options = get_unique_values('Комнат')
        rooms_input = st.selectbox('Комнат', options=[None] + rooms_options)
        
        floor_min = st.number_input('Этаж от', min_value=0, value=0)
        floor_max = st.number_input('Этаж до', min_value=0, value=0)
        if floor_max > 0 and floor_min > floor_max:
            st.error("Максимальный этаж не может быть меньше минимального")
        
        district_options = get_unique_values('Район Город')
        district_input = st.selectbox('Район', options=[None] + district_options)
        
        builder_options = get_unique_values('Застройщик')
        builder_input = st.selectbox('Застройщик', options=[None] + builder_options)
    
    st.subheader("🏗️ Инфраструктура (диапазон значений)")
    infra_cols = st.columns(5)
    infrastructure_options = {}
    
    infra_columns = ['Школа/Детский Сад', 'Парк/Зона отдыха', 'Спорт', 'Парковка', 'Рестораны']
    
    for i, col_name in enumerate(infra_columns):
        if col_name in data.columns:
            numeric_values = pd.to_numeric(data[col_name], errors='coerce').dropna()
            if not numeric_values.empty:
                min_val = numeric_values.min()
                max_val = numeric_values.max()
                
                with infra_cols[i]:
                    st.write(f"**{col_name}**")
                    infra_min = st.number_input(f'{col_name} от', min_value=float(min_val), 
                                              max_value=float(max_val), value=float(min_val), 
                                              key=f"{col_name}_min")
                    infra_max = st.number_input(f'{col_name} до', min_value=float(min_val), 
                                              max_value=float(max_val), value=float(max_val), 
                                              key=f"{col_name}_max")
                    infrastructure_options[col_name] = (infra_min, infra_max)
    
    if st.button('Найти квартиры', type='primary'):
        filtered_df = data.copy()
        
        if class_input and 'Класс К....' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['Класс К....'] == class_input]
        
        if area_min > 0:
            filtered_df = filtered_df[filtered_df['Площадь'] >= area_min]
        if area_max > 0:
            filtered_df = filtered_df[filtered_df['Площадь'] <= area_max]
        
        if rooms_input:
            filtered_df = filtered_df[filtered_df['Комнат'] == int(rooms_input)]
        
        if floor_min > 0:
            filtered_df = filtered_df[filtered_df['Этаж'] >= floor_min]
        if floor_max > 0:
            filtered_df = filtered_df[filtered_df['Этаж'] <= floor_max]
        
        if district_input:
            filtered_df = filtered_df[filtered_df['Район Город'] == district_input]
        
        if builder_input:
            filtered_df = filtered_df[filtered_df['Застройщик'] == builder_input]
        
        for col_name, (min_val, max_val) in infrastructure_options.items():
            if col_name in filtered_df.columns:
                filtered_df[col_name] = pd.to_numeric(filtered_df[col_name], errors='coerce')
                filtered_df = filtered_df[
                    (filtered_df[col_name] >= min_val) & 
                    (filtered_df[col_name] <= max_val)
                ]
        
        if len(filtered_df) == 0:
            st.warning("Не найдено объектов по указанным параметрам")
            st.session_state.filtered_apartments = None
        else:
            st.success(f"Найдено {len(filtered_df)} объектов")
            st.session_state.filtered_apartments = filtered_df
            
            # Отображение статистики по цене за квадратный метр
            if 'Цена кв м' in filtered_df.columns:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Максимальная цена за м²", f"{filtered_df['Цена кв м'].max():,.0f} руб.")
                with col2:
                    st.metric("Средняя цена за м²", f"{filtered_df['Цена кв м'].mean():,.0f} руб.")
                with col3:
                    st.metric("Минимальная цена за м²", f"{filtered_df['Цена кв м'].min():,.0f} руб.")
            
            # Отображение статистики по общей цене квартиры
            if 'Цена' in filtered_df.columns:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Максимальная цена квартиры", f"{filtered_df['Цена'].max():,.0f} руб.")
                with col2:
                    st.metric("Средняя цена квартиры", f"{filtered_df['Цена'].mean():,.0f} руб.")
                with col3:
                    st.metric("Минимальная цена квартиры", f"{filtered_df['Цена'].min():,.0f} руб.")
            
            display_columns = ['Номер квартиры', 'Площадь', 'Комнат', 'Этаж', 'Район Город', 'Цена', 'Цена кв м', 'Класс К....']
            display_columns.extend([col for col in infra_columns if col in filtered_df.columns])
            
            available_columns = [col for col in display_columns if col in filtered_df.columns]
            
            display_df = filtered_df[available_columns].copy()
            display_df.rename(columns={
                'Класс К....': 'Класс',
                'Район Город': 'Район',
                'Цена кв м': 'Цена за м²',
                'Номер квартиры': '№ Квартиры'
            }, inplace=True)
            
            st.dataframe(
                display_df.style.format({
                    'Цена': '{:,.0f} руб.',
                    'Цена за м²': '{:,.0f} руб.',
                    'Площадь': '{:.1f} м²',
                    '№ Квартиры': '{:.0f}'
                }),
                height=400
            )

def show_polynomial_regression():
    if not SKLEARN_AVAILABLE:
        st.error("Функция прогнозирования недоступна. Установите scikit-learn.")
        return
        
    st.header("📈 Прогнозирование цен")
    
    if st.session_state.filtered_apartments is None:
        st.info("Используются все данные для прогнозирования")
        analysis_data = data
    else:
        st.info(f"Используются отфильтрованные данные: {len(st.session_state.filtered_apartments)} квартир")
        analysis_data = st.session_state.filtered_apartments
    
    numeric_cols = analysis_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = analysis_data.select_dtypes(include=['object']).columns.tolist()
    
    target_col = st.selectbox("Целевая переменная (y):", 
                             options=['Цена кв м', 'Цена'],
                             index=0)
    
    if target_col not in analysis_data.columns:
        st.error(f"Колонка '{target_col}' не найдена в данных!")
        return
    
    price_columns_to_exclude = ['Цена кв м', 'Цена', 'Изменение цены последнее', 'Изменение цены', 'Цена со скидкой']
    available_features = [col for col in numeric_cols + categorical_cols 
                         if col not in price_columns_to_exclude and col in analysis_data.columns]
    
    available_numeric = [col for col in available_features if col in numeric_cols]
    available_categorical = [col for col in available_features if col in categorical_cols]
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        selected_features = st.multiselect("Признаки для модели (X):", 
                                          options=available_features,
                                          default=['Площадь', 'Комнат', 'Этаж'])
    
    with col2:
        st.write("")
        st.write("")
        
        if st.button("Все признаки", key="select_all_btn"):
            selected_features = available_features
            st.rerun()
        
        if st.button("Очистить", key="clear_btn"):
            selected_features = []
            st.rerun()
    
    if not selected_features:
        st.warning("Выберите хотя бы один признак для построения модели")
        return
    
    col1, col2, col3 = st.columns(3)
    with col1:
        degree = st.slider("Степень полинома", 1, 3, 2)
    with col2:
        test_size = st.slider("Размер тестовой выборки", 0.1, 0.5, 0.3)
    with col3:
        random_state = st.number_input("Random state", 0, 100, 42)
    
    if st.button("Обучить модель", type='primary'):
        try:
            missing_features = [col for col in selected_features if col not in analysis_data.columns]
            if missing_features:
                st.error(f"Следующие признаки не найдены в данных: {missing_features}")
                return
            
            X = analysis_data[selected_features].copy()
            y = analysis_data[target_col]
            
            X_clean = X.dropna()
            y_clean = y.loc[X_clean.index]
            
            if len(X_clean) == 0:
                st.error("После обработки пропусков не осталось данных для обучения!")
                return
            
            if len(X_clean) < 10:
                st.warning(f"Мало данных для обучения: всего {len(X_clean)} строк. Результаты могут быть ненадежными.")
            
            numeric_features = X_clean.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_features = X_clean.select_dtypes(include=['object']).columns.tolist()
            
            if not numeric_features and not categorical_features:
                st.error("Не осталось признаков для обучения после обработки данных!")
                return
            
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            
            transformers = []
            if numeric_features:
                transformers.append(('num', numeric_transformer, numeric_features))
            if categorical_features:
                transformers.append(('cat', categorical_transformer, categorical_features))
            
            preprocessor = ColumnTransformer(transformers=transformers)
            
            full_pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('poly', PolynomialFeatures(degree=degree, include_bias=False))
            ])
            
            X_processed = full_pipeline.fit_transform(X_clean)
            
            if X_processed.shape[0] == 0:
                st.error("После преобразований не осталось данных для обучения!")
                return
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y_clean, test_size=test_size, random_state=random_state
            )
            
            if len(X_test) == 0:
                st.error("Тестовая выборка пустая! Уменьшите размер тестовой выборки.")
                return
            
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            st.success("Модель успешно обучена!")
            
            st.session_state.trained_model = model
            st.session_state.model_pipeline = full_pipeline
            st.session_state.model_features = selected_features
            st.session_state.target_column = target_col
            
            st.session_state.analysis_results = {
                'train_samples': len(X_train),
                'r2_score': r2,
                'rmse': rmse,
                'mae': mae
            }
            
            try:
                all_predictions = model.predict(full_pipeline.transform(X))
                
                forecast_df = analysis_data[['Номер квартиры'] + selected_features].copy()
                forecast_df['Фактическая цена'] = analysis_data[target_col]
                forecast_df['Прогноз'] = all_predictions
                forecast_df['Изменение, %'] = ((forecast_df['Прогноз'] - forecast_df['Фактическая цена']) / 
                                             forecast_df['Фактическая цена'] * 100)
                
                st.subheader("📊 Результаты прогнозирования")
                st.dataframe(
                    forecast_df.style.format({
                        'Фактическая цена': '{:,.0f}',
                        'Прогноз': '{:,.0f}',
                        'Изменение, %': '{:.1f}%'
                    }),
                    height=400
                )
                
            except Exception as e:
                st.warning(f"Не удалось сделать прогноз для всех данных: {e}")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("R² Score", f"{r2:.3f}")
            with col2:
                st.metric("RMSE", f"{rmse:.2f}")
            with col3:
                st.metric("MAE", f"{mae:.2f}")
            with col4:
                st.metric("Обучено на", f"{len(X_train)} samples")
            
        except Exception as e:
            st.error(f"Ошибка при обучении модели: {str(e)}")
            st.info("Попробуйте выбрать меньше признаков или уменьшить степень полинома")

if section == "Поиск квартир":
    show_apartment_search()
elif section == "Прогнозирование" and SKLEARN_AVAILABLE:
    show_polynomial_regression()
