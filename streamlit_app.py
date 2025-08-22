def show_polynomial_regression():
    if not SKLEARN_AVAILABLE:
        st.error("Функция прогнозирования недоступна. Установите scikit-learn.")
        return
        
    st.header("📈 Прогнозирование цен на 2026 год")
    
    st.info("""
    В этом разделе строится полиномиальная регрессионная модель для прогнозирования цены за м² 
    на основе выбранных признаков. Модель может быть обучена на всех данных или на отфильтрованных квартирах.
    """)
    
    use_filtered = st.checkbox("Использовать отфильтрованные квартиры из поиска", 
                              value=st.session_state.filtered_apartments is not None,
                              disabled=st.session_state.filtered_apartments is None)
    
    if use_filtered and st.session_state.filtered_apartments is not None:
        analysis_data = st.session_state.filtered_apartments
        st.info(f"Используются отфильтрованные данные: {len(analysis_data)} квартир")
    else:
        analysis_data = data
        st.info("Используются все данные")
    
    numeric_cols = analysis_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = analysis_data.select_dtypes(include=['object']).columns.tolist()
    
    target_col = st.selectbox("Целевая переменная (y):", 
                             options=['Цена кв m', 'Цена'],
                             index=0)
    
    if target_col not in analysis_data.columns:
        st.error(f"Колонка '{target_col}' не найдена в данных!")
        return
    
    # Фиксированный список признаков как вы указали
    fixed_features = [
        'Школа/Детский Сад', 
        'Парк/Зона отдыха', 
        'Спорт', 
        'Парковка', 
        'Рестораны', 
        'Комнат', 
        'Площадь',
        'Этаж', 
        'Район Город', 
        'Класс К....', 
        'Застройщик'
    ]
    
    # Оставляем только те признаки, которые есть в данных
    available_features = [col for col in fixed_features if col in analysis_data.columns]
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.write("Выбранные признаки:")
        st.write(", ".join(available_features))
    
    if not available_features:
        st.warning("Нет доступных признаков для построения модели")
        return
    
    temp_data = analysis_data[available_features + [target_col]].copy()
    initial_count = len(temp_data)
    temp_data = temp_data.dropna()
    final_count = len(temp_data)
    
    if final_count == 0:
        st.error("""
        ❌ После обработки пропусков в выбранных признаках не осталось данных!
        
        **Рекомендации:**
        1. Проверьте наличие данных в выбранных признаках.
        """)
        return
    
    st.info(f"Данные для обучения: {final_count} из {initial_count} записей (удалено {initial_count - final_count} записей с пропусками)")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        degree = st.slider("Степень полинома", 1, 5, 2)
    with col2:
        test_size = st.slider("Размер тестовой выборки", 0.1, 0.5, 0.3)
    with col3:
        random_state = st.number_input("Random state", 0, 100, 42)
    
    if st.button("Обучить модель", type='primary'):
        try:
            X = temp_data[available_features].copy()
            y = temp_data[target_col]
            
            numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_features = X.select_dtypes(include=['object']).columns.tolist()
            
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_features),
                    ('cat', categorical_transformer, categorical_features)
                ])
            
            full_pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('poly', PolynomialFeatures(degree=degree, include_bias=False))
            ])
            
            X_processed = full_pipeline.fit_transform(X)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y, test_size=test_size, random_state=random_state
            )
            
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
            st.session_state.model_features = available_features
            st.session_state.target_column = target_col
            
            all_predictions = model.predict(full_pipeline.transform(X))
            
            forecast_df = analysis_data.loc[X.index, ['Номер квартиры'] + available_features].copy()
            forecast_df['Фактическая цена'] = y
            forecast_df['Прогноз на 2026 год'] = all_predictions
            forecast_df['Изменение, %'] = ((forecast_df['Прогноз на 2026 год'] - forecast_df['Фактическая цена']) / 
                                         forecast_df['Фактическая цена'] * 100)
            
            st.subheader("📊 Прогноз цен на 2026 год")
            st.dataframe(
                forecast_df.style.format({
                    'Фактическая цена': '{:,.0f}',
                    'Прогноз на 2026 год': '{:,.0f}',
                    'Изменение, %': '{:.1f}%'
                }),
                height=400
            )
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("R² Score", f"{r2:.3f}")
            with col2:
                st.metric("RMSE", f"{rmse:.2f}")
            with col3:
                st.metric("MAE", f"{mae:.2f}")
            with col4:
                st.metric("Обучено на", f"{len(X_train)} samples")
            
            if MATPLOTLIB_AVAILABLE:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
                
                ax1.scatter(y_test, y_pred, alpha=0.5)
                ax1.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
                ax1.set_xlabel('Фактические значения')
                ax1.set_ylabel('Предсказанные значения')
                ax1.set_title('Фактические vs Предсказанные значения')
                
                residuals = y_test - y_pred
                ax2.scatter(y_pred, residuals, alpha=0.5)
                ax2.axhline(y=0, color='r', linestyle='--')
                ax2.set_xlabel('Предсказанные значения')
                ax2.set_ylabel('Остатки')
                ax2.set_title('Анализ остатков')
                
                plt.tight_layout()
                st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Ошибка при обучении модели: {str(e)}")
