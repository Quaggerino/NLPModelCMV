# ****Entrenamiento de Modelo de Clasificación de Sentimiento con BERT****

### Guía de Entrenamiento de Modelo de Clasificación de Sentimiento con BERT

Esta guía describe los pasos necesarios para entrenar un modelo de clasificación de sentimiento utilizando BERT (Bidirectional Encoder Representations from Transformers). Se utiliza una configuración con Python y PyTorch, y el modelo se entrena con un conjunto de datos que contiene opiniones en español.

### Preparación del Entorno

1. **Importaciones Iniciales:**
    - Asegúrese de importar todas las bibliotecas necesarias antes de comenzar el entrenamiento.
    
    ```python
    import multiprocessing
    import torch
    import pandas as pd
    from transformers import BertTokenizer, BertForSequenceClassification
    # ...otros imports...
    
    ```
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/05f7aede-7011-479c-99d5-395a9bffa1f7/589b4067-1deb-4e04-967a-fa9c406f5330/Untitled.png)
    
2. **Configuración de Multiprocessing y CUDA:**
    - Establezca el método de inicio de `multiprocessing` y verifique la disponibilidad de CUDA para usar la GPU.
    
    ```python
    multiprocessing.set_start_method('spawn', True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    
    ```
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/05f7aede-7011-479c-99d5-395a9bffa1f7/beec8851-68ca-4a5f-8293-ecd132c7cd23/Untitled.png)
    

### Preparación de Datos

1. **Carga y Limpieza:**
    - Cargue los datos desde un archivo CSV y realice una limpieza inicial, como dividir textos largos.
    
    ```python
    df = pd.read_csv('combinado.csv')
    df = split_long_texts(df)
    
    ```
    

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/05f7aede-7011-479c-99d5-395a9bffa1f7/c923ab09-bca8-45b0-bf9d-0394835f903f/Untitled.png)

1. **Tokenización y Dataset:**
    - Utilice `BertTokenizer` para tokenizar los datos y prepare un `Dataset` personalizado para PyTorch.
    
    ```python
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    dataset = CustomDataset(df, tokenizer, MAX_LEN)
    
    ```
    

### Configuración del Modelo

1. **Carga del Modelo Preentrenado:**
    - Cargue BERT con la configuración necesaria para clasificación de secuencia.
    
    ```python
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
    model.to(device)
    
    ```
    
2. **Optimizador y Scheduler:**
    - Establezca el optimizador y el scheduler para el entrenamiento del modelo.
    
    ```python
    optimizer = AdamW(model.parameters(), lr=2e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(df)*3)
    
    ```
    

### Proceso de Entrenamiento

1. **Entrenamiento por Épocas:**
    - Entrene el modelo por varias épocas, registrando y ajustando los parámetros según sea necesario.
    
    ```python
    for epoch in range(EPOCHS):
        # Entrenamiento y validación...
        print(f"Epoch: {epoch+1}/{EPOCHS}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
    
    ```
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/05f7aede-7011-479c-99d5-395a9bffa1f7/4e62a19c-c8d8-4efd-82bf-0c17caafe19e/Untitled.png)
    
2. **Evaluación y Ajustes:**
    - Evalúe el modelo con el conjunto de validación para determinar la precisión.
    
    ```python
    train_accuracy, train_loss = train_epoch(...)
    val_accuracy, val_loss = eval_model(...)
    ```
    

### Predicción y Evaluación

1. **Función de Predicción:**
    - Desarrolle una función para predecir los sentimientos de nuevos textos.
    
    ```python
    predictions = predict_sentiments(test_texts, model, tokenizer, MAX_LEN)
    
    ```
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/05f7aede-7011-479c-99d5-395a9bffa1f7/1cb9b1b7-2914-4cd8-ab68-0106be0d3b1c/Untitled.png)
    
    > 'excelente servicio del cirujano medico' is predicted as Positivo.
    
    'pesima atención, la gente del SOME pasa viendo el teléfono y no atiende a la gente' is predicted as Negativo.
    
    'no hay hora para matrona y no me dan solución de hora' is predicted as Negativo.
    
    'el cesfam esta bonito pero atienden mal' is predicted as Negativo.
    
    'alcade saque a su amigo del cesfam' is predicted as Irrelevante.
    
    'alguna vecina q este el el consultorio y le diga q respondan el teléfono porfa q estoy de la mañana comunicandome' is predicted as Negativo.
    
    'Hola les comentamos que somos una agrupación de zumba de valparaíso' is predicted as Irrelevante.
    
    'Me hecho 3 mamografías 2 ecosmamarias porq me sale un nódulo de 10cm pero después de la operación de vesícula en solo semanas creció 5 centímetros y ahora solo esperar q me llamen de salud pública por no tener dinero tengo q esperar q me llame el especialista para la biopsia 😔' is predicted as Negativo.
    
    'Recien a los 45 me empiezan hacer la mamografia preventiva en el cesfam... Como puedo optar a ella antes??' is predicted as Irrelevante.
    
    'Programas disponibles? Donde? Si es casi imposible llegar a salud mental a través de la salud pública, a no ser que uno intente suicidarse 🤷🏾‍♀️ La concientización no sirve si no existen los medios.' is predicted as Negativo.
    
    'Excelente iniciativa 🙌🏽❤️ Visibilizar y hacernos responsables como sociedad es fundamental' is predicted as Positivo.
    
    'No me parece que se ocupen dineros públicos en tratamientos que no tienen evidencia científica' is predicted as Irrelevante.
    
    'Ese Director del CESFAM atendía a mi hijo en el Mena. Qué lindo que alguien como él este en ese cargo 🙌' is predicted as Positivo.
    
    'Que buena noticia, estos enfoques visualiza el todo y no las partes y posibilita un auto análisis para ser responsable de la promoción y prevención de la salud. Espero se puedan abrir más espacios así, siempre existieran personas que les sea de mucha utilidad ❤️' is predicted as Positivo.
    
    'Medicina alternativa! Excelente 👏👏👏' is predicted as Positivo.
    
    'Y los que tienen horas agendada?' is predicted as Irrelevante.
    
    'Los felicito, estoy feliz, buen servicio' is predicted as Positivo.
    > 
2. **Resultados y Afinamiento:**
    - Interprete los resultados, ajuste el modelo si es necesario y realice predicciones sobre datos no vistos.
    
    ```python
    sentiments, probabilities = predict_sentiments(test_texts, test_targets, model, tokenizer, MAX_LEN)
    
    ```
    
    > Precisión general: 92.54%
    
    Correctos: 62 ✅
    Incorrectos: 5 ❌
    Advertencias (Confianza baja): 7 ⚠️
    
    'Me hecho 3 mamografías 2 ecosmamarias porq me sale un nódulo de 10cm pero después de la operación de vesícula en solo semanas creció 5 centímetros y ahora solo esperar q me llamen de salud pública por no tener dinero tengo q esperar q me llame el especialista para la biopsia 😔'
    Fue predecido como: 'Negativo' con: '99.93% de confianza'.
    La etiqueta correcta es: 'Negativo'.
    Info: ✅
    
    'Recien a los 45 me empiezan hacer la mamografia preventiva en el cesfam... Como puedo optar a ella antes??'
    Fue predecido como: 'Irrelevante' con: '75.78% de confianza'.
    La etiqueta correcta es: 'Irrelevante'.
    Info: ✅ ⚠️
    
    'Programas disponibles? Donde? Si es casi imposible llegar a salud mental a través de la salud pública, a no ser que uno intente suicidarse 🤷🏾‍♀️ La concientización no sirve si no existen los medios.'
    Fue predecido como: 'Negativo' con: '99.63% de confianza'.
    La etiqueta correcta es: 'Negativo'.
    Info: ✅
    
    'Excelente iniciativa 🙌🏽❤️ Visibilizar y hacernos responsables como sociedad es fundamental'
    Fue predecido como: 'Positivo' con: '99.94% de confianza'.
    La etiqueta correcta es: 'Positivo'.
    Info: ✅
    
    'Por eso aumenten la contratación de matronas en Cefam!!! En @cesfamreinaisabeliioficial llevo meses esperando que me vean, me imagino como debe ser para las mujeres mayores,peor todavia!! @cmvalparaiso'
    Fue predecido como: 'Negativo' con: '99.89% de confianza'.
    La etiqueta correcta es: 'Negativo'.
    Info: ✅
    
    'No me parece que se ocupen dineros públicos en tratamientos que no tienen evidencia científica'
    Fue predecido como: 'Irrelevante' con: '99.95% de confianza'.
    La etiqueta correcta es: 'Irrelevante'.
    Info: ✅
    
    'Ese Director del CESFAM atendía a mi hijo en el Mena. Qué lindo que alguien como él este en ese cargo 🙌'
    Fue predecido como: 'Positivo' con: '99.95% de confianza'.
    La etiqueta correcta es: 'Positivo'.
    Info: ✅
    
    'Que buena noticia, estos enfoques visualiza el todo y no las partes y posibilita un auto análisis para ser responsable de la promoción y prevención de la salud. Espero se puedan abrir más espacios así, siempre existieran personas que les sea de mucha utilidad ❤️'
    Fue predecido como: 'Positivo' con: '99.62% de confianza'.
    La etiqueta correcta es: 'Positivo'.
    Info: ✅
    
    'Medicina alternativa! Excelente 👏👏👏'
    Fue predecido como: 'Positivo' con: '99.87% de confianza'.
    La etiqueta correcta es: 'Positivo'.
    Info: ✅
    
    'Y los que tienen horas agendada?'
    Fue predecido como: 'Irrelevante' con: '99.98% de confianza'.
    La etiqueta correcta es: 'Irrelevante'.
    Info: ✅
    
    'Peguense la cacha y don respuesta a los miles de porteros que hoy están sufriendo de trombosis, acv, paros cardiaco, turbocancer por sus famosas vacunas...Los datos los pueden encontrar en deis.minsal.cl'
    Fue predecido como: 'Negativo' con: '99.96% de confianza'.
    La etiqueta correcta es: 'Negativo'.
    Info: ✅
    
    'Felicitaciones por su gran labor y el cariño que siempre ponen !!!'
    Fue predecido como: 'Positivo' con: '99.96% de confianza'.
    La etiqueta correcta es: 'Positivo'.
    Info: ✅
    
    'Recordar siempre que el consentimiento se puede quitar en cualquier instante de una relación sexual 🙏🏽'
    Fue predecido como: 'Irrelevante' con: '97.29% de confianza'.
    La etiqueta correcta es: 'Irrelevante'.
    Info: ✅
    
    'Los funcionarios tambien necesitamos atencion en salud mental.. para entregar un servicio de calidad tambien debemos estar bien fisica y psicologicamente..'
    Fue predecido como: 'Negativo' con: '99.80% de confianza'.
    La etiqueta correcta es: 'Negativo'.
    Info: ✅
    
    'Trabajo en educación y a diario se ve lo saturado de los centros de salud familiar, tienen lista de espera enormes y niños y niñas sin atención ! Nadie se hace cargo'
    Fue predecido como: 'Negativo' con: '99.90% de confianza'.
    La etiqueta correcta es: 'Negativo'.
    Info: ✅
    
    'Que lindo los videos, pero feo es ver a adultos mayores estar desde las 8:00hrs a las 17:00hrs esperando por renovar su licencia'
    Fue predecido como: 'Negativo' con: '99.53% de confianza'.
    La etiqueta correcta es: 'Negativo'.
    Info: ✅
    
    'La directora del cesfam Rodelillo, podría pedir que mejoren la atención. Horrible cómo atienden a las personas.'
    Fue predecido como: 'Negativo' con: '99.59% de confianza'.
    La etiqueta correcta es: 'Negativo'.
    Info: ✅
    
    'Cual de todos maneja más mal el Cesfam de los sectores señalados… Placeres vale callampa desde hace por los menos 20 años…'
    Fue predecido como: 'Negativo' con: '98.96% de confianza'.
    La etiqueta correcta es: 'Negativo'.
    Info: ✅
    
    'el de esperanza es de viña?'
    Fue predecido como: 'Irrelevante' con: '99.96% de confianza'.
    La etiqueta correcta es: 'Irrelevante'.
    Info: ✅
    
    'Planilla...tuve una urgencia a las 22:30 hasta y el DOCTOR DE TURNO se había ido a las 22 hrs y llegaba a las 00 hrs...en resumen los pacientes quedan sin atención de urgencia...preocuparse 😮'
    Fue predecido como: 'Negativo' con: '99.95% de confianza'.
    La etiqueta correcta es: 'Negativo'.
    Info: ✅
    
    'Como ciudadana les indico que este último tiempo, he vivido una serie de negligencias por parte del servicio de salud de Valparaíso.'
    Fue predecido como: 'Negativo' con: '99.95% de confianza'.
    La etiqueta correcta es: 'Negativo'.
    Info: ✅
    
    'Placeres Salud? Porfavor fui una vez que cosa más insaluble ni papel para el baño ,una suciedad'
    Fue predecido como: 'Negativo' con: '99.93% de confianza'.
    La etiqueta correcta es: 'Negativo'.
    Info: ✅
    
    'Hacen el ridiculo al no conocer para lo que fueron designado Rina Isabel'
    Fue predecido como: 'Irrelevante' con: '99.98% de confianza'.
    La etiqueta correcta es: 'Irrelevante'.
    Info: ✅
    
    'Chile cada día peor. De las estupideces que andan preocupados'
    Fue predecido como: 'Irrelevante' con: '99.93% de confianza'.
    La etiqueta correcta es: 'Irrelevante'.
    Info: ✅
    
    'Que le vaya excelente y que logre solucionar la problemática de farmacia.. la lentitud es increíble... abuelos esperando 2 horas para recibir sus medicamentos es realmente injusto'
    Fue predecido como: 'Negativo' con: '99.94% de confianza'.
    La etiqueta correcta es: 'Negativo'.
    Info: ✅
    
    'Ojalá que con esto mejore la gestión de las horas médicas y la disponibilidad de los profesionales.'
    Fue predecido como: 'Irrelevante' con: '99.97% de confianza'.
    La etiqueta correcta es: 'Irrelevante'.
    Info: ✅
    
    'Me hubiese gustado participar, y porder haber compartido todos mis conocimientos, para la proxima sera ❤️'
    Fue predecido como: 'Irrelevante' con: '99.97% de confianza'.
    La etiqueta correcta es: 'Irrelevante'.
    Info: ✅
    
    'Increíble labor la que se hace en APS allá, a seguir fortaleciendo esas estrategias y crear otras!!🙌❤️'
    Fue predecido como: 'Positivo' con: '99.91% de confianza'.
    La etiqueta correcta es: 'Positivo'.
    Info: ✅
    
    'Felicidades, los recuerdo con mucho cariños. Aprendí demasiado como interno y despues como profesional, gracias por abrirme.las puertas en mi primera experiencia de trabajo en la salud❤️❤️❤️🔥🔥🔥'
    Fue predecido como: 'Positivo' con: '99.93% de confianza'.
    La etiqueta correcta es: 'Positivo'.
    Info: ✅
    
    'Deberían ver el tema de anticonceptivos 😴'
    Fue predecido como: 'Irrelevante' con: '99.98% de confianza'.
    La etiqueta correcta es: 'Irrelevante'.
    Info: ✅
    
    'Buenas me mandan el link de inscripción porfi?'
    Fue predecido como: 'Irrelevante' con: '99.90% de confianza'.
    La etiqueta correcta es: 'Irrelevante'.
    Info: ✅
    
    'Hola, yo quiero saber para cuándo va a estar disponible la vacuna contra la influenza para todo público. Creo que hasta el momento es para personas de riesgo de la tercera edad y embarazadas. Saludos. 💉'
    Fue predecido como: 'Irrelevante' con: '98.87% de confianza'.
    La etiqueta correcta es: 'Irrelevante'.
    Info: ✅
    
    'El personal del centro de salud siempre ha sido muy atento y profesional. Estoy agradecido por la atención recibida.'
    Fue predecido como: 'Positivo' con: '99.93% de confianza'.
    La etiqueta correcta es: 'Positivo'.
    Info: ✅
    
    'Es una lástima que los equipos médicos estén tan desactualizados, afecta la calidad del diagnóstico y tratamiento.'
    Fue predecido como: 'Negativo' con: '99.91% de confianza'.
    La etiqueta correcta es: 'Negativo'.
    Info: ✅
    
    'No hay suficiente personal para atender a todos los pacientes, las esperas son eternas.'
    Fue predecido como: 'Negativo' con: '99.64% de confianza'.
    La etiqueta correcta es: 'Negativo'.
    Info: ✅
    
    'Increíble el compromiso de los trabajadores de este centro, a pesar de las adversidades siempre dan lo mejor.'
    Fue predecido como: 'Positivo' con: '99.79% de confianza'.
    La etiqueta correcta es: 'Positivo'.
    Info: ✅
    
    'Deberían mejorar la comunicación entre los departamentos para evitar errores y malentendidos.'
    Fue predecido como: 'Negativo' con: '99.90% de confianza'.
    La etiqueta correcta es: 'Negativo'.
    Info: ✅
    
    'Me cancelaron la cita el mismo día, no pueden ser tan desorganizados.'
    Fue predecido como: 'Negativo' con: '99.95% de confianza'.
    La etiqueta correcta es: 'Negativo'.
    Info: ✅
    
    'Los baños están sucios y no hay suficientes medidas de higiene en las instalaciones.'
    Fue predecido como: 'Negativo' con: '99.97% de confianza'.
    La etiqueta correcta es: 'Negativo'.
    Info: ✅
    
    'Excelente la atención del pediatra, muy profesional y cariñoso con los niños.'
    Fue predecido como: 'Positivo' con: '99.90% de confianza'.
    La etiqueta correcta es: 'Positivo'.
    Info: ✅
    
    'Las instalaciones son antiguas y pequeñas, no están a la altura de las necesidades actuales.'
    Fue predecido como: 'Negativo' con: '99.36% de confianza'.
    La etiqueta correcta es: 'Negativo'.
    Info: ✅
    
    'A veces siento que los médicos no tienen suficiente tiempo para atender y escuchar a los pacientes adecuadamente.'
    Fue predecido como: 'Negativo' con: '99.90% de confianza'.
    La etiqueta correcta es: 'Negativo'.
    Info: ✅
    
    'Sería bueno tener más especialistas, para no tener que esperar meses por una cita.'
    Fue predecido como: 'Negativo' con: '99.98% de confianza'.
    La etiqueta correcta es: 'Negativo'.
    Info: ✅
    
    'Los administrativos son muy groseros y hacen que la experiencia en el centro sea desagradable.'
    Fue predecido como: 'Negativo' con: '99.96% de confianza'.
    La etiqueta correcta es: 'Negativo'.
    Info: ✅
    
    'Las enfermeras hacen un gran trabajo, siempre están dispuestas a ayudar.'
    Fue predecido como: 'Positivo' con: '99.66% de confianza'.
    La etiqueta correcta es: 'Positivo'.
    Info: ✅
    
    '¿Podrían mejorar la señalización dentro del centro? Es fácil perderse.'
    Fue predecido como: 'Negativo' con: '99.97% de confianza'.
    La etiqueta correcta es: 'Negativo'.
    Info: ✅
    
    'Hay mucha demora para la entrega de medicamentos, deberían ser más eficientes.'
    Fue predecido como: 'Negativo' con: '99.94% de confianza'.
    La etiqueta correcta es: 'Negativo'.
    Info: ✅
    
    'Gracias por ofrecer charlas y talleres para la comunidad, son de mucha ayuda.'
    Fue predecido como: 'Positivo' con: '99.88% de confianza'.
    La etiqueta correcta es: 'Positivo'.
    Info: ✅
    
    'La página web siempre está desactualizada, no se puede confiar en la información que aparece.'
    Fue predecido como: 'Negativo' con: '99.78% de confianza'.
    La etiqueta correcta es: 'Negativo'.
    Info: ✅
    
    'La atención es buena, pero los costos de los tratamientos son muy altos.'
    Fue predecido como: 'Negativo' con: '99.94% de confianza'.
    La etiqueta correcta es: 'Negativo'.
    Info: ✅
    
    'Esperé más de una hora más allá de mi cita, deberían respetar más el tiempo de los pacientes.'
    Fue predecido como: 'Negativo' con: '99.96% de confianza'.
    La etiqueta correcta es: 'Negativo'.
    Info: ✅
    
    'Los servicios en línea son muy prácticos y facilitan muchos trámites. Buen trabajo en eso.'
    Fue predecido como: 'Positivo' con: '99.35% de confianza'.
    La etiqueta correcta es: 'Positivo'.
    Info: ✅
    
    'Aunque las instalaciones son algo antiguas, el personal hace un trabajo extraordinario manteniendo todo limpio y organizado.'
    Fue predecido como: 'Positivo' con: '99.72% de confianza'.
    La etiqueta correcta es: 'Positivo'.
    Info: ✅
    
    'Es complicado obtener una cita debido a la alta demanda, pero una vez que estás allí, la atención es de primera clase.'
    Fue predecido como: 'Negativo' con: '80.04% de confianza'.
    La etiqueta correcta es: 'Positivo'.
    Info: ❌ ⚠️
    
    'Los trámites administrativos son un dolor de cabeza; sin embargo, los médicos y enfermeras son realmente compasivos y profesionales.'
    Fue predecido como: 'Positivo' con: '94.78% de confianza'.
    La etiqueta correcta es: 'Negativo'.
    Info: ❌
    
    'Aunque hay algunos aspectos que podrían mejorar, como la rapidez en la entrega de resultados, la calidad humana del personal es invaluable.'
    Fue predecido como: 'Positivo' con: '99.40% de confianza'.
    La etiqueta correcta es: 'Positivo'.
    Info: ✅
    
    'La espera fue larga y el lugar estaba algo desordenado, pero el médico fue muy atento y resolvió todas mis dudas.'
    Fue predecido como: 'Positivo' con: '82.65% de confianza'.
    La etiqueta correcta es: 'Negativo'.
    Info: ❌ ⚠️
    
    'Los procesos online podrían ser más intuitivos. Me costó mucho realizar mi agendamiento, pero el seguimiento fue constante y preciso.'
    Fue predecido como: 'Negativo' con: '64.38% de confianza'.
    La etiqueta correcta es: 'Negativo'.
    Info: ✅ ⚠️
    
    'Aprecio la variedad de especialistas disponibles, pero mejorar la coordinación entre los diferentes departamentos haría la experiencia mucho mejor.'
    Fue predecido como: 'Positivo' con: '74.53% de confianza'.
    La etiqueta correcta es: 'Negativo'.
    Info: ❌ ⚠️
    
    'Increíble la dedicación del personal, pero las instalaciones necesitan una renovación urgente para estar a la altura.'
    Fue predecido como: 'Negativo' con: '99.93% de confianza'.
    La etiqueta correcta es: 'Negativo'.
    Info: ✅
    
    'Los servicios de urgencias son eficientes, pero las áreas de espera podrían ser más cómodas y acogedoras.'
    Fue predecido como: 'Negativo' con: '92.59% de confianza'.
    La etiqueta correcta es: 'Negativo'.
    Info: ✅
    
    'La atención telefónica es impecable, pero la página web necesita mejoras para facilitar la navegación.'
    Fue predecido como: 'Negativo' con: '98.75% de confianza'.
    La etiqueta correcta es: 'Negativo'.
    Info: ✅
    
    'El centro ofrece una amplia gama de servicios, pero encontrar estacionamiento es una verdadera odisea.'
    Fue predecido como: 'Negativo' con: '99.07% de confianza'.
    La etiqueta correcta es: 'Negativo'.
    Info: ✅
    
    'El equipo médico es altamente calificado, pero a veces la comunicación con el paciente podría ser más clara y empática.'
    Fue predecido como: 'Negativo' con: '97.52% de confianza'.
    La etiqueta correcta es: 'Negativo'.
    Info: ✅
    
    'La rapidez en la atención de emergencias es destacable, pero el seguimiento post-visita podría ser más consistente.'
    Fue predecido como: 'Negativo' con: '88.35% de confianza'.
    La etiqueta correcta es: 'Negativo'.
    Info: ✅ ⚠️
    
    'Los servicios preventivos y los programas de bienestar son excelentes, pero la atención en casos de emergencia podría ser más rápida.'
    Fue predecido como: 'Negativo' con: '74.87% de confianza'.
    La etiqueta correcta es: 'Negativo'.
    Info: ✅ ⚠️
    
    'La calidad de la atención es muy buena, pero sería ideal que implementaran un sistema de recordatorios para las citas y medicamentos.'
    Fue predecido como: 'Negativo' con: '99.05% de confianza'.
    La etiqueta correcta es: 'Positivo'.
    Info: ❌
    > 

### Conclusión y Pasos Siguientes

- Se analizan los resultados para comprender las fortalezas y debilidades del modelo.
- Se considera realizar más entrenamientos con datos adicionales o ajustar los parámetros del modelo para mejorar la precisión.
- Al finalizar el entrenamiento, este modelo y su tokenizer se guardan para su uso posterior en la aplicación SoftwareBERTSaludCMV:
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/05f7aede-7011-479c-99d5-395a9bffa1f7/f4a8771a-fab7-436f-acf4-23e29d15d9fb/Untitled.png)
