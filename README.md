# Interfaz de segmentación celular con Streamlit

La aplicación analiza **fotos y videos** con el modelo YOLO de segmentación. En las fotos conserva la imagen original y permite mostrar u ocultar las máscaras superpuestas; las células sanas se muestran en verde y las enfermas en rojo. El video exportado conserva el video original y superpone las máscaras, sin cajas ni etiquetas.

## Análisis de fotos

El modo **Foto** es el predeterminado y hace una sola inferencia, por lo que es mucho más rápido que el video. Acepta PNG, JPG, JPEG y TIFF, muestra las máscaras sobre la fotografía, cuenta las células detectadas, calcula el porcentaje de enfermas y permite descargar la imagen en PNG y un reporte CSV.

El tamaño de inferencia inicial es 1280 px, igual que el usado al entrenar el modelo, para preservar el detalle de células pequeñas.

## Análisis de videos

También hace seguimiento de objetos para que el número final sea de células únicas, en vez de sumar la misma célula en cada fotograma. La clase final de una célula es la más frecuente durante su seguimiento.

### Cámara en Streamlit Cloud

La cámara en vivo usa WebRTC. La configuración predeterminada intenta primero
una conexión directa y usa TURN únicamente como respaldo cuando la red bloquea
la ruta directa. Esto evita consumir cuota TURN durante una conexión normal.

La aplicación usa Cloudflare TURN como proveedor predeterminado. Crea una TURN
key en Cloudflare Calls y configura estos Secrets en **Manage app → Settings →
Secrets**:

```toml
CLOUDFLARE_TURN_KEY_ID = "uid_de_la_turn_key"
CLOUDFLARE_TURN_KEY = "clave_larga_de_la_turn_key"
CLOUDFLARE_TURN_TTL_SECONDS = "86400"
RTC_TURN_PROVIDER = "cloudflare"
RTC_ENABLE_TURN = "true"
```

`CLOUDFLARE_TURN_KEY` es la clave larga devuelta al crear la TURN key; no es
una clave pública ni debe enviarse al navegador. El servidor solicita
credenciales efímeras y solo esas credenciales llegan al componente WebRTC.

Si el navegador muestra **“Connection is taking longer than expected”** y el
video queda en blanco, comprueba primero que los Secrets anteriores estén
completos. Para probar sin ningún retransmisor, configura:

```toml
RTC_ENABLE_TURN = "false"
```

Metered ya no se usa automáticamente. Solo se puede reactivar como alternativa
de emergencia con `RTC_TURN_PROVIDER = "metered"` y sus Secrets. También se
aceptan credenciales TURN estáticas con:

```toml
RTC_TURN_PROVIDER = "custom"
RTC_TURN_URLS = "turn:servidor:80,turn:servidor:443,turns:servidor:443?transport=tcp"
RTC_TURN_USERNAME = "tu_usuario_turn"
RTC_TURN_CREDENTIAL = "tu_credencial_turn"
```

No publiques esas credenciales en GitHub. La cámara se inicia con **INICIAR
CÁMARA** dentro del recuadro; después se habilita **Iniciar detección**.

### Enviar las cuatro muestras al Excel de OneDrive

Cuando se completan las cuatro muestras del lote, aparece **Enviar al Excel** y
una vista previa con las columnas `Código`, `Vacuolización` y `White Spot
(WSSV)`. El botón agrega las filas al libro en línea; no reemplaza las filas
anteriores. Por ahora `Vacuolización` se envía en blanco y `White Spot (WSSV)`
recibe el grado calculado por la detección actual.

La conexión usa Microsoft Graph desde el servidor de Streamlit. Para el piloto,
registra una aplicación en Microsoft Entra ID, concede el permiso de aplicación
`Files.ReadWrite.All` con consentimiento administrativo y agrega estos Secrets
en **Manage app → Settings → Secrets**:

```toml
MS_TENANT_ID = "tu-tenant-id"
MS_CLIENT_ID = "tu-client-id"
MS_CLIENT_SECRET = "tu-client-secret"
MS_ONEDRIVE_USER = "usuario@empresa.com"
MS_ONEDRIVE_FILE_PATH = "WSSV_Plantilla_Piloto.xlsx"
```

`MS_ONEDRIVE_FILE_PATH` también puede incluir carpetas dentro del OneDrive del
usuario. El secreto nunca debe escribirse en `app.py` ni confirmarse en GitHub.
Si falta alguna configuración, el botón seguirá visible y mostrará el dato que
falta sin indicar que el lote fue enviado.

## Modelo

El modelo revisado es de segmentación y declara estas clases:

- `0`: `celula_enferma`
- `1`: `celula_sana`

En ejecución local, la app usa `models/best.pt` si está disponible. En la versión publicada, el modelo se descarga automáticamente desde la release pública `v1.0.0`, por lo que no depende de rutas específicas de un equipo.

### Backend OpenVINO para CPU

La app intenta usar OpenVINO como backend para CPU y conserva PyTorch como respaldo automático. Para forzar una prueba local con un modelo ya convertido:

```bash
export CELL_MODEL_BACKEND=openvino
export CELL_OPENVINO_MODEL_DIR="/ruta/a/best_openvino_model"
streamlit run app.py
```

La carpeta OpenVINO debe contener los archivos `.xml` y `.bin` del modelo de segmentación. Si no se configura una carpeta, la app intenta convertir `best.pt` automáticamente y conserva la conversión en la caché temporal. También se puede configurar `CELL_OPENVINO_MODEL_URL` con un ZIP público de esa carpeta. Para forzar el backend anterior, usa `CELL_MODEL_BACKEND=pt`.

Si se reemplaza el modelo local, conserva este nombre y ubicación:

```bash
mkdir -p models
cp "/ruta/al/nuevo/modelo.pt" models/best.pt
```

## Instalación y ejecución

La versión publicada usa Python 3.14. `requirements.txt` fija las versiones
del componente WebRTC, del modelo y de OpenVINO para evitar que una nueva
instalación cambie silenciosamente el comportamiento de la cámara o del
conteo. Si se cambia alguna de esas versiones, valida primero cámara, inicio y
detención de detección, métricas y guardado de muestras.

Desde esta carpeta:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Calificación

La calificación se calcula usando:

```text
porcentaje enfermo = células enfermas / (células sanas + células enfermas) × 100
```

Una foto con células detectadas y **0% de células enfermas** recibe **Grado 0**. Como aún no se proporcionó una rúbrica, los cortes iniciales para los grados 1 a 3 son 10%, 30% y 60%; un valor superior queda en grado 4. Si no se detectan células, la app muestra “Sin datos” en lugar de asignar una calificación. Reemplaza estos rangos en el código por los criterios científicos o clínicos de tu estudio antes de interpretar el resultado.

## Medición de tamaño celular en fotos

Para las fotografías tomadas a **40×**, la app calcula por separado una estimación del diámetro promedio equivalente (µm), el perímetro promedio (µm) y el área promedio (µm²) de las células **sanas** y de las **enfermas**. La calibración usa la referencia de **50 µm** proporcionada para este conjunto de imágenes:

```text
escala estimada = 50 µm / longitud de referencia equivalente en píxeles
```

Si una foto conserva una barra de escala visible, la app la usa directamente. Si no, aplica la calibración estimada para 40×. Esta estimación es válida solamente si las fotos conservan la misma magnificación, cámara, campo de visión y proporción de la imagen de referencia; no debe usarse tras recortar la foto. Las mediciones se incluyen también en el CSV.

## Al agregar un video

La app acepta MP4, AVI, MOV y MKV. Tras pulsar **Analizar y reproducir detección**, muestra una previsualización mientras procesa, permite descargar el video original con máscaras superpuestas en MP4 y un reporte CSV con los conteos, porcentaje y calificación. Para acelerar el análisis en equipos sin GPU, se analiza uno de cada dos fotogramas y el resultado conserva aproximadamente la duración del video.
