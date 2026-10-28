# Interfaz de segmentación celular con Streamlit

La aplicación segmenta células de **WSSV** con los checkpoints finos de CEDIA v6. Foto y video cargado usan Mask2Former; la cámara en vivo usa RF-DETR-Seg. Ambos modelos producen máscaras de células sanas y enfermas para las métricas y el historial.

## Análisis de fotos

El modo **Foto** es el predeterminado y hace una sola inferencia, por lo que es mucho más rápido que el video. Acepta PNG, JPG, JPEG y TIFF, muestra las máscaras sobre la fotografía, cuenta las células detectadas, calcula el porcentaje de enfermas y permite descargar la imagen en PNG y un reporte CSV.

Mask2Former procesa las fotos a **1280 × 960** para preservar el detalle de las
células pequeñas. RF-DETR usa la resolución guardada en su checkpoint. En la
pestaña de cámara, la resolución elegida controla la captura del navegador y la
frecuencia de inferencia; no cambia la resolución interna entrenada del modelo.

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

## Modelo activo

Las rutas activas de **Mancha blanca (WSSV)** usan estos checkpoints exactos:

- **Foto y video cargado:** Mask2Former Swin-S, checkpoint de mejor época de
  validación (`mask2former-mejor.pth`).
- **Cámara en vivo:** RF-DETR-Seg 2XL, checkpoint de mejor mAP total de
  validación (`rfdetr-checkpoint-best-total.pth`).

Los pesos se descargan desde la release `wssv-v6-models`, se verifican contra su
tamaño y SHA-256 publicados y se almacenan en caché en el servidor. La descarga
se instala de forma atómica para evitar que una sesión lea un archivo parcial.
Mask2Former selecciona como máximo una clase por consulta; una misma máscara no
puede sumarse a la vez como sana y enferma. En cámara, el tracker asocia cajas
entre fotogramas aunque la clase fluctúe; el conteo final usa el voto
mayoritario de cada seguimiento, con un mínimo de dos observaciones. La cámara
usa ahora el BoT-SORT configurado en `trackers/live_botsort.yaml`, incluida la
compensación de movimiento de cámara; el video cargado usa ByteTrack de
Ultralytics. Ambos devuelven IDs que el adaptador vuelve a asociar con las
máscaras originales. Las detecciones de baja confianza solo se conservan si el
tracker las vincula a una identidad existente; las detecciones de alta
confianza siguen visibles aunque el tracker aún no les asigne ID.

La expansión **Verificación del modelo** muestra la huella del código, las
versiones instaladas, la ruta activa para cada modo y el archivo y SHA-256
esperados. Al cargar un modelo también muestra las clases, el dispositivo, la
resolución y el número de consultas de esa sesión. `CELL_MODEL_URL`,
`CELL_MODEL_BACKEND` y las variables OpenVINO pertenecen al cargador YOLO
heredado y **no cambian** las rutas actuales de WSSV.

### Compatibilidad heredada con YOLO/OpenVINO

El repositorio conserva el cargador anterior de YOLO11m/OpenVINO y las
variables `CELL_MODEL_BACKEND`, `CELL_OPENVINO_MODEL_DIR`,
`CELL_OPENVINO_MODEL_URL` y `CELL_MODEL_URL` para compatibilidad con artefactos
antiguos. La interfaz actual de WSSV v6 no llama ese cargador. Esos Secrets no
seleccionan el modelo activo de foto, video ni cámara.

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
