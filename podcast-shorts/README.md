# CorteStudio · Podcast a Shorts

MVP para convertir un episodio propio en clips verticales. Subes un video, el servidor lo transcribe con faster-whisper, propone hasta cinco fragmentos según señales del diálogo y exporta el recorte que elijas en MP4 vertical con subtítulos.

## Ejecutar en tu computadora

1. Instala Docker Desktop.
2. Abre una terminal dentro de esta carpeta.
3. Ejecuta `docker compose up --build`.
4. Abre `http://localhost:8000`.
5. Sube un MP4, MOV, MKV, WEBM, M4V o AVI y pulsa **Encontrar momentos**.

La primera ejecución descarga el modelo `small`; después queda guardado en el volumen `whisper-models`. El procesamiento usa CPU e INT8 por defecto, así que el tiempo depende de la duración del episodio y del equipo.

## Qué hace el MVP

- Transcribe audio en español y otros idiomas detectados automáticamente.
- Revisa fragmentos de 18 a 58 segundos y puntúa señales editoriales como pregunta inicial, contraste, frase completa, densidad de diálogo y ritmo.
- Sugiere un título breve y términos/hashtags extraídos de la transcripción.
- Permite elegir el recorte y la posición horizontal del cuadro (centro, izquierda o derecha).
- Exporta un MP4 vertical 9:16 con subtítulos incrustados.

La puntuación es una heurística editorial; no predice viralidad ni vistas. Los hashtags son relevantes al tema transcrito, pero no son una lista de tendencias en vivo. El enlace de Google Trends abre la búsqueda con filtro de YouTube para comprobar interés reciente. En YouTube, título, miniatura y descripción pesan más para descubrimiento que las etiquetas de metadatos; las etiquetas tienen un papel limitado salvo errores ortográficos frecuentes ([ayuda oficial](https://support.google.com/youtube/answer/146402)).

## Límites actuales

- Para encontrar buenos cortes, el audio debe ser claro. Revisa la transcripción y los tiempos antes de publicar.
- El recorte vertical usa un encuadre fijo centro/izquierda/derecha; no hace seguimiento automático de caras.
- Este MVP no consulta tendencias de YouTube automáticamente ni publica videos en tu canal.
- El archivo se procesa en el servidor local de Docker y se borra al terminar el análisis/exportación. El modelo se conserva para evitar descargarlo cada vez.

Para publicarlo en internet habría que desplegar el servidor en una máquina con FFmpeg y suficiente memoria, configurar almacenamiento temporal y proteger la subida con límites, cuotas, autenticación, HTTPS y borrado de archivos. No publiques el servidor local tal cual abierto a internet.

## Variables opcionales

- `WHISPER_MODEL`: `base`, `small`, `medium`, `large-v3` (más calidad consume más tiempo/recursos).
- `WHISPER_DEVICE`: `cpu` o `cuda` si el contenedor dispone de CUDA compatible.
- `WHISPER_COMPUTE_TYPE`: por defecto `int8` para CPU.
- `WHISPER_CPU_THREADS`: hilos que usa el modelo.
- `MAX_UPLOAD_MB`: tamaño máximo de archivo, por defecto 1800.
