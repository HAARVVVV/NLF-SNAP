
# NLF-SNAP

**Noise-driven Latent Feature Synthesis Network for Audio Percussion**

Una red neuronal generativa que sintetiza sonidos de percusión controlables, en lugar de generar audio como una caja negra sin posibilidad de control parámetrico explícito, funciona como un instrumento paramétrico que deja el control creativo en manos de quien produce. Combina un autoencoder variacional condicional (**C-VAE**) con procesamiento diferenciable de señales (**DDSP**), esculpiendo cada percusión a partir de ruido mediante filtros FIR aprendidos.

Proyecto propio, ideado y desarrollado por mí. Nació como mi Trabajo de Fin de Grado en Matemáticas (Universidad Complutense de Madrid), calificado con **10**. El proyecto sigue en desarrollo.

**Autor:** Andrés Sánchez Ortiz · [LinkedIn](https://www.linkedin.com/in/andres-so/) · [GitHub](https://github.com/HAARVVVV)

---

## 🎧 Escúchalo

Sonidos generados por NLF-SNAP (uno de cada clase principal):

- **Kick:** [`▶ Escuchar Kick generado (descarga)`](https://github.com/user-attachments/files/30130719/003_KICK_sampled.wav)
- **Snare:** [`▶ Escuchar Snare generado (descarga)`](https://github.com/user-attachments/files/30130737/005_SNARE_sampled.wav)
- **Hi-Hat:** [`▶ Escuchar Hi-Hat generado (descarga)`](https://github.com/user-attachments/files/30130757/006_HIHAT_sampled.wav)


### Demo musical

Un pequeño tema construido usando **únicamente** percusión generada por NLF-SNAP, secuenciada en Strudel:

🎵 **Audio:** [`▶ Escuchar Demo (descarga)`](https://github.com/user-attachments/files/30130786/NLF-SNAP.DEMO01.wav)

🎥 **Video:** [`▶ Ver Demo (Online, Recomendado)`](https://github.com/user-attachments/assets/920572b0-9415-43c8-9949-1390591cd4fe)

---

## El problema

En la producción musical, diseñar sonidos de percusión suele obligar a elegir entre dos malas opciones: las librerías de *samples*, que suenan bien pero son estáticas y poco maleables, y los sintetizadores clásicos, potentes pero que exigen un conocimiento técnico profundo de síntesis y procesamiento de señales.

**NLF-SNAP** explora una tercera vía. En lugar de un modelo generativo que entrega audio cerrado e inmodificable (como las arquitecturas de difusión o los GANs), plantea una red que actúa como un *instrumento paramétrico controlable*: comprime la complejidad espectral de la percusión en un espacio latente de baja dimensión, y desde ahí permite reconstruir, interpolar y generar timbres nuevos, preservando la agencia creativa de quien produce.

La arquitectura es un **C-VAE** acoplado a un sintetizador sustractivo diferenciable (**DDSP**, inspirado en el [trabajo de Magenta](https://github.com/magenta/ddsp)): la red no genera la onda muestra a muestra, sino que infiere los coeficientes de un filtro FIR que esculpe el sonido a partir de ruido. Se especializa en cinco clases de percusión: Hi-Hats, Kicks, Snares, Toms y percusiones auxiliares. Sirve además como banco de pruebas para estudiar cómo distintas estrategias de regularización del espacio latente (KLD annealing, free-bits) afectan a la fidelidad de los sonidos generados.

### Diagramas de la Arquitectura 

He hecho unos diagrámas para representar el pipeline de datos: 

1. Arquitectura General:
   ![Diagrama general NLF-SNAP](./TFG_DIAGRAMA_GENERAL.png)
  
2. Diagrama del Encoder:
   ![Diagrama del Encoder](./TFG_DIAGRAMA_ENCODER.png)

3. Diagrama del Decoder:
   ![Diagrama general NLF-SNAP](./TFG_DIAGRAMA_DECODER.png)

---

## Estructura del repositorio

El repositorio contiene el código, un checkpoint del modelo entrenado y los resultados (audios y gráficas). Los datasets de audio y los pesos completos de todos los experimentos se han dejado fuera por tamaño.

* **`/CVAE_outputs`** — audios `.wav` generados por NLF-SNAP.
  * `/sample` — sonidos generados de forma independiente por la red.
  * `/reconstruct` — reconstrucciones a partir de un sonido de entrada.
* **`/DEMO`** — demo musical: un tema hecho solo con percusión generada por el modelo (audio y vídeo).
* **`/CVAE_CSV`** — historiales de entrenamiento por época de cada experimento (Train KLD, Val MSS, etc.).
* **`/plots_and_graphs`** — gráficas generadas: evolución de pérdidas, distancias acústicas y PCA del espacio latente, por experimento.
* **`best_nddsp.pt`** — pesos del modelo ganador (Linear Annealing 150, guardado por mejor `Val MSS`).
* **`PERCUSSION CLASSES.txt`** — división exacta del dataset en las cinco clases.

<details>
<summary><b>Scripts principales</b> (desplegar)</summary>

### Arquitectura

* `tfg_encoder.py` — red convolucional que comprime el espectrograma Mel en el espacio latente.
* `tfg_decoder_NDDSP.py` — genera los coeficientes de un filtro FIR variable en el tiempo a partir del vector latente condicionado.
* `tfg_model.py` — integra encoder y decoder, y aplica el truco de reparametrización $\mathcal{N}(0, I)$ del VAE.
* `tfg_NoiseSynth.py` — aplica el filtro sobre ruido para construir el audio final.
* `tfg_loss_NDDSP_mss.py` — pérdida espectral multiescala (MSS) que guía la optimización.

### Pipeline

* `tfg_dataset.py` y `tfg_dataloader.py` — preparación y carga de los espectrogramas Mel a partir del audio.
* `tfg_train_mss.py` — script central de entrenamiento. Guarda checkpoints por época; el modelo definitivo se selecciona por el mínimo de `Val MSS` (ver nota más abajo).
* `tfg_pca.py` — reducción de dimensionalidad (PCA) para analizar la estructura del espacio latente tras el entrenamiento.

### Nota: 

Dentro de los propios archivos se encuentra toda la información detallada de los procesos informaticos y matemáticos utilizados pertinente para la evaluación correcta del modelo.  El orden recomendado de lectura es el siguiente: 

- PERCUSSION CLASSES.txt
- audio_processor.py
- preprocess_script.py
- tfg_datatester.py
- tfg_encoder.py
- tfg_model.py
- tfg_decoder_NDDSP.py
- tfg_loss_NDDSP_mss.py
- tfg_dataset.py
- tfg_dataloader.py
- tfg_train_mss.py
- tfg_pca.py
- tfg_producer.py
- ./plots_and_graphs
- ./CVAE_outputs


</details>

---

## Experimentos y resultados

NLF-SNAP se usó como banco de pruebas para estudiar el **colapso posterior** del espacio latente, uno de los problemas clásicos de los VAE. Se entrenaron 11 modelos variando la estrategia de regularización sobre el peso ($\beta$) de la divergencia de Kullback-Leibler:

- **Baseline constante:** $\beta = 0.5$ y $\beta = 1$
- **Free-Bits:** $\lambda = 0.5$ y $\lambda = 1$
- **Linear Annealing:** rampas de 40, 60, 100 y 150 épocas
- **Cyclical Annealing:** ciclos de 20, 40 y 100 épocas

El modelo con mejor fidelidad acústica fue el de **Linear Annealing a 150 épocas**:

| Métrica | Valor |
|---|---|
| LSD media (distancia espectral lineal) | **11.27 dB** |
| MSD media (distancia sobre espectrograma Mel) | **12.35 dB** |

> **Hallazgo principal.** Al usar la pérdida total (`Val Total`) como criterio para guardar el mejor checkpoint, las estrategias de *annealing* generan falsos positivos: como $\beta$ crece con las épocas, la pérdida total sube artificialmente y se acaba guardando un modelo temprano y peor. Seleccionar el checkpoint por la **pérdida de reconstrucción** (`Val MSS`) en lugar de la total mejora de forma notable la calidad acústica real del audio generado.

---

## Limitaciones y trabajo futuro

El proyecto está vivo y hay frentes claros de mejora, algunos identificados durante la propia evaluación:

- **Síntesis solo por ruido.** El sintetizador esculpe todo a partir de ruido, sin osciladores armónicos. Funciona sorprendentemente bien —la red aprende a fabricar tonos graves con filtros paso-banda estrechos— pero limita la fidelidad de sonidos con fuerte componente tonal (kicks, toms). El siguiente paso natural es añadir una fuente armónica (DDSP completo).
- **Métricas perceptuales.** LSD y MSD penalizan injustamente los sonidos muy estocásticos (hi-hats). Se plantea evaluar con Fréchet Audio Distance (FAD), más robusta a la aleatoriedad de fase.
- **Etiquetas del dataset.** Las clases (snare/tom/perc) tienen solapamiento acústico; un condicionamiento por características continuas en vez de etiquetas discretas podría organizar mejor el espacio latente.
- **Tiempo real.** El objetivo a largo plazo es exportar el modelo a un plugin (VST/AU) para usarlo dentro de un DAW.

---

## Uso

### Requisitos

El entorno completo (Python, PyTorch, torchaudio, librosa, etc.) está en `environment.yml`. Con Conda:

```bash
conda env create -f environment.yml
conda activate TFG_CVAE_clean
```

### Generar sonidos nuevos

Es el camino principal y **solo necesita el modelo entrenado**, no el dataset:

1. El modelo `best_nddsp.pt` ya viene incluido en `checkpoints/`, no requiere descarga aparte`.
2. Ejecuta:

```bash
python tfg_producer.py
```
Los sonidos generados aparecerán en `CVAE_outputs/sample/`, organizados por clase de percusión. La red los crea muestreando el espacio latente desde cero, sin partir de ningún audio de referencia.

### Reconstrucción (opcional)

`tfg_producer.py` también puede reconstruir sonidos a partir de audios reales y generar los espectrogramas comparativos, pero esto **requiere el dataset preprocesado** (carpeta `processed_dataset/`), que no se incluye en el repositorio por tamaño. Si el dataset no está disponible, el script omite esta parte automáticamente y genera solo los sonidos nuevos.

---

### Ejecutar con Docker

Alternativa que no requiere instalar Python ni las dependencias: solo Docker.

```bash
docker build -t nlf-snap .
```

Ejecutar (Windows):

```bash
docker run -v "%cd%\salida:/app/CVAE_outputs" nlf-snap
```

Ejecutar (Linux / macOS):

```bash
docker run -v "$(pwd)/salida:/app/CVAE_outputs" nlf-snap
```

El contenedor genera los sonidos y los deja en la carpeta `salida/` de tu máquina. La imagen usa PyTorch en versión CPU, suficiente para inferencia.

Se puede ejecutar cualquiera de los archivos contenidos en la imagen de Docker, con `docker run -it nlf-snap bash` + `ls` se pueden ver todos los archivos. Por ejemplo: docker run nlf-snap python tfg_pca.py
