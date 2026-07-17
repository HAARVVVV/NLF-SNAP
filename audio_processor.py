"""
Created on Sat Dec 13 10:48:41 2025

@author: Andrés Sánchez Ortiz
"""

import librosa
import numpy as np
import torch


'''

La clase 'AudioProcessor' procesa los audios extraidos que utilizaremos para el 
entrenameinto. Nuestro objetivo es normalizar y homogeneizar duración y ganancias 
de los diversos audios para poder transformarlos en 'ESPECTROGRAMAS MEL'. 

Una señal de audio estándar discurre a lo largo del dominio del tiempo tomando 
como recorrido la amplitud, ahora, mediante la 'TRANSFORMADA DE FOURIER DE TIEMPO 
CORTO' (STFT) seremos capaces de reflejar la misma información sobre el 'DOMINIO
DE LA FRECUENCIA' tomando como recorrido unos frames temporales. 

Mediante la STFT estamos desglosando las diferentes frecuencias del sonido 
que 'actúan' en conjunto para construirlo, es de esta forma que seremos capaces 
de conocer que bandas frecuenciales existen dentro del sonido y con qué intesidad 
y extensión suenan. 

La STFT al ser una transformación lineal no nos proporciona todas las ventajas 
que nos gustaría para desarrollar el proyecto, encontramos dificultades, por 
ejemplo, a la hora de calcular la función de pérdida. El espectro de magnitud 
que nos proporciona la STFT es complejo, necesitaremos transformarlo al plano 
real para poder operar con él a lo largo de la red.

Aún así ya encontramos ventajas sustanciales con respecto al tratamiento con la 
onda sonora pura al trivializar con la STFT el problema del tratamiento de la fase,
pequeños desplazamientos a lo largo del tiempo en la señal cruda causaría una 
diferencia abismal dentro de la fase, durante el entrenamiento la función de 
pérdida (MSE) se dispararía, aumentando considerablemente los tiempos de 
entrenamiento, evaluación y test y, probablemente, proporcionando resultados pobres
al no captar realmente el concepto de la reconstrucción al intentar clavar la fase. 
    
Para pasar al plano real el espectrograma de la STFT aplicamos una nueva transformación
lineal de magnitud (^2). Por último en el pipeline de preprocesamiento de los 
sonidos, aplicaremos una transformación final, en este caso no lineal, para, 
traducir la onda sonora orginal bajo la llamada 'ESCALA MEL', que modela la percepción 
no lineal frecuencial del oído humano. 

Para ello multiplicamos un banco de 'FILTROS MEL' triangulares en forma de matriz 
sobre el espectro colapsado de la STFT, comprimiendo logaritmicamente las frecuencias 
altas y resolviendo mejor las bajas, cuyas variaciones somos capaces de captar 
mucho mejor, obteniendo así un 'ESPECTROGRAMA MEL'.

Además, esta propiedad de resolución de bajas frecuencias hace a un espectrograma 
Mel ideal para el tratamiento de sonidos con fuerte presencia de graves, como 
las percusiones que estaremos tratando a lo largo del trabajo. 

El tray de operaciones completo para pasar de la onda de audio cruda al espectrograma 
frecuencial lineal de la STFT y a continaución al Espectrograma Mel será: 
    

################################ OPERACIONES ##################################    

    1. STFT de la señal inicial:
        
        STFT{x[n]}[k,t] = sum_{-\infty}^{\infty} x[n] w[n-rH] e^{(-j2\pi kn)/N}
        
       1. x[n]: Señal de Audio.  
       
       2. w[n-tH]: La ventana de análisis desplazada al frame r-ésimo.
       
       3. H: Hop length, longitud en samples del salto entre ventanas de análisis.
       
       4. N: Es el n_fft, es decir, el número de samples por ventana de análisis.
    
       5. k: El número de bin de la FFT dentro de cada frame analizado. Representa 
             una pequeña banda de frecuencia del espectro lineal resultante de 
             la FFT aplicada a una ventana concreta de la señal.

    
    2. S[k,t] = |STFT{x[n]}[k,t]|^p (p=2 por defecto en librosa -> versión real 
                                     no negativa que es la que aplicaremos para
                                     obtener el espectrograma Mel)
    
    
    3. Y_{mel} [m,t] = \sum_{k=0}^{K-1} M[m,k] * S[k,t]
    
        1. m: Índice de banda Mel. Representa uno de los filtros triangulares 
              del banco Mel recogidos en la matriz M. 
    
        2. t: Frame temporal. 
        
        
    4. Podemos expresar la misma sucesión de operaciones con la siguiente expresión
       matricial: 
        
           Y_{mel} = M * X_{fft}
 
    
Agruparemos las frecuencias resultantes sobre las llamadas 'BANDAS MEL'. 
La STFT opera bajo la premisa de que la señal es estacionaria, invariante en el 
tiempo, sin embargo, ninguna onda real que trataremos tiene estas cualidades, es 
por ello que toma ventanas sobre las que si podremos asumir y anañizar la señal 
de esta forma, los parámetros relacionados con la STFT serán: 


################################# PARÁMETROS ##################################

1. sample_rate: (sr). Parámetro inicial de frecuencia de muestreo, analizamos la 
                amblitud de la onda en relación al sample_rate. Establecemos 
                un valor moderado para ayudar al entrenamiento. (16000)
             
                ----------------------------- NOTA ----------------------------
              
                Con 16000 Hz es quizas difícil capturar los agudos característicos
                de los 'HIHATS' puede ser útil hacer una pequeña disertación al 
                poseer agudos (> 8000 Hz).
                
                Una vez realizado el primer entrenamiento y comprobado los 
                resultados podemos intentar cambiar el sr. 
          
                ---------------------------------------------------------------
                
                
2. duration: Parámetro inicial de duración del audio original, no queremos que 
             ningún oneshot exceda un segundo de duración, tomaremos un valor 
             suficiente para homogeneizar todos los oneshots. (0.8)
          
            
3. n_samples: El número de samples total contenido en una señal. lo calculamos 
              como:
    
               n_samples = sample_rate * duration
                
              (12800)
           
           
4. n_mels: El número de bandas Mel del espectrograma Mel que obtenemos 
           como resultado. Estas bandas Mel agrupan conjuntos de frecuencias 
           proximas. Elegiremos una resolución lo suficientemente buena para
           obtener resultados audibles, pero tampoco excederemos el nº de 
           bandas para facilitar el procesamiento y entrenamiento.
        

5. n_fft: Es el tamaño de la ventana de Fourier, el nº de samples que incluimos  
          en cada frame sobre los que aplicaremos el análisis de la transformada
          FFT. (1024)
          
          Por defecto, la función STFT de Librosa asume 'center = True'. Esto 
          significa que añade "ceros" (padding) al principio y al final del audio 
          (media ventana, 512 muestras) para que el centro de la primera ventana 
          coincida exactamente con la muestra 0 del audio a analizar.
       
          Debido al teorema de Nyquist-Shannon, cómo el espectrograma lineal de 
          la STFT es complejo presenta información por duplicado, obtendremos 
          entonces que la información relevante (los bins frecuenciales obtenidos
          de la transformación) será de la mitadq:
              
              frec_div_STFT = (n_fft/2) + 1
              
              2048/2 +1 = 1025 divisiones frecuanciales en la representación STFT.
       

6. hop_length: Es el nº de samples que recorremos antes de inicializar la siguiente
               ventana. (256)
               
               Con esto ya podremos obtener el número de frames temporales en 
               los que se dividirán los espectrogramas Mel:
                   
                   
                   n_frames = (\floor(n_samples/hop_length) + 1)
              
                   floor(25600/512) + 1 = 51 frames temporales. 
                
                           
7. overlap: Indice de solapamiento entre ventanas consecutivas, viene determinado 
            por: 
                
                overlap = n_fft - hop_lentgh 
            
            Lo hemos escogido de manera premeditada sobre n_fft para obtener un 
            solapamietno determinado, lo suficientemente grande para capturar 
            las cualidades de los graves y los transientes pero no exceso para 
            no saturar el número de ventanas de Fourier.
            (768 - 75% (1024))
         
            Ahora, ¿por qué hacemos que las ventanas se solapen? 
            
            Para evitar la creación de artefactos auditivos a causa de cortes 
            abruptos y/o diferencias en la fase de las frecuencias extraidas del
            STFT. 


8. Hann window: Las ventanas de análisis particulares que nosotros 
                usaremos son 'Hann windows', funciones especiales que aplican 
                un suavizados sobre las zonas solapadas para minimizar los 
                efectos indeseados en la reconstrucción del audio. Una ventana 
                de este estilo tiene la fórmula: 
                    
                    w(n) = 0.54 - 0,46 cos(2\pi n)/(N-1)


Veamos en que consiste 'audio_processor.py' exactamente:
    

################################ OPERACIONES ##################################    

1. fix_length: Asegura que el array audio contenga exactamente n_samples para
               mantener constante la dimensionalidad en el procesamiento de datos
               y su posterior uso. Si es demasiado largo recortamos, por otro 
               lado si es demasiado corto, como no queremos añadir más info 
               relevante, para igualar las longitudes de los datos añadimos 
               tantos ceros como samples extra hagan falta. 


2. audio_to_mel: Calculamos el espectrograma Log-Mel, será la información que el 
                 encoder finalmente reciba. (np.ndarray[shape=(…, n_mels, t)]) 
                 ((64, 51))
                 
                 Además con este esprectrograma adaptamos también la amplitud 
                 del audio sobre la escala logarítmica de los decibelios.
                 
                 
                 ######################## PARÁMETROS ##########################
                 
                 1. fmin/fmax: Marcamos los límites de la frecuencia, tomará la 
                               frecuencia de Nyquist.  (sr/2 = 8000)
                 
                
                 ####################### OPERACIONES ##########################
                 
                 2. power_to_db: Cambiamos la escala de la energía (amplitud) a 
                                 log dB, de la misma forma que con los Espectrogramas 
                                 Mel, adaptamos la información sobre escalas estables
                                 significativas para la percepción humana, 
                                 asistiendo al entrenamiento a generar cambios 
                                 significativos. Con ref = np.max, el máximo 
                                 queda en 0 dB y el top_db por defecto es 80 dB.
                              
                                 Normalizaremos después los decibelios para que 
                                 exista coherencia entre todos los datos en el
                                 [0,1]. 
                 
                    
                 3. np.clip(): Aseguramos que el recorte de la normalización quede 
                               bien ajustado, en algunas muestras obteníamos pequeñas
                               extensiones no deseados en los extremos. 


3. preprocess_file: La unidad principal del preprocesamiento de audios: 
                    

                    ####################### PARÁMETROS ########################
             
                    1. file_path: Indicamos el directorio de los ficheros.

                    
                    ####################### OPERACIONES #######################
 
                    1. librosa.load(): Cargamos el audio en file_path en el sistema
                                       a modo de array. sr = self.sample_rate y
                                       remuestrear para homogeneizar los samples.
                                       Mono = True para no duplicar el tamaño del 
                                       preprocesado con información redundante. 
                                       En principio este proyecto se plantea 
                                       unicamente para sonidos Mono. 
                    
                    
                    2. librosa.effects.trim(): Podemos encontrar inconsistencias
                                               a lo largo de los audios cargados,
                                               algunos presentan silencios mínimos
                                               al inicio (y al final). Recortamos 
                                               estos 'silencios' con un 'threshold'
                                               top_db (RMS). Así conseguimos 
                                               alinear todos los inicions. Ahora,
                                               hacemos uso de fix_length() y 
                                               rellenamos de forma consecuente.
                
                
                    3. pico: Normalizamos la energía con respecto al pico de 
                             ganancia máximo. 
                       
                             
                    4. torch.from_numpy...: Cargamos el audio y el espectrograma 
                                            en forma de array de numpy como un 
                                            tensor. convertimos los valores a.float()
                                            para asegurarnos del formato correcto.
                                            unsqueeze(0) para añadir una dimensión
                                            extra fundamental para el correcto 
                                            funcionamiento del encoder. 


'''

class AudioProcessor:
    
    
    def __init__(self):
         
        
        self.sample_rate = 32000 
        self.duration = 0.8 
        self.n_samples = int(self.sample_rate * self.duration) # = 25600
        self.n_mels = 64    
        self.n_fft = 2048      
        self.hop_length = 512   
        

    def fix_length(self, audio):
        
        
        if len(audio) > self.n_samples:
            
            return audio[:self.n_samples] # Recortamos hasta el máximo 
        
        else:
            
           
            extra = self.n_samples - len(audio) # Extendemos la duración
            
            return np.pad(audio, (0, extra), mode = 'constant') 
        
    '''
    
    https://archive.org/details/NumPyBook
    
    '''
    
    def audio_to_mel(self, audio):
        
         
        Spectr = librosa.feature.melspectrogram(y = audio, sr = self.sample_rate, 
                                           n_fft = self.n_fft, hop_length = self.hop_length,
                                           n_mels = self.n_mels, fmin = 20, fmax = 16000)
        
        '''
        https://librosa.org/doc/latest/generated/librosa.feature.melspectrogram.html
        '''
         
        
        log_spec = librosa.power_to_db(Spectr, ref = np.max)
        log_spec = (log_spec + 80) / 80 # Normalización [0,1]
        
        return np.clip(log_spec, 0, 1) 
    

    def preprocess_file(self, file_path): 
        
        # Procesamiento y normalización: Carga -> Mono -> Fix Length -> Norm -> Mel
        
        audio_pre, _ = librosa.load(file_path, sr = self.sample_rate, mono = True)
        audio_pre, _ = librosa.effects.trim(audio_pre, top_db = 30) # Recortar silencios. 
        
        '''
        https://doi.org/10.5281/zenodo.15006942
        '''
        
        audio_pre = self.fix_length(audio_pre) # Ajustar longitud (Padding/Trimming)
        
        # pico = int(np.abs(audio_pre).max())
        
        pico = np.abs(audio_pre).max() 
         
        if pico > 0: # Hay alguna señal vacia asi que creé la condición por si acaso. 
            
            audio_pre = audio_pre/pico # Generar Espectrograma Mel (I PARA EL ENCODER)
        
        mel_spec = self.audio_to_mel(audio_pre)
        
        return {
            
            'audio': torch.from_numpy(audio_pre).float().unsqueeze(0), # pt[1, N_samples]
            'mel': torch.from_numpy(mel_spec).float().unsqueeze(0) # pt[1, n_mels, Time]
        
        }

# ------------------------------- PRUEBA PROCESADO ----------------------------

# if __name__ == "__main__":

#     archivo_prueba = "C:\Users\andre\Desktop\TFG_Audio\raw_dataset\KICKS\ALL.wav"
    
#     processor = AudioProcessor()
    
#     try:
#         data = AudioProcessor.preprocess_file(archivo_prueba)
#         print(f"Audio procesado shape: {data['audio'].shape}") # [1, 12800]
#         print(f"Mel Spectrogram shape: {data['mel'].shape}")   # [1, 64, 51]
#         print("¡El procesador funciona matemáticamente!")       