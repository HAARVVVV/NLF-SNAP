"""
Created on Fri Jan 30 16:39:16 2026

@author: andre
"""

import torch
import torchaudio
import os
import glob
import random
from torch.utils.data import Dataset, DataLoader
from audiomentations import Compose, OneOf, PitchShift, AddColorNoise, TanhDistortion, PeakingFilter, Shift, PolarityInversion

class PercussionDataset(Dataset):
   
    '''
    
    Clase Dataset personalizada para cargar audios percusivos y sus espectrogramas 
    Mel.
    
    '''
    
    def __init__(self, data_dir, augment):
       
        '''
        
        ############################# PARÁMETROS ##############################
        
        1. data_dir: Ruta a la carpeta donde están guardados los .pt prepreprocesados.
        
        2. augment: Bool que determina si aplicamos data augmentation al conjunto 
                    de datos para el entrenamiento.
                    
        ############################# OPERACIONES #############################
        
        1. .file_paths:  Buscamos  los archivos .pt dentro de cada carpeta
                         y subcarpetas. Muy similar a la búsqueda en 
                         'preprocess_script.py'. Escribimos un error en caso de 
                         no encontrar archivos. 

        2. .file_paths.sort(): Organizamos la lectura de los archivos para evitar 
                               incosnsitencias a lo largo del procesamiento y 
                               entrenamiento.
                                   
        3. .wave_mod: En caso de que decidamos aplicar el aumento de datos pasamos 
                      a aplicar el pipeline de modificaciones para el data 
                      augmentation. Todos los siguiente efectos se evalúan uno 
                      por uno, en orden y de manera independiente respecto al resto.
                        
                    1. Shift: Desplaza el audio sutilmente, tomamos un rango del 
                              2% a la izquierda y 2% a la derecha del núnmero 
                              de samples, lo hemos seleccionado de esta forma 
                              porque coincide con el hop length que hemos definido
                              desde un inicio, con lo que como máximo desplazaremos
                              la onda un frame adelante o atrás. 
                              
                              25600 * (2/100) = 512 samples 
                              
                              Es entonces equivalente a un desplazamiento de, 
                              como maximo, 16 milisegundos. 
                              
                              rollover = True trata al tensor bajo una estructura
                              de datos circular, así evitamos hacer desplazamientos 
                              que añadan o borren frames del audio preprocesado 
                              original y a la vez realmente no perdemos información. 
                              
                              Esta aplicación es a su vez realmente útil ya que 
                              permitiremos a la red aprender más allá de que la 
                              transiente esté en el primer o segundo frame, creo 
                              que con este desplazamiento podría generalizar
                              mejor. 
                              
                              audio_real = [x_0, x_1, ..., x_49, x_50]
                              
                              shift_aureal (-0.02) = [x_1, x_2, ..., x_50, x_0]
                              
                              shift_aureal (0.02) = [x_50, x_0, ..., x_48, x_49]
                              
                        
                    3. OneOf: Se asegura de que se aplica solo uno de los efectos 
                              que contiene. He deciddio que se excluyan mutuamente 
                              para no sobrecargar ni ensuciar en exceso 
                              las señales y perder expresividad en el 
                              entrenamiento. Estos son: 
                                  
                              1. PitchShift: Modifica el tono del audio, este 
                                             tono depende directamente de las 
                                             frecuencias que podemos encontrar
                                             y se refiere de alguna forma a lo 
                                             agudo o grave que percibimos dicho
                                             sonido. 
                                             
                                             Acotaremos la modificación del tono
                                             a 3 semitonos en cada dirección,
                                             
                                             Existen dos formas para el PitchShift, 
                                             pero hemos descartada el uso de 
                                             voocoders de fase debido a la 
                                             degradación acústica de transitorios 
                                             que suponen. Así, hemos elegido el 
                                             método de signalsmith stretch, este 
                                             encaja a la percepción con el 
                                             tratamiento de percusiones ya que 
                                             analiza y ancla las transientes y 
                                             aplica la transformación de tono 
                                             sobre el resto de la onda. 
                                             
                                             Para la transformación del tono 
                                             sigue el siguiente proceso:
                                                 
                                             1. Transformación STFT. 
                                             
                                             2. Interpolación de la matriz obtenida
                                                para subir o bajar la frecuencia 
                                                fundamental y sus armónicos tantos
                                                semitonos como indicado. 
                                                
                                            3. Ajuste de fase. 
                                            
                                            4. Devolvemos el tensor sobre el 
                                               dominio del tiempo con ISTFT. 
                                               

        --------------------------------- NOTA -------------------------------- 
        
        En la teoría musical occidental que todos conocemos, una octava, una unidad 
        completa de notas diferentes, viene dividida en 12 semitonos. En términos
        matemáticos subir una octava es lo mismo que doblar la frecuencia 
        fundamental de la onda onda. Una sinusoidal de 440 Hz la conocemos como 
        un La 4, es decir, la nota La de la 4ª octava, por ende el mismo tipo de 
        onda a 880 Hz será también un La, pero de la octava superior.

        Nosotros no conocemos ni nos importa realmente el tono orginal del audio
        que esta siendo aumentado, como se puede intuir por la explicación anterior, 
        las transformaciones sobre el tono función de forma proporcional relativa,
        la distancia matemática exacta de un único semitono se calcula dividiendo
        ese factor de 2 en 12 partes logarítmicas iguales:

        R_{st} = 2^{1/12} \approx 1.0594

        De tal manera el tono modificado corresponde con:

        f_new = f_old * 2 ^{n/12}; n = nº de semitonos a desplazar. Respecto a 
        cada una de las frecuencias del análisis de Fourier del sonido entero.  
        
        -----------------------------------------------------------------------
        
                              2. TanhDistortion: Este modo de distorsion o saturación 
                                                 aplica especificamente una función 
                                                 tangente hiperbólica sobre el 
                                                 rango dinámico, superando los 
                                                 límites establecidos en el 
                                                 preprocesamiento aplicando lo 
                                                 que se conoce como 'soft clip'
                                                 a medida que nos acercamos a 
                                                 sendos límites, muy importante, 
                                                 sin excedernos de los mismos. 
                                                 
                                                 lim_{x \to \infty} \tanh(x) = 1 
                                                 
                                                 lim_{x \to -\infty} \tanh(x) = -1 
                                   
                                                 lo que hace exactamente es aplicar 
                                                 una ganancia sobre la amplitud, 
                                                 lo que podemos llamar overdrive, 
                                                 y luego aplicar tanh, es decir
                                                 
                                                 audio_mod = tanh(k * audio_real)
                                                 
                                                 k = overdrive. 
                                                 
                                                 Es así como obtenemos esa 
                                                 sensación de aplastamiento o 
                                                 'SATURACIÓN' del sonido. 
                           
                                                 
                    4. AddColorNoise: Como última transformación posible hemos 
                                      pensado en añadir 'ruido rosa'. Es el estándar 
                                      absoluto en tratamiento de audio porque 
                                      casa perfectamente con cualquier señal sin 
                                      enmascarar las transientes. min_f_decay 
                                      selecciona la variabilidad de la potencia 
                                      de las frecuencias generadas del ruido. 
                                      
        --------------------------------- NOTA --------------------------------
        
        El ruido no deja de ser una distribución relativamente homogénea de la 
        potencia sobre todas las frecuencias del espectro audible. Es decir, para 
        entenderlo, el ruido blanco presenta potencia constante a lo largo de 
        todas las frecuencias.
        
        En general cualquier tipo de ruido viene definido por: 
            
            P(f) \propto \frac{1}{f^\beta}
            
        Es decir, la potencia es inversamente proporciona a la frecuencia. En 
        nuestro caso hemos decidido usar 'ruido rosa'; es inmediato que el ruido 
        blanco corresponde a  \beta = 0, por otro lado el ruido rosa es con 
        \beta = 1. Este ruido resulta mas natural, ya que compensa la alta energía 
        de las frecuencias mayores.
        
        -----------------------------------------------------------------------
        
        4. .mel_modifiers: Para simular variaciones en las propiedades físicas y
                           tímbricas como, cualidades del micrófono, material de 
                           construcción  del instrumento, resonancia y tamaño de 
                           las cajas percusivas, color, etc. hemos pensado en implementar
                           una técnica de deformación frecuencial inspirada por 
                           los Vocoder. Para ello usamos el llamado 'warping'. 
                           
                           Realmente el warping consiste en deformar las propias 
                           frecuencias sobre la onda cruda, o aplicar una 
                           interpolación no lineal sobre la STFT, cabe recalcar 
                           que ambos métodos son altamente costosos en cómputo. 
                           
                           La forma eficiente que hemos decidido usar se basa en 
                           modificar directamente las frecuencias de los filtros 
                           triangulares del banco Mel. En lugar de transformar la 
                           señal, modificamos lo que captura dicha señal adaptando 
                           más bien la percepción sensorial en lugar de la señal
                           en si misma.
                           
                           Entonces, matemáticamente hablando, el warping no deja 
                           de ser la alteración del límite superior de frecuencia
                           en la construcción del banco de filtros Mel.
                           
                           Con esto conseguimos moldear la distribuciónde los filtros 
                           triangulares, modificando de manera NO LINEAL la energía 
                           frecuencial del espectrograma (de los armónicos) 
                           sin alterar la resolución temporal.
                           
            
        '''
        
        self.file_paths = glob.glob(os.path.join(data_dir, '**', '*.pt'), recursive = True)
            
        self.file_paths.sort() 
        
        print(f'Dataset inicializado: {len(self.file_paths)} archivos encontrados.', end = '\n\n')

        self.augment = augment
        
        if self.augment == True:
            
            self.wave_mod = Compose([
                
                Shift(min_shift=-0.02,       
                      max_shift=0.02,        
                      shift_unit="fraction", 
                      rollover = True,
                      p = 0.15),

                OneOf([PitchShift(min_semitones = -3, max_semitones = 3,
                                  method = 'signalsmith_stretch', p = 1.0),
                       
                       TanhDistortion(min_distortion = 0.01, max_distortion = 0.08, p=1.0)], 
                      
                      p=0.5),
                
                AddColorNoise(min_snr_db=15.0, max_snr_db=35.0, 
                              min_f_decay = -3.01, max_f_decay= -3.01, 
                              p = 0.3)

            
            ])
            
            self.mel_modifiers = [
            
                torchaudio.transforms.MelSpectrogram()(sample_rate = 32000, 
                                                     n_fft = 2048,
                                                     hop_length = 512, 
                                                     n_mels = 64, 
                                                     f_min = 20, 
                                                     f_max = 14000),
                
                torchaudio.transforms.MelSpectrogram(sample_rate = 32000, 
                                                     n_fft = 2048,
                                                     hop_length = 512, 
                                                     n_mels = 64, 
                                                     f_min = 20, 
                                                     f_max = 15000),
                
                torchaudio.transforms.MelSpectrogram(sample_rate = 32000, 
                                                     n_fft = 2048,
                                                     hop_length = 512, 
                                                     n_mels = 64, 
                                                     f_min = 20, 
                                                     f_max = 16000)
                
                ]
            
            self.power_to_db = torchaudio.transforms.AmplitudeToDB(top_db = 80)
            
            
    def __len__(self):
       
        '''
        
        Devuelve el número total de archivos en el dataset.
        PyTorch necesita esta función para calcular cuándo termina un Epoch.  
        
        '''
        
        return len(self.file_paths)

    def __getitem__(self, idx):
        
        '''
        
        __getitem__ representa el núcleo de PercussionDataset. 
        
        
        ############################## VARIABLES ##############################
        
        1. idx: Índice del archivo de interés. 
        
        
        ############################# OPERACIONES #############################
        
        1. file_path: Obtenemos la ruta del archivo específico.
        
        2. torch.load():  Cargamos el diccionario que guardamos durante el 
                          preprocesamiento.
        
        3. mel_spectr: Extraemos el tensor del espectrograma Mel. ([1, 64, 51])
        
        4. audio_real: Extraemos el tensor del audio orignial. ([Samples])
         
        5. label: Extraemos el tensor asociado a la etiqueta. ([1])
        
        '''
        
        file_path = self.file_paths[idx]
        
        data = torch.load(file_path, weights_only=False) 
        
        audio_real = data['audio'].float()
        label = data['label'].long()
        
        if self.augment:
            
            audio_real_np = audio_real.squeeze().numpy()  
            audio_aug = self.wave_mod(samples = audio_real_np, sample_rate = 32000)
            audio_real = torch.from_numpy(audio_aug).unsqueeze(0).float()
        
            if random.random() < 0.1:
              
                fade = torchaudio.transforms.Fade(fade_in_len = random.randint(0, 20), 
                                                  fade_out_len = random.randint(0, 4000),
                                                  fade_shape = 'exponential')
                audio_real = fade(audio_real)
            
            mel_modified = random.choice(self.mel_modifiers)
            mel_spectr = mel_modified(audio_real)
                
            mel_spectr = self.power_to_db(mel_spectr)
            mel_spectr = (mel_spectr + 80) / 80 # Normalización
            mel_spectr = torch.clamp(mel_spectr, 0, 1) # Asegura rango [0,1]
        
             
        else: 
            
            mel_spectr = data['mel'].float()
            
        return mel_spectr, audio_real, label # Devolvemos los tensores