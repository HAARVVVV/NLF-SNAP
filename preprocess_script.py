"""
Created on Sat Dec 15 12:15:27 2025

@author: andre
"""

import os # Tratamiento del sistema operativo (unicamente carpetas y directorios)
import torch
import glob  # Optimizar la búsqueda de archivos 
from audio_processor import AudioProcessor # Importamos la clase que creamos 
                                           # en audio_processor


'''

Este script aplica el preprocesamiento de 'audio_process.py' en lotes. 
'preprocess_script.py' se encarga de, recorrer automáticamente automáticamente 
la estructura de carpetas del dataset ya descrito, identificar las distintas 
clases de sonidos percusivos que hemos seleccionado (etiquetas) y
aplicar a cada archivo el tratamiento ya mencionado.

Estos datos procesados los guardaremos en el disco como tensor de PyTorch (.pt)
unto con su etiqueta de clase, lo que constituye fundamentalmente un CVAE. De esta 
manera el conjunto de datos queda preparado para su carga eficiente durante el 
entrenamiento del modelo.

################################ OPERACIONES ##################################    

1. for... (1): Buscamos todas los elementos del orig_folder, si dichos elementos 
               son carpetas las registramos como un tipo (kicks, snares, etc.)
               
2. tags: Creamos las etiquetas (tipos de percusiones) en forma de diccionario en
         función del directorio cargado. 
                        
3. for... (2): Bucle principal para el preprocesamiento en bloque. Este discurre
               sobre los tipos de percusiones que hemos localizado.
               
               1. os.makedirs(): Creamos la carpeta de salida si no existe,
                                 no la genera si ya existe. 
                                    
               2. n_tag: Obtenemos el número entero que corresponde a la carpeta
                         que estamos recorriendo.                                                           
    
               3. for... (1): Ahora, por cada tipo de percusión localizado recorremos
                              los respectivos directorios en busca de archivos 
                              sonoros con diferentes extensiones, estos los guardamos
                              en una lista. 
                              
               4. for... (2): Recorremos ahora esta lista de archivos sonoros 
                   
                              1. torch.tensor(): Inyectamos la etiqueta numérica
                                                 como parte del tensor del sonido 
                                                 obtenido. Esto será obligatorio 
                                                 para el entrenamiento de un CVAE, 
                                                 necesitamos saber en todo momento 
                                                 el tipo decada sonido procesado. 
                                                 Formato .long de la etiqueta 
                                                 fundamental para cuestiones 
                                                 posteriores. 
                                                 
                              2. filename: Creamos el nobre del archivo. Copiamos
                                           el nombre del archivo original sin la 
                                           extensión y usamos .pt. 
                              
                              3. torch.save(): Guardamos el tensor en el disco, 
                                               compuesto por (etiqueta, audio, mel). 
                                               Esto es lo que finalmente usaremos 
                                               para alimentar la red. 
                                               

'''

orig_folder = "./raw_dataset"        # .wav
dest_folder = "./processed_dataset" # .pt (pytorch tensor)

def preprocess_dataset():
    
    # Inicializamos el tratamiento del audio
    
    processor = AudioProcessor()
    
    percussions = []
    
    for direct in os.listdir(orig_folder):
        
        if os.path.isdir(os.path.join(orig_folder, direct)):
            percussions.append(direct)
    
    
    print(f"Tipos de percusiones localizadas: {percussions}")
    
    # Creamos los directorios de llegada. 
    
    tags = {perc: indice for indice, perc in enumerate(percussions)}
   
    print(f"Etiquetas: {tags}")
    
    for perc in percussions:
        
        input_folder = os.path.join(orig_folder, perc)
        output_folder = os.path.join(dest_folder, perc)
    
        os.makedirs(output_folder, exist_ok = True)  
        
        n_tag = tags[perc]
        
        '''
        https://www.geeksforgeeks.org/python/python-os-path-join-method/
        https://www.geeksforgeeks.org/python/python-os-makedirs-method/
        '''
        
        sounds = []
        
        for ext in ['*.wav', '*.aif', '*.mp3', '*.FLAC']: 
            
            sounds.extend(glob.glob(os.path.join(input_folder, ext)))
            
            # Usamos .extend() para evitar errores 
            
        print(f"Procesando {perc}: {len(sounds)} sonidos localizados...")
        
        # Procesar y guardar.
        
        for sound in sounds: 
        
            # Preprocesar audio (Normalizar, Trim, Padding, Mel-Spec, etc.)
            
            data = processor.preprocess_file(sound)
            data['label'] = torch.tensor(n_tag, dtype = torch.long)
            
            filename = os.path.splitext(os.path.basename(sound))[0] + ".pt"
            
            torch.save(data, os.path.join(output_folder, filename)) 
            
                
if __name__ == "__main__":
    
    preprocess_dataset() 
    
   
    print("\n¡Procesamiento completado! Los datos listos están en:", dest_folder)