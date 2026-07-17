"""
Created on Fri Jan 30 16:39:16 2026

@author: andre
"""

import torch
from torch.utils.data import DataLoader
from tfg_dataset import PercussionDataset
# from tfg_model import CVAE 
# from tfg_loss_NDDSP import nddsp_loss


'''

Función que caraga el tren de datos por batch a procesar por el script de 
entrenamiento.

Definimos get_train_dataloader por separado de get_val_dataloader para separar 
el flujo de ambos conjuntos y evitar problemas de configuración 

'''


def get_train_dataloader(dataset, batch_size, shuffle, num_workers, drop_last):

    loader = DataLoader(
        dataset = dataset,
        batch_size = batch_size, # Procesamos 32 audios de golpe.
        shuffle = shuffle,       # Barajamos los datos.
        num_workers = num_workers, # Usa 2 núcleosl del procesador.
        drop_last  = drop_last     # Ignoramos el último lote si tiene menos de 32.
    )                          

    return loader


def get_val_dataloader(dataset, batch_size, shuffle, num_workers, drop_last):

    loader = DataLoader(
       dataset = dataset,
       batch_size = batch_size, # Procesamos 32 audios de golpe.
       shuffle = shuffle,       # Barajamos los datos.
       num_workers = num_workers, # Usa 2 núcleosl del procesador.
       drop_last  = drop_last     # Ignoramos el último lote si tiene menos de 32.
   )             

    return loader

def get_test_dataloader(dataset, batch_size, shuffle, num_workers, drop_last):

    loader = DataLoader(
       dataset = dataset,
       batch_size = batch_size, # Procesamos 32 audios de golpe.
       shuffle = shuffle,       # Barajamos los datos.
       num_workers = num_workers, # Usa 2 núcleosl del procesador.
       drop_last  = drop_last     # Ignoramos el último lote si tiene menos de 32.
   )              

    return loader
# ----------------------------- PRUEBA DE TIPADO ------------------------------

# if __name__ == "__main__":
    
#     carpeta_datos = "./processed_dataset"
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     dataset = PercussionDataset(data_dir = carpeta_datos)
    
#     tray_datos = get_train_dataloader(dataset, 32,  True,  0, True)
    
#     print(f"DataLoader preparado. {len(tray_datos)} lotes por época.", end ="\n\n" )
    
#     for mel, audio, label in tray_datos:
        
#         print(f"Forma Tensor de Espectr Mel: {mel.shape}") # torch.Size([32, 1, 64, 51])
#         print(f"Forma Tensor de Audio: {audio.shape}") #  torch.Size([32, 1, 12800])
#         print(f"Forma Tensor de Etiqueta de Clase: {label.shape}") # torch.Size([32])
#         print(f"Fromato de Etiqueta de Clase: {label.dtype}") # torch.int64
        
#         break