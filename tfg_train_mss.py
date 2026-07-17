"""
Created on Sun Feb 8 16:39:31 2026

@author: andre
"""

# -*- coding: utf-8 -*-

import os
import torch
from torch.optim import Adam # Optimizador de pesos
from tfg_dataloader import get_train_dataloader, get_val_dataloader, get_test_dataloader
from tfg_dataset import PercussionDataset
from tfg_model import CVAE # Encapsula encoder, reparametrización y decoder.
from tfg_NoiseSynth import NoiseSynthesizer # Importa el sintetizador diferenciable.
from tfg_loss_NDDSP_mss import nddsp_loss_mss # Importa la función de pérdida. 
from torch.utils.data import random_split, Subset
import csv
import numpy as np

'''

Construimos la función que entrena un epoch completo. Por tanto solo hará falta 
concatenar tantas de estas funciones como epochs que necesitemos sobre el conjunto
de datos en el entrenamiento. 


################################# PARÁMETROS ##################################

1. model.train(): Pone el CVAE en 'modo entrenamiento'. Por ejemplo en capas como
                  'BatchNorm2d' en el encoder el modelo trabaje con estadísticas
                  del batch actual en vez de las estadísticas acumuladas de 
                  la evaluación.
    
2. synth.train(): Pone el sintetizador de sonido en 'modo entrenamiento'.

3. running_x: Inicializamos las pérdidas acumuladas de cada batch. 


################################# OPERACIONES #################################

1. for...: Cargamos los tensores del dataloader_train por cada batch

    1. .to(device).float(): Movemos los tensores al dispositivo en uso. Usamos 
                            .float() para evitar errores de tipo en el 
                            entrenamiento. 
                            
    2. .to(device).long(): Convierte las etiquetas a un entero largo, que es el 
                           tipo que espera nn.Embedding.

    3. .squeeze(1): Pasamos de [B, 1, Samples] a [B, Samples], que es la entrada
                    requerida por 'tfg_loss_NDDSP.py'
                    
    4. optimizer.zero_grad(): Eliminamos los grandientes del epoch anterior. 

    5. model(mel, label): Llamamos al CVAE.
    
    6. audio_generado: Generamos el audio con el sintentizador. 
    
    7. nddsp_loss_mss: Llamamos al cñalculo de pérdidas. 
    
    8. .backward(): Aplica el algoritmo de backpropagation, calcula los gradientes
                    en todos los pesos. 
                    
    9. optimizer.step(): Actualiza el optimizador Adam con los nuevos gradientes.
    
    10.loss_x.item(): Tomamos el elemento único del tensor como valor. 
                
    11.  if ... or ...: Vamos viendo y comprobando el progreso de los epochs, las 
                        pérdidas, etc. 
                        
'''

def train_one_epoch(
        
        model, 
        noise_synth,
        loader, 
        optimizer, 
        device, 
        beta, 
        epoch_idx,
        epochs,
        audio_length):

    model.train()
    noise_synth.train()

    running_total = 0.0  
    running_mss = 0.0
    running_kld = 0.0

    print(f"########################### Epoch [{epoch_idx+1}/{epochs}] ###########################", end = '\n\n')
    
    for batch_idx, (mel, audio_real, label) in enumerate(loader):
        
        mel = mel.to(device).float()                 # [B, 1, 64, 51]
        audio_real = audio_real.to(device).float()   # [B, 1, 25600]
        label = label.to(device).long()              # [B]

        audio_real = audio_real.squeeze(1)           # [B, 25600]

        optimizer.zero_grad()

        filter_coeffs, mu, logvar = model(mel, label)
        
        audio_generado = noise_synth(
            
            mel_filter_coeffs = filter_coeffs,
            target_audio_length = audio_length
        
            )

        loss_total, loss_mss, loss_kld = nddsp_loss_mss( 
            
            audio_real = audio_real,
            audio_generado = audio_generado,
            mu = mu,
            logvar = logvar,
            beta = beta
        
            )

        loss_total.backward()
        optimizer.step()

        running_total = running_total + loss_total.item()
        running_mss = running_mss + loss_mss.item()
        running_kld = running_kld + loss_kld.item()

        if (batch_idx + 1) % 25 == 0 or (batch_idx + 1) == len(loader):
            
            print(
                
                f"Batch [{batch_idx+1}/{len(loader)}]: "
                f"Loss = {loss_total.item():.3f} "
                f"MSS = {loss_mss.item():.3f} "
                f"KLD = {loss_kld.item():.3f}"
                
                )

    num_batches = len(loader)
    
    print()
    
    print("--------------------------------------------------------------------"
         )
    return (
        
        running_total / num_batches,
        running_mss / num_batches,
        running_kld / num_batches
        
        )



'''

Construimos la función para validar un epoch completo. La validación consiste 
en procesar los datos del conjunto de validación (aquellos no usados en el 
entrenamiento). A diferencia de en el entrenamiento, en la validación no aplicamos
backpropagation, es decir, no calculamos el descenso de gradiente y por tanto 
no actualizamos los pesos por cada iteración del epoch. Calcularemos las mismas
medidas sobre la Validación que sobre el Entrenamiento 


################################# PARÁMETROS ##################################

1. model.eval(): Pone el CVAE en 'modo evaluación'. . 
    
2. synth.eval(): Pone el sintetizador de sonido en 'modo evaluación'.

3. running_x: Inicializamos las pérdidas acumuladas de cada batch. 


################################# OPERACIONES #################################

1. with torch.no.grad(): No calculamos los gradientes.
    
2. for ...:  Cargamos los tensores del dataloader_val por cada batch
    
    1. .to(device).float(): Movemos los tensores al dispositivo en uso. Usamos 
                            .float() para evitar errores de tipo en el 
                            entrenamiento. 
                             
    2. .to(device).long(): Convierte las etiquetas a un entero largo, que es el 
                           tipo que espera nn.Embedding.

    3. .squeeze(1): Pasamos de [B, 1, Samples] a [B, Samples], que es la entrada
                    requerida por 'tfg_loss_NDDSP.py'.
                    
    5. model(mel, label): Llamamos al CVAE.
 
    6. audio_generado: Generamos el audio con el sintentizador. 
 
    7. nddsp_loss_mss: Llamamos al cñalculo de pérdidas. 
 
    8. loss_x.item(): Tomamos el elemento único del tensor como valor.

3. return ...: Devuelve el promedio de cada pérdida por batch.   

'''
     
def validate_one_epoch(model, noise_synth, val_loader, device, beta, audio_length):
   
    model.eval()
    noise_synth.eval()

    running_total = 0.0
    running_mss = 0.0
    running_kld = 0.0

    with torch.no_grad():
        
        for mel, audio_real, label in val_loader:
            
            mel = mel.to(device).float()
            audio_real = audio_real.to(device).float()
            label = label.to(device).long()

            audio_real = audio_real.squeeze(1)

            filter_coeffs, mu, logvar = model(mel, label)

            audio_generado = noise_synth(
               
                mel_filter_coeffs=filter_coeffs,
                target_audio_length=audio_length
            
                )

            loss_total, loss_mss, loss_kld = nddsp_loss_mss(
                
                audio_real=audio_real,
                audio_generado=audio_generado,
                mu=mu,
                logvar=logvar,
                beta=beta
            
                )

            running_total = running_total + loss_total.item()
            running_mss = running_mss + loss_mss.item()
            running_kld = running_kld + loss_kld.item()

    num_batches = len(val_loader)
    
    return (
        
        running_total / num_batches,
        running_mss / num_batches,
        running_kld / num_batches
    )

def test_one_epoch(model, noise_synth, test_loader, device, beta, audio_length):
   
    model.eval()
    noise_synth.eval()

    running_total = 0.0
    running_mss = 0.0
    running_kld = 0.0

    with torch.no_grad():
        
        for mel, audio_real, label in test_loader:
            
            mel = mel.to(device).float()
            audio_real = audio_real.to(device).float()
            label = label.to(device).long()

            audio_real = audio_real.squeeze(1)

            filter_coeffs, mu, logvar = model(mel, label)

            audio_generado = noise_synth(
               
                mel_filter_coeffs=filter_coeffs,
                target_audio_length=audio_length
            
                )

            loss_total, loss_mss, loss_kld = nddsp_loss_mss(
                
                audio_real = audio_real,
                audio_generado = audio_generado,
                mu = mu,
                logvar = logvar,
                beta = beta
            
                )

            running_total = running_total + loss_total.item()
            running_mss = running_mss + loss_mss.item()
            running_kld = running_kld + loss_kld.item()

    num_batches = len(test_loader)
    
    return (
        
        running_total / num_batches,
        running_mss / num_batches,
        running_kld / num_batches
    )
     
'''

Hacemos el main() porque estábamos teniendo problemas con el fork de procesos
en Pytorch. 

################################# PARÁMETROS ##################################

1. data_dir: Directorio de datos preprocesados.

2. save_dir: Directorio donde guardaremos la información del entrenamiento. 

3. batch_size: Número de señales que procesaremos en simultaneo. 

4. epochs: Interaciones sobre el conjunto de datos de entrenamiento que vamos a 
           realizar. 
 
5. lr: Learning rate, parámetro interno para la tasa de aprendizaje del modelo. 
    
6. beta: Parámetro interno de la divergencia KL. 
    
7. num_percs: Número de clases de sonidos diferenciados al inicio, coherente 
                con el dataset inicial.
               
8. latent_dim: Número de dimensiones del espacio latente,  coherente con la  
               dimensión latente definida en el encoder y el decoder. 

9. n_mels: Número de bandas Mel, coherente con el dataset inicial. 

10. audio_length: Longitud en samples de los audios procesados y generado,
                  coherente con el dataset inicial. 

11. device (cuda): ""
      
12. history: Inicializamos el historial para guardar los checkpoints del modelo
             y poder crear tablas y gráficos con los que podremos comparar 
             resultados y poder argumentar conclusiones sobre el modelo. 
             
             
################################# OPERACIONES #################################
      
1. x_dataload: Llamamos al PercussionDataset en instancias separadas para cada 
               cada split del entrenamiento. Asignamos Augment según queramos 
               o no aplicar el data augmentation on the y fly creado en 
               dataset. 

2. x_size: Realizamos una división 80/10/10 del tamaño entre los conjuntos de 
           entrenamiento, validación y test. 
                   
3. generator: Cargamos una seed concreta por temas de consistencia en componentes
              aleatorias a lo largo de distintas pruebas. La seed fija las fuentes 
              de azar para que las corridas sean comparables, mejorando la 
              reproducibilidad. 

4. random_split: Asignamos los índices de los datos que queremos en cada conjunto, 
                 en relación con los tamaños calculados con la seed generada. 

5. x_dataset: Asignamos definitivamente los sonidos del dataset correspondiente
              a los indices previamente calculados, usamos Subset para no usar 
              el mismo dataset en todos los conjuntos, ya que no queremos 
              contaminar la validación o el test con el data augmentation. 
              
5. get_x_dataloader: Llamamos a los dataloaders de 'tfg_dataloader.py' 
                     para cargar los datos a la memoria del sistema del train, 
                     la val y el test, crea la cola de batches por epoch para 
                     estos procesos.

6. model: Cargamos el CVAE.

7. noise_syntn: Cargamos el sintetizador de ruido. 

8. Optimizer: Usamos el optimizador Adam para actualizar los pesos del CVAE y 
              del sistentizador de sonido mediante backpropagation.
                     
9. best_loss: Inicializamos la mejor pérdida como el valor infinito, cualquier
              interación inicial mejorará el resultado. 

10. for ...: Empezamos el bucle principal de entrenamiento, actúa sobre el 
             número de Epochs. 

             1. checkpoint: Por cada epoch guardamos un punto de control en un 
                            tensor con los diversos estado, funciones de pérdida 
                            aprendidos relevantes, pesos de convoluciones, 
                            embeddings y capas lineales. Iremos actualizando 
                            los mejores resultados.              
                            
                            1. config: Guardamos los hiperparámetros del modelo
                                       estos nos servirán para hacer la 
                                       reconstrucción del audio sin depender de 
                                       código ajeno.
                                       
             2. .state_dict(): Devuelve un diccionario de Python con el estado 
                               actual.

             3. torch.save(): Guardamos el checkpoint. 

             4. history.append: Guardamos los estados del diccionario pertinentes
                                para la construcción de gráficos.   
                                
             5. if val_total < best_loss: Si mejoramos la pérdida gaurdamos el 
                                          checkpoint como el mejor estado. 
                                          Elegimos el modelo por su rendimiento 
                                          en datos no usados del Val para actualizar 
                                          los pesos, no por lo bien que se ajusta
                                          a los datos de entrenamiento.
                                                
11. .load_state_dict(): Señalamos a la ruta del mejor checkpoint que hemos 
                        encontrado durante el entrenamiento y cargamos los pesos 
                        correspondientes del modelo y del sintetizador. 
                        
12. test_one_epoch: Llamamos por último al test. !!!DESARROLLAR¡¡¡

13. with ...: Guardamos el history en un .CSV para su uso posterior. 

'''


def main():
    
    data_dir = "./processed_dataset"
    save_dir = "./checkpoints"

    os.makedirs(save_dir, exist_ok = True)

    batch_size = 32
    epochs = 150
    lr = 1e-4
    
    beta = 1.0

    num_percs = 5
    latent_dim = 16

    n_mels = 64
    # num_frames = 51
    audio_length = 25600
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Entrenando en: {device}", end='\n\n')

    history = []

    train_dataload = PercussionDataset(data_dir, augment = True)
    val_dataload = PercussionDataset(data_dir, augment = False)
    test_dataload = PercussionDataset(data_dir, augment = False)
    
    # full_dataset = PercussionDataset(data_dir = data_dir)
    
    total_length = len(train_dataload)
    train_size = int(0.8 * total_length)
    val_size = int(0.1 * total_length)
    test_size = total_length - train_size - val_size
    
    generator = torch.Generator().manual_seed(0)

    train_idx, val_idx, test_idx = random_split(range(total_length), 
                                    [train_size, val_size, test_size], 
                                    generator = generator)

    train_dataset = Subset(train_dataload, train_idx.indices)
    val_dataset = Subset(val_dataload, val_idx.indices)
    test_dataset = Subset(test_dataload, test_idx.indices)
    
    train_loader = get_train_dataloader(
              dataset = train_dataset, 
              # data_dir = data_dir,
              batch_size = batch_size,
              shuffle = True, # Mezcla el orden de los datos por cada epoch. 
              num_workers = 0, # Aumento carga de procesado. 
              drop_last = True

              )
    
    val_loader = get_val_dataloader(
            dataset = val_dataset, 
            # data_dir = data_dir,
            batch_size = batch_size,
            shuffle = False, # Mezcla el orden de los datos por cada epoch. 
            num_workers = 0, # Aumento carga de procesado. 
            drop_last = False

            )
    
    test_loader = get_test_dataloader(
              dataset = test_dataset, 
              # data_dir = data_dir,
              batch_size = batch_size,
              shuffle = False, # Mezcla el orden de los datos por cada epoch. 
              num_workers = 0, # Aumento carga de procesado. 
              drop_last = True

              )

    
    print(f"Lotes por Epoch (TRAIN): {len(train_loader)}")
    print(f"Lotes por Epoch (VAL): {len(val_loader)}")
    print(f"Lotes por Epoch (TEST): {len(test_loader)}", end = '\n\n')
    print("--------------------------------------------------------------------", end = '\n\n'
         ) 
   
    
    model = CVAE(
        num_percs = num_percs,
        latent_dim = latent_dim).to(device) # Movemos el proceso para que viva 
                                            # en el mismo lugar que los tensores.

    noise_synth = NoiseSynthesizer(
        window_size = 2048,
        hop_size = 512,
        n_mels = n_mels,
        sample_rate = 32000).to(device) 
    
    
    optimizer = Adam(list(model.parameters()) + list(noise_synth.parameters()), lr = lr)

    # ########################### BUCLE PRINCIPAL #############################

    best_loss = float("inf")
     
    
    for epoch in range(epochs):
        
        # ############ PRUEBA ANNEALING LINEAL EN BETA KLD ############
        
        beta_epoch = beta # Sustitución directa.
        
        # beta_epoch =  min(1.0, (epoch + 1) / 5) # Annealing lineal.
        
        # cycle_length = 20
        # cycle_pos = (epoch % cycle_length) / (cycle_length - 1) 
        
        # beta_epoch = 0.5 * (1 - np.cos(np.pi * cycle_pos)) # Annealing cíclico.
        
        train_total, train_mss, train_kld = train_one_epoch(
            
            model = model,
            noise_synth = noise_synth,
            loader=train_loader,
            optimizer =optimizer,
            device =  device,
            beta = beta_epoch,
            epoch_idx = epoch,
            epochs = epochs,
            audio_length = audio_length
           
            )
        
        
        val_total, val_mss, val_kld = validate_one_epoch(
            
            model = model, 
            noise_synth = noise_synth, 
            val_loader = val_loader, 
            device = device, 
            beta = beta_epoch, 
            audio_length = audio_length
        
            )
        
        # test_total, test_mss, test_kld = test_one_epoch(
            
        #     model = model, 
        #     noise_synth = noise_synth, 
        #     test_loader = test_loader, 
        #     device = device, 
        #     beta = beta_epoch, 
        #     audio_length = audio_length
        
        #     )
        
        print(
            
            f"Train Total = {train_total:.4f} | "
            f"Train MSS = {train_mss:.4f} | "
            f"Train KLD = {train_kld:.4f}"
            
            )
        print("--------------------------------------------------------------------"
             ) 
        
        print(
            
            f"Val Total = {val_total:.4f} | "
            f"Val MSS = {val_mss:.4f} | "
            f"Val KLD = {val_kld:.4f}"

            )
        
        print("--------------------------------------------------------------------", end ='\n\n'
             ) 
        
       
       
        
        checkpoint = {
            
            "epoch": epoch + 1, # Empieza enumerando en el 0
            "model_state_dict": model.state_dict(),
            "noise_synth_state_dict": noise_synth.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "beta_epoch": float(beta_epoch),
            "train_total": train_total,
            "train_mss": train_mss,
            "train_kld": train_kld,
            "val_total": val_total,
            "val_mss": val_mss,
            "val_kld": val_kld,
            # "test_total": test_total,
            # "test_mss": test_mss,
            # "test_kld": test_kld,
            "config": {"num_percs": num_percs,
                       "latent_dim": latent_dim,
                       "n_mels": n_mels,
                       "audio_length": audio_length,
                       "sample_rate": 32000,
                       "window_size": 2048,
                       "hop_size": 512,
                       "data_dir": data_dir}
            
        }

        torch.save(checkpoint, os.path.join(save_dir, f"nddsp_epoch_{epoch+1:03d}.pt"))
        
        history.append({"epoch": epoch + 1,
                        "beta_epoch": float(beta_epoch),
                        "train_total": train_total,
                        "train_mss": train_mss,
                        "train_kld": train_kld,
                        "val_total": val_total,
                        "val_mss": val_mss,
                        "val_kld": val_kld
                        
                        })
            
        if val_total < best_loss:
            
            best_loss = val_total
            torch.save(checkpoint, os.path.join(save_dir, "best_nddsp.pt"))

    best_checkpoint_path = os.path.join(save_dir, "best_nddsp.pt")  
    checkpoint = torch.load(best_checkpoint_path)

    model.load_state_dict(checkpoint["model_state_dict"])
    noise_synth.load_state_dict(checkpoint["noise_synth_state_dict"])

    test_total, test_mss, test_kld = test_one_epoch(
        
        model = model,
        noise_synth = noise_synth,
        test_loader = test_loader,
        device = device,
        beta = checkpoint["beta_epoch"], # evaluamos el modelo final con el mismo 
                                         # criterio con el que fue seleccionado.
        audio_length = audio_length

    )


    print("####################################################################")  
    
    print(
         
        f"Test Total = {test_total:.4f} | "
        f"Test MSS = {test_mss:.4f} | "
        f"Test KLD = {test_kld:.4f}"

        )
     
    print("####################################################################", end ='\n\n'
          )   
    
    save_dir = "./CVAE_CSV"
    
    summary_csv_path = os.path.join(save_dir, "experiment_summary.csv")
    file_exists = os.path.exists(summary_csv_path)
    
    with open(summary_csv_path, mode = "w", newline = "", encoding = "utf-8") as f:
      
        writer = csv.writer(f)
        if not file_exists:
            
            writer.writerow([
             
                "experiment_name",
                "augmentation",
                "epochs",
                "batch_size",
                "learning_rate",
                "latent_dim",
                "best_epoch",
                "best_beta_epoch",
                "best_val_total",
                "best_val_mss",
                "best_val_kld",
                "test_total",
                "test_mss",
                "test_kld"
            ])
            
        writer.writerow([
            
            "aug_cyclic_beta_run_01",
            True,
            epochs,
            batch_size,
            lr,
            latent_dim,
            checkpoint["epoch"],
            checkpoint["beta_epoch"],
            checkpoint["val_total"],
            checkpoint["val_mss"],
            checkpoint["val_kld"],
            test_total,
            test_mss,
            test_kld
            
        ])
    
    history_csv_path = os.path.join(save_dir, "training_history.csv") # Cremaos ruta.

    with open(history_csv_path, mode = "w", newline = "", encoding = "utf-8") as f:

        writer = csv.writer(f)
        writer.writerow(["epoch", 
                         "beta_epoch", 
                         "train_total",
                         "train_mss", 
                         "train_kld",
                         "val_total", 
                         "val_mss", 
                         "val_kld"])

        for row in history:
            writer.writerow([
                row["epoch"],
                row["beta_epoch"],
                row["train_total"],
                row["train_mss"],
                row["train_kld"],
                row["val_total"],
                row["val_mss"],
                row["val_kld"]
                
                ])

if __name__ == "__main__": # COLAPSABA DE OTRA FORMA
    main()
    

