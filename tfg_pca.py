"""
Created on Wed Apr 22 15:38:29 2026

@author: andre
"""

import os
import numpy as np
import pandas as pd
import torch

from sklearn.decomposition import PCA
from torch.utils.data import random_split, Subset

from tfg_dataset import PercussionDataset
from tfg_dataloader import get_test_dataloader
from tfg_model import CVAE

'''

PCA es una técnica de reducción de dimensionalidad mediante proyecciones lineales
sobre espacios reducidos, transforma un conjunto de variables posiblemente 
correlacionadas en nuevas direcciones ortogonales a las que llamamos 'COMPONENTES 
PRINCIALES', estas direcciones son seleccionadas explicitamente de forma ordenada 
para capturar la mayor varianza posible de los datos seleccionados del conjunto de
test.
 
La primera componente principal maximiza la varianza proyectada, la segunda 
maximiza la varianza restante bajo ortogonalidad con respecto a la primera, y 
así sucesivamente.

vamos a intentar no ser redundante con muchos procedimientos que ya se han repetido
a lo largo de la implementación del proyecto. 


################################# OPERACIONES #################################

1. Como tenemos que hacer un proceso de inferencia sobre el espacio latente del 
   conjunto del 'TEST', tenemos que cargar el estado del mejor checkpoint y 
   recuperar la configuración inial del entrenamiento, para ello la inicializamos 
   con la misma seed y splits con los que establecemos el entrenamiento. 
   
2. model: Cargamos el modelo en modo evaluación. 

3. test_loader: Cargamos los archivos del dataset atendiendo a los indices.

4. mu_batches, label_batches, sample_indices: Inicializamos los puntos de interés 
                                              del PCA. 
                                              
5. for...: Recorremos los archivos del test, guardamos las distribuciones y 
           las etiquetas del tipo de percusión pertinentes. 

           1. mu, logvar: Calculamos los parámetros de la distribución latente 
                          a priori de cada sonido. Solo usaremos mu como representante
                          determinista de cada representación latente para el PCA.
                          
                          Recordemos que son tensores de dimensión 16. 
                          
           2. .numpy().cpu(): Queremos transformar los tensores de la distribución 
                              latente a priori a arrays, en GPU estaba dando 
                              problema de ejecución así que hemos decidido asignarlo 
                              dentro de la CPU por defecto.

           mu_batches = [[mu1_0, \dots, mu1_15], \dots, [mun_0, \dots, mun_15]]
           
           label_batches = [label1, \dots, labeln]
           
6. latent_mu, labels: Concatenamos los elementos de la lista ya creada pero en el
                      propio array, creamos un ndarray de 2 dimensiones:
                      
                      1. axis = 0: Queremos asegurarnos de que forman una matriz
                                   tratando a las variables del espacio latente 
                                   como filas, siendo por tanto cada columna 
                                   cada una de las dimensiones latentes.

                      latent_ mu = [mu1_0, \dots, mu1_15, 
                                    \dots, \dots, \dots, 
                                    mun_0, \dots, mun_15]
                      
7. latent_columns: Nombramos las columnas de la tabla que queremos generar. 
                   Como hemos explicado en el punto anterior, latent_mu.shape[1]
                   se refiere a las columnas, es decir, a las dimensiones del 
                   espacio latente.


8. latent_df: Creamos la estructura tabular de Pandas con la matriz calculada 
              y nombrando las columnas con los indices de latent_columns.

                  
9. pca: Inicializamos el PCA mediante la funcion de scikit learn. 

10.pcs: Ajusta el PCA a los datos, es decir, calculamos las direcciones principales 
        de máxima varianza, y luegotransformamos cada vector latente a sus nuevas 
        coordenadas en el espacio proyectivo. Es decir, contiene la proyección 
        bidimensional de cada muestra sobre las componentes prinicaples.                             
        
        1. explained: Indica cuanto explica cada eje principal independientemente.
                      Guardamos el porcentaje de varianza explicada por cada 
                      componente principal, en este caso PC1 y PC2. Es decir, mide
                      la dispersión capturada por los ejes principales.
                      
                      Cuanto mayor sea esta cantidad para una componente principal
                      mejor capturará la información del espacio de mayor dimensión. 
                      
        2. cumulative_2d: Sumamos la varianza explicada de las componentes 
                          princiaples, es decir, nos indica el porcentaje total
                          de información se conserva ante el colapso del 
                          espacio latente a los ejes indicados.
                      
        3. loadings: Recogen los pesos con los que las variables originales 
                     forman cada componente principal. Un loading positivo o 
                     negativo indica la dirección de la relación con cada 
                     una de las componentes. 
                        
10. loadings_df: Los loadings calculan cuánto contribuye cada variable original
                 del espacio latente, es decir, cada dimensión mun_i, a cada
                 componente principal final.

11. summary_df: Una tabla-resumen de una sola fila con metadatos del análisis, 
                epoch óptimo, beta del mejor checkpoint, dimensión latente, número
                de muestras de test y porcentaje de varianza explicada por PC1,
                PC2 y su suma.  

'''
def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = "./checkpoints/best_nddsp.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    save_dir = "./CVAE_CSV"
    os.makedirs(save_dir, exist_ok = True)
    
    config = checkpoint["config"]

    data = config["data_dir"]
    num_percs = config["num_percs"]
    latent_dim = config["latent_dim"]

    model = CVAE(num_percs = num_percs,
                 latent_dim = latent_dim).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    batch_size = 32

    train_dataload = PercussionDataset(data, augment = True)
    val_dataload = PercussionDataset(data, augment = False)
    test_dataload = PercussionDataset(data, augment = False)

    total_length = len(train_dataload)
    train_size = int(0.8 * total_length)
    val_size = int(0.1 * total_length)
    test_size = total_length - train_size - val_size

    generator = torch.Generator().manual_seed(0)

    train_idx, val_idx, test_idx = random_split(range(total_length),
                                                [train_size, val_size, test_size],
                                                generator=generator
                                                
                                                )

    test_dataset = Subset(test_dataload, test_idx.indices)

    test_loader = get_test_dataloader(dataset = test_dataset,
                                      batch_size = batch_size,
                                      shuffle = False,
                                      num_workers = 0,
                                      drop_last = False
    
                                      )

    mu_batches = []
    label_batches = []
    sample_indices = []

    with torch.no_grad():
        
        idx = 0

        for mel, audio_real, label in test_loader:
            
            mel = mel.to(device).float()
            label = label.to(device).long()

            mu, logvar = model.encoder(mel, label)

            batch_size = mel.size(0)

            mu_batches.append(mu.cpu().numpy()) #[mu1_0, \dots]]
            label_batches.append(label.cpu().numpy())
            sample_indices.extend(range(idx, idx + batch_size))

            idx = idx + batch_size

    latent_mu = np.concatenate(mu_batches, axis = 0)
    labels = np.concatenate(label_batches, axis = 0)

    latent_columns = [f"mu_{i}" for i in range(latent_mu.shape[1])]
    
    latent_df = pd.DataFrame(latent_mu, columns = latent_columns)
   
    latent_df.insert(0, "label", labels)
    latent_df.insert(0, "sample_idx", sample_indices)
    
    latent_df.to_csv(os.path.join(save_dir, "latent_mu_test.csv"), 
                     index = False, 
                     encoding = "utf-8")
    
    pca = PCA(n_components = 2, random_state = 0)
    pcs = pca.fit_transform(latent_mu)

    explained = pca.explained_variance_ratio_ * 100.0
    cumulative_2d = explained.sum()
    loadings = pca.components_.T


    pca_df = pd.DataFrame({"sample_idx": sample_indices,
                           "label": labels,
                           "pc1": pcs[:, 0],
                           "pc2": pcs[:, 1]
    
                            })

    pca_df.to_csv(os.path.join(save_dir, "latent_pca_test.csv"),
                  index = False,
                  encoding="utf-8")


    loadings_df = pd.DataFrame({"latent_dim_idx": np.arange(latent_mu.shape[1]),
                                "loading_pc1": loadings[:, 0],
                                "loading_pc2": loadings[:, 1]
    
                                })

    loadings_df.to_csv(os.path.join(save_dir, "latent_pca_loadings.csv"),
                       index = False,
                       encoding = "utf-8")
    

    summary_df = pd.DataFrame([{"best_epoch": checkpoint["epoch"],
                                "best_beta_epoch": checkpoint["beta_epoch"],
                                "latent_dim": latent_dim,
                                "n_test_samples": len(sample_indices),
                                "pc1_explained_variance_percent": explained[0],
                                "pc2_explained_variance_percent": explained[1],
                                "pc1_pc2_cumulative_percent": cumulative_2d

                                }])

    summary_df.to_csv(os.path.join(save_dir, "latent_pca_summary.csv"),
                      index = False,
                      encoding = "utf-8") 

if __name__ == "__main__":
    
    main()

