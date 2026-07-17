# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 18:50:13 2026

@author: andre
"""

# infer_nddsp.py

import os
import torch
import soundfile as sf
import numpy as np 
import torchaudio

from tfg_model import CVAE
from tfg_NoiseSynth import NoiseSynthesizer
from tfg_dataset import PercussionDataset
import matplotlib.pyplot as plt

torch.manual_seed(0) # SEEDING
      
    
'''

Este es el script que traduce los tensores obtenidos del entrenamiento en sonidos. 
Veamos las funciones que lo componen: 
    
1. load_system(): Función que carga el mejor checkpoint y carga las instancias 
                  del modelo y el sintetizador correspondientes a dicho mejor 
                  checkpoint en la memoria en la memoria:
                  
                  ######################## OPERACIONES ########################
                  
                  
                  1. config: Cargamos la configuración de hiperparámetros como 
                             los itemos de un diccionario. 
                             
                  2. .load_state_dict(): toma el diccionario de parámetros 
                                         guardado dentro de chekpoint correspondiente
                                         a la estructura indicada, y lo copia 
                                         sobre dicha arquitectura. Es decir, 
                                         estamos introduciendo los pesos yaa 
                                         aprendidos en el checkpoint sobre el 
                                         esqueleto vacío. 
                  
                  3. .eval(): Como queremos reconstruir la señal ponemos los 
                              módulos en modo evaluación, así no seguimos con 
                              el entrenamiento, de otro modo funciones como 
                              BatchNorm2d() cambiarían su comportamiento y 
                              podrían seguir generando resultados inconsistentes. 
                              

2. @torch.no_grad(): Al estar tratando la reconstrucción mediante inferencia
                     no es necesario calcular ni almacenar el gradiente al no 
                     llamar nunca a backward. 
                     

3. reconstruct_sound: Esta función se encarga de hacer la reconstrucción de audios
                      de ejemplo a partir del mejor checkpoint obtenido. 
                      
                      ###################### OPERACIONES ######################
                      
                      1. while...: Recorremos la cantidad de sonidos que queremos
                                   hacer. 
                      
                      2. dataset[i]: Tomamos los audios del dataset correspondiente
                                     al ínidice. 
                                     
                                     ------------------ NOTA ------------------
                                     
                                     Como estoy tomando índices desde el inicio
                                     empezará por recorrer mi primera percusión 
                                     que son los HIHATS, implementaré un módulo
                                     que tome los n primeros sonidos de cada 
                                     percusión para realizar la reconstrucción.
                                     
                                     ------------------------------------------
                                     
                      1. .unsqueeze(0): Añadimos una dimensión adicional en los
                                        tensores para tratar el formato correcto
                                        dentro del encoder [Batch, 1, 64, 51].
                                        Al estar tratando ejemplos de forma 
                                        individual => Batch = 1. 
                                        
                      2. .squeeze(0): Hacemos algo muy parecido para audio_real
                                      en este caso como en 'audio_processor.py'
                                      guardamos los tensores de audio_real como 
                                      [1, 12800] => [12800]. En este caso para 
                                      evitar estar moviendo datos entre dispositivos
                                      como torch.save guarda en CPU, tratamos el
                                      audio_real en CPU. 
                                      
                      3. model(): Al tener los espectrogramas Mel de los audios 
                                  que queremos reconstruir, podemos llamar al 
                                  forward del modelo directamente, procesando 
                                  encoder -> reparametrize -> decoder con los 
                                  pesos correspondientes al mejor checkpoint del
                                  entrenamiento, con lo que acabamos con los mejores 
                                  coeficientes del filtro Mel de reconstrucción 
                                  de los sonidos de referencia. 
                                  
                      4. noise_synth(): Ahora, llamamos al sintetizador del modelo
                                        para terminar la reconstrucción a partir
                                        de los coeficientes del filtro obtenidos
                                        anteriormente. 


4. sample_perc: Esta función se encarga de generar sonidos de percusiones de todos
                los tipos unicamente a partir de la representación del embedding, 
                sin ningún sonido real como referencia: 
                    
                    
                ######################### OPERACIONES #########################
                
                1. for...(1): Recorremos la lista de percusiones.
                
                2. for...(2): recorremos la cantidad de sonidos que queremos 
                              generar por cada tipo de percusión. 
                                    
                              1. z: Como queremos generar sonidos nuevos, establecemos
                                    los valores del tensor del espacio latente 
                                    que el decoder reciba de forma aleatoria. 
                                    
                              2. label: Creamos el tensor de la etiqueta.
                              
                              3. filter_coeffs: Llamamos al decoder con el tensor 
                                                latente aleatorio (z) y la etiqueta 
                                                conducional de la percusión pertinente
                                                (label). 
                
                              4. audio_gen: Llamamos al sintetizador con los 
                                            coeficientes del filtro Mel generado. 
                
'''


def load_system(device, 
                num_percs, 
                latent_dim, 
                window_size, 
                hop_size, 
                n_mels, 
                sample_rate,
                model_state_dict,
                noise_synth_state_dict):
    
    
    model = CVAE(num_percs = num_percs, latent_dim = latent_dim).to(device)

    noise_synth = NoiseSynthesizer(window_size = window_size,
                                   hop_size = hop_size,
                                   n_mels = n_mels,
                                   sample_rate = sample_rate).to(device)

    model.load_state_dict(model_state_dict)
    noise_synth.load_state_dict(noise_synth_state_dict)

    model.eval()
    noise_synth.eval()

    return model, noise_synth


@torch.no_grad()
def reconstruct_perc(percussions, model, noise_synth, dataset, device, out, audio_length, sample_rate, num_sounds):
    
    reconstruct_dir = os.path.join(out, "reconstruct")
    os.makedirs(reconstruct_dir, exist_ok = True)
    
    i = 0
    
    percussions_count = {}

    for percussion in percussions:
        percussions_count[percussion] = 0
        
    while i < len(dataset):
        
        mel, audio_real, label = dataset[i]
        label_int = int(label.item())
        
        if percussions_count[label_int] >= num_sounds: 
            
            if all (c >= num_sounds for c in percussions_count.values()):
                break 
            
            i = i + 1
            continue 
        
        mel = mel.unsqueeze(0).to(device).float() # [1, 1, 64, 51]
        label = label.unsqueeze(0).to(device).long() # [1]
        
        # audio_real = audio_real # [1, 256000]
        audio_real = audio_real.squeeze(0) # [25600]

        filter_coeffs, _, _ = model(mel, label)
       
        audio_gen = noise_synth(mel_filter_coeffs = filter_coeffs,
                                target_audio_length = audio_length).cpu()

        label_int = int(label.item())
        label_name = percussions.get(label_int, f"class_{label_int}")
        
        perc_dir = os.path.join(reconstruct_dir, label_name)
        os.makedirs(perc_dir, exist_ok = True)
       
        real_path = os.path.join(perc_dir, f"{i+1:03d}_{label_name}_real.wav")
        gen_path = os.path.join(perc_dir, f"{i+1:03d}_{label_name}_recon.wav")

        save_audio(real_path, audio_real, sample_rate)
        save_audio(gen_path, audio_gen, sample_rate)
        
        i = i + 1 
       
        percussions_count[label_int] += 1 
        
@torch.no_grad()
def sample_perc(percussions, model, noise_synth, device, out, audio_length, sample_rate, num_percs, latent_dim, num_sounds):
    
    sample_dir = os.path.join(out, "sample") 
    os.makedirs(sample_dir, exist_ok = True)
    
    label_int = 0 
    
    for label_name in list(percussions.values()):
        
        perc_dir = os.path.join(sample_dir, label_name)
        os.makedirs(perc_dir, exist_ok = True)
        
        for i in range(num_sounds):
            
            z = torch.randn(1, latent_dim, device = device)
            
            label = torch.tensor([label_int], device = device).long()

            filter_coeffs = model.decoder(z, label)
            
            audio_gen = noise_synth(mel_filter_coeffs = filter_coeffs,
                                    target_audio_length=audio_length).cpu()

            out_path = os.path.join(perc_dir, f"{i+1:03d}_{label_name}_sampled.wav")
            save_audio(out_path, audio_gen, sample_rate)
            
        label_int = label_int + 1 


def save_audio(path, tensor, sample_rate): # SAVER
    x = tensor.detach().cpu()

    if x.dim() == 2 and x.size(0) == 1:
        x = x.squeeze(0)

    x = x.numpy().astype(np.float32)
    sf.write(path, x, sample_rate)
     
      
@torch.no_grad()
def generar_imagenes_memoria(model, dataset, device, out_dir):
    import os
    os.makedirs(out_dir, exist_ok=True)
    
    # Mapeo idéntico al de tu dataset (0: HIHAT, 1: KICK, 2: SNARE, 3: TOM, 4: PERC)
    tag_mapping = {0: "HIHAT", 1: "KICK", 2: "SNARE", 3: "TOM", 4: "PERC"}
    
    # Diccionario para saber si ya hemos encontrado y graficado una clase
    encontrados = {k: False for k in tag_mapping.keys()}
    
    print("\n-> Iniciando generación cualitativa de espectrogramas (1 por clase)...")
    
    for i in range(len(dataset)):
        # Si ya hemos encontrado uno de cada clase, paramos el bucle
        if all(encontrados.values()):
            break
            
        mel, _, label = dataset[i]
        label_int = int(label.item())
        
        # Si esta clase aún no la hemos procesado
        if not encontrados[label_int]:
            class_name = tag_mapping[label_int]
            
            # --- INTENTO DE SACAR EL NOMBRE DEL ARCHIVO ---
            try:
                if hasattr(dataset, 'files'):
                    file_path = dataset.files[i]
                elif hasattr(dataset, 'file_paths'):
                    file_path = dataset.file_paths[i]
                elif hasattr(dataset, 'data'): 
                    file_path = dataset.data[i]
                else:
                    file_path = f"Muestra_Test_n{i}"
                file_name = os.path.basename(str(file_path))
            except Exception:
                file_name = f"Muestra_n{i}"
            # ----------------------------------------------
                
            print(f"   --> Procesando {class_name} | Archivo: {file_name}")
            
            mel_input = mel.unsqueeze(0).to(device).float()
            label_tensor = label.unsqueeze(0).to(device).long()
            
            # Inferimos los coeficientes
            filter_coeffs, _, _ = model(mel_input, label_tensor)
            
            matrix_original = mel.squeeze().numpy()
            matrix_reconstruida = filter_coeffs.squeeze().cpu().numpy().T
            
            fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
            
            fig.suptitle(f"Análisis Cualitativo: {class_name} ({file_name})", fontsize=14, fontweight='bold')
            
            im0 = axs[0].imshow(matrix_original, aspect='auto', origin='lower', cmap='magma', vmin=0, vmax=1)
            axs[0].set_title("Espectrograma Original Mel")
            axs[0].set_xlabel("Tiempo (Frames)")
            axs[0].set_ylabel("Frecuencia (Bandas Mel)")
            fig.colorbar(im0, ax=axs[0], label='Amplitud (dB norm)')
            
            im1 = axs[1].imshow(matrix_reconstruida, aspect='auto', origin='lower', cmap='magma', vmin=0, vmax=1)
            axs[1].set_title("Espectrograma Reconstruido (NDDSP)")
            axs[1].set_xlabel("Tiempo (Frames)")
            fig.colorbar(im1, ax=axs[1], label='Amplitud (dB norm)')
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            safe_name = file_name.replace('.pt', '').replace('.wav', '')
            filename_out = f"espectrogramas_{class_name}_{safe_name}.png"
            
            plt.savefig(os.path.join(out_dir, filename_out), dpi=300)
            plt.close(fig)
            
            # Marcamos esta clase como completada
            encontrados[label_int] = True
            
    print(f"-> ¡Gráficas guardadas con éxito en: {out_dir}\n")
    
def main():
    
    # Cargamos el Checkpoint.
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = "./checkpoints/best_nddsp.pt"
    
    checkpoint = torch.load(checkpoint_path, map_location = device)
    
    config = checkpoint["config"]
    
    model_state_dict = checkpoint["model_state_dict"]
    noise_synth_state_dict = checkpoint["noise_synth_state_dict"]
    
    percussions = {0: "HIHAT", 1: "KICK", 2: "SNARE", 3: "TOM", 4: "PERC"}
    out = "./CVAE_outputs"
    data = config["data_dir"]
    audio_length = config["audio_length"]  # audio_length = 25600
    num_percs = config["num_percs"] # num_percs = 5
    latent_dim = config["latent_dim"]  # latent_dim = 16
    window_size = config["window_size"]
    n_mels = config["n_mels"]
    hop_size = config["hop_size"]
    sample_rate = config["sample_rate"] # sample_rate = 32000
   

    # Preparamos el CVAE con los parámetros del checkpoint.
    
    model, noise_synth = load_system(device = device,
                                     num_percs = num_percs,
                                     latent_dim = latent_dim,
                                     window_size = window_size, 
                                     hop_size = hop_size, 
                                     n_mels = n_mels, 
                                     sample_rate = sample_rate,
                                     model_state_dict = model_state_dict,
                                     noise_synth_state_dict = noise_synth_state_dict)
                                     
                                     
    print(f"Mejor checkpoint cargado: epoch {checkpoint['epoch']}")

    dataset = PercussionDataset(data, augment = False)

    reconstruct_perc(percussions = percussions, 
                     model = model, 
                     noise_synth = noise_synth,
                     dataset = dataset, 
                     device = device, 
                     out = out, 
                     audio_length = audio_length, 
                     sample_rate = sample_rate,
                     num_sounds = 12) # Variamos según la cantidad de sonidos que queramos generar. 
    
    
    sample_perc(percussions = percussions, 
                model = model, 
                noise_synth = noise_synth, 
                device = device, 
                out = out, 
                audio_length = audio_length, 
                sample_rate = sample_rate,
                num_percs = num_percs,
                latent_dim = latent_dim,
                num_sounds = 15) # Variamos según la cantidad de sonidos que queramos generar.
    
    # --- LLAMADA A LA NUEVA FUNCIÓN ---
    generar_imagenes_memoria(model=model, dataset=dataset, device=device, out_dir=out)
    
if __name__ == "__main__":

    main()