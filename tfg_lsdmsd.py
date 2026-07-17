"""
Created on Tue Apr 28 10:05:49 2026

@author: andre
"""

import os
import torch
import torchaudio
import soundfile as sf 
import numpy as np 
import pandas as pd

def calcular_lsd(audio_real, audio_gen, n_fft = 2048, hop_length = 512):
    
    '''
    
    Calcula la Log-Spectral Distance con límite de rango dinámico para evitar 
    penalizar el silencio.
    
    '''
    
    transform = torchaudio.transforms.Spectrogram(n_fft = n_fft, 
                                                  hop_length = hop_length, 
                                                  power = 2.0)
    
    spec_real = transform(audio_real)
    spec_gen = transform(audio_gen)
    
    
    db_transform = torchaudio.transforms.AmplitudeToDB(stype = 'power', top_db = 80.0)
    log_spec_real = db_transform(spec_real)
    log_spec_gen = db_transform(spec_gen)
    
    
    '''
    
    Calculamos la LSD
    
    '''
    
    diff_sq = (log_spec_real - log_spec_gen) ** 2
    lsd_final = torch.mean(torch.sqrt(torch.mean(diff_sq, dim = 0)))
    
    return lsd_final.item()

def calcular_msd(audio_real, 
                 audio_gen, 
                 sample_rate = 32000, 
                 n_mels = 64,
                 n_fft = 2048,
                 hop_length = 512):
    
    """Calcula la Mel-Spectral Distance con límite de rango dinámico."""
    
    mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate = sample_rate, 
                                                         n_fft = n_fft, 
                                                         hop_length = hop_length,
                                                         n_mels=n_mels)
    
    mel_real = mel_transform(audio_real)
    mel_gen = mel_transform(audio_gen)
    
    db_transform = torchaudio.transforms.AmplitudeToDB(stype = 'power', top_db = 80.0)
    log_mel_real = db_transform(mel_real)
    log_mel_gen = db_transform(mel_gen)
    
    diff_sq = (log_mel_real - log_mel_gen) ** 2
    msd_final = torch.mean(torch.sqrt(torch.mean(diff_sq, dim = 0)))
    
    return msd_final.item()


def main():
    
    reconstruct_dir = "./CVAE_outputs/reconstruct"

    resultados = []
    
    
    for clase_perc in os.listdir(reconstruct_dir):
        clase_dir = os.path.join(reconstruct_dir, clase_perc)
        
        if not os.path.isdir(clase_dir):
            continue
            
        
        lsd_lista = []
        msd_lista = []
        
        archivos = os.listdir(clase_dir)
        archivos_reales = [f for f in archivos if f.endswith("_real.wav")]
        
        for archivo_real in archivos_reales:
           
            archivo_recon = archivo_real.replace("_real.wav", "_recon.wav")
            
            ruta_real = os.path.join(clase_dir, archivo_real)
            ruta_recon = os.path.join(clase_dir, archivo_recon)
            
            if os.path.exists(ruta_recon):
                
                audio_real_np, sr = sf.read(ruta_real)
                audio_recon_np, _ = sf.read(ruta_recon)
                
                audio_real = torch.from_numpy(audio_real_np).float()
                audio_recon = torch.from_numpy(audio_recon_np).float()
                
                audio_real = audio_real.squeeze()
                audio_recon = audio_recon.squeeze()
                
                lsd = calcular_lsd(audio_real, audio_recon)
                msd = calcular_msd(audio_real, audio_recon, sample_rate=sr)
                
                lsd_lista.append(lsd)
                msd_lista.append(msd)
        
       
        if lsd_lista:  # Calculamos la media de esta clase
            
            lsd_media = sum(lsd_lista) / len(lsd_lista)
            msd_media = sum(msd_lista) / len(msd_lista)
            
          
            resultados.append({
                "Percusion": clase_perc,
                "LSD_dB": round(lsd_media, 2),
                "MSD_dB": round(msd_media, 2)
            })
            
    if resultados:
        
        df_resultados = pd.DataFrame(resultados)
        
        media_total_lsd = df_resultados["LSD_dB"].mean()
        media_total_msd = df_resultados["MSD_dB"].mean()
        
        df_resultados.loc[len(df_resultados)] = ["MEDIA_TOTAL", round(media_total_lsd, 2), round(media_total_msd, 2)]
        
        ruta_csv = "./CVAE_CSV/acoustic_metrics.csv"
        
        df_resultados.to_csv(ruta_csv, index=False)
        
if __name__ == "__main__":
    main()