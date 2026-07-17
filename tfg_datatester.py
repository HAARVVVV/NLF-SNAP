"""
Created on Tue Dec 16 19:33:20 2025

@author: andre
"""


import torch
import matplotlib.pyplot as plt

def datatester(archivo : str): 
    
    data = torch.load(f"{archivo}", weights_only=False)
    
    print("KEYS:", data.keys())
    print(f"Dim AUDIO: {data['audio'].shape}")
    print(f"Dim MEL: {data['mel'].shape}")
    print(f"Etiqueta numérica de clase: {data['label']}")
    print(data['mel'][0, :5, :5])
    print(f"Valor máximo en el tensor: {data['mel'].max()}") # <= 1.0
    print(f"Valor mínimo en el tensor: {data['mel'].min()}") # >= 0.0

    mel_matrix = data['mel'].squeeze().numpy()

    plt.figure(figsize=(10, 4))
    plt.imshow(mel_matrix, aspect='auto', origin='lower', cmap='magma')     
    
    plt.colorbar(label='Amplitud (dB, Normalizado)')
    plt.title("Espectrograma Mel del Audio, 07_Snare_19_SP.pt (Etiqueta: 'Snare')")
    plt.xlabel("Tiempo (Frames)")
    plt.ylabel("Frecuencia (Bandas Mel)")
    
    plt.tight_layout()
    plt.show()
     

datatester("./processed_dataset/SNARES/07_Snare_19_SP.pt") # PONER UN ARCHIVO CUALQUIERA