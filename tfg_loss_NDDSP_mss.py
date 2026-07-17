"""
Created on Wed Feb 18 16:36:53 2026

@author: andre
"""

import torch
import torch.nn.functional as Func

'''
 
Calculamos la función de pérdida total combinando MSS Loss (Multi-Scale Spectral 
Loss) y la Divergencia KL.

------------------------------------- NOTA ------------------------------------

Estos aspectos teóricos se desarrollarán dentro de la memoria, no quiero extenderme
en las anotaciones del programa sobre la naturaleza del CVAE y en qué consiste 
KLD. 

-------------------------------------------------------------------------------


################################## PARÁMETROS #################################
 
1. audio_real: Tensor de audio original ([Batch, Samples])
 
2. audio_generado: Tensor de audio creado por el Sintetizador ([Batch, Samples])
 
3. mu, logvar: Parámetros del espacio latente obtenidos del encoder.
 
4. beta: Peso de la regularización KL 
   

   ----------------------------------- NOTA -----------------------------------
   
   El valor beta será de los principales factores que determinarán como de bueno 
   es el modelo para el entrenamiento, iremos probando numerosos cambios y formatos
   para esta. 
   
   ----------------------------------------------------------------------------
   
5. fft_sizes: Tomamos varios tipos de ventana para calcular la pérdida sobre 
              partes de tamaño variable del sonido, múltiples resoluciones STFT 
              permiten capturar rasgos con diferente extensión temporal.         
                 
6. mss_loss_total: Inicializamos la MSS Loss. 
 
7. Epsilon: Inicializamos el parámetro auxiliar, no lo hacemos 0 para evitar 
            cálculos de log(0).
             
             
################################# OPERACIONES #################################
 
1. hop_length: Tomamos el 75% de solapamiento como hicimos en 'audio_processor.py'.
 
2. mag_real(mag_reg): Calculamos las STFT de ambos audios, el original y el 
                      generado, para calcular MSS requeriremos unicamente de 
                      la magnitud, con lo que tomamos valores absolutos de 
                      las STFT e ignoramos la fase. ([Batch, Frecuencias, Tiempo])
 
3. loss_lin: Calculamos el error de Magnitud Lineal, enfocado en los picos 
             del volumen.
              
4. loss_log: Calculamos el error de Magnitud Logarítmica, enfocado en las colas
             de sonido y silencios. 
             
             
5. mss_loss_total: Añadimos ambas pérdidas a la mss_loss_total. Una vez recorridos 
                   todos los tipos de ventanas lo dividimos entre el número de 
                   estas para que no dependa de cuantas ventanas hemos seleccionado. 
              
6. kld_loss: Calculamos la 'DIVERGENCIA DE KULLBACK-LEIBER', es una medida 
             que cuantifica la diferencia entre dos distribuciones probabilísticas.
             Penaliza que la distribución latente aproximada obtenida del 
             encoder se aleje demasiado de la distribución a priori gaussiana
             estándar. Su minimización empuja al encoder a producir
             distribuciones latentes más compactas y regulares.
              
             Sumamos por cada muestra los valores de las dimensiones y promediamos
             con respecto el Batch. 
'''


def nddsp_loss_mss(audio_real, audio_generado, mu, logvar, beta):
  
    fft_sizes = [64, 128, 256, 512, 1024]
    # fft_sizes = [128, 256, 512, 1024]
    mss_loss_total = 0.0
    epsilon = 1e-5 
    
    
    for n_fft in fft_sizes:
        
        hop_length = n_fft // 4
        
        device = audio_real.device
        stft_aureal = torch.stft(
                    
                      audio_real, 
                      n_fft = n_fft,
                      hop_length = hop_length, 
                      win_length = n_fft,
                      window = torch.hann_window(n_fft).to(device),
                      return_complex = True, 
                      center = True, 
                      pad_mode='constant'
                    
                      )
        
        stft_augen = torch.stft(
            
                      audio_generado, 
                      n_fft = n_fft,
                      hop_length = hop_length, 
                      win_length = n_fft,
                      window = torch.hann_window(n_fft).to(device),
                      return_complex = True, 
                      center = True, 
                      pad_mode='constant'
                    
                      )
        
        mag_real = torch.abs(stft_aureal)
        mag_gen = torch.abs(stft_augen)   
        
        loss_lin = Func.l1_loss(mag_real, mag_gen, reduction='mean')
       
        loss_log = Func.l1_loss(
            
                                torch.log(mag_real + epsilon), 
                                torch.log(mag_gen + epsilon), 
                                reduction='mean'
                             
                                )
        
        mss_loss_total = mss_loss_total + (loss_lin + loss_log)
        
        
    mss_loss_total = mss_loss_total/len(fft_sizes)
    
    
# %% CÁLCULO ESTÁNDAR.

    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim = 1)
    kld_loss = kld_loss.mean()
    
    loss_total = mss_loss_total + (beta * kld_loss)
    
    return loss_total, mss_loss_total, kld_loss

# %% FREE BITS.
    

    # '''
    
    # Con Free-Bits permitimos una cantidad mínima de información por dimensión 
    # antes de volver a penalizarla con normalidad el KLD.
    
    
    # ############################### PARÁMETROS ################################
    
    # 1. free_bits: Cantidad mínima de información libre que dejamos por cada 
    #               dimensión del espacio latente en la KLD. Usamos esto para 
    #               evitar el colapso posterior. 
                  
                  
    # ############################### OPERACIONES ###############################
    
    # 1. kld_per_dim: Cálculo igual de la KLD. 
    
    # 2. kld_mean: Hacemos la media por batch de los valores de KLD en cada dimensión.
    #              Entonces sabremos por cada epoch la cantidad de información 
    #              retenida en el espacio latente, en cada dimensión del mismo. 
    
    # 3. threshold: Creamos un tensor del mismo tamaño que kld_mean, pero 
    #               relleno con free_bits. Free-bits quiere controlar cuánto usa
    #               cada coordenada del latente.
                     
    # 4. kld_loss: Calculamos la KLD final como la suma de los máximos teniendo 
    #              en cuenta los freebits, bloquea al optimizador de seguir 
    #              reduciendo la relevancia del espacio latente. 
                 
    # '''
    
    free_bits = 1.0
    
    kld_per_dim = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar) # [Batch, latent_dim]
    kld_mean = kld_per_dim.mean(dim=0)  # [latent_dim]

    threshold = torch.full_like(kld_mean, free_bits)
    
    kld_loss = torch.maximum(kld_mean, threshold).sum()

    loss_total = mss_loss_total + beta * kld_loss

    return loss_total, mss_loss_total, kld_loss