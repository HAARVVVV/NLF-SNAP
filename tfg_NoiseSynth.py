"""
Created on Sun Mar 1 11:15:16 2026

@author: andre
"""

import torch
import torch.nn as nn
import torchaudio
import torchaudio.functional as F_audio
import sys


'''

NoiseSynth presenta una arquitectura simple propia que se encarga de:
    
    1. Tomar los coeficientes mel_filter_coeffs generados por el decoder. 

    2. Estos los proyectamos sobre un dominio lineal de STFT mediante una 
       pseudoinversa de Moore-Penrose de la matriz de filtros mel.
       
    3. Generar ruido blanco.
    
    4. Calcular su STFT compleja.
    
    5. Multiplicat ese espectro de ruido por el filtro real aprendido. 
    
    6. Mediante una ISTFT (Inversa STFT) volvemos al dominio temporal.


NoiseSynth es un 'SINTETIZADOR SUSTRACTIVO DIFERENCIABLE', la red no predice la
onda directamente como en 'tfg_decoder.py', sino que crea una envolvente espectral 
(un FILTRO de señal) dependiente del tiempo que modula un ruido de excitación, en 
nuestro caso, un ruido blanco (ALEATORIO) generado mediante una distribución
Normal Univariable ~N(0,1). 


################################## PARÁMETROS #################################

1. n_stft: representa el número de bandas lineales final de la representación 
           espectral de la STFT del sonido original, la fórmula viene determinada 
           por el teorema de Shannon-Nyquist: 
           
               n_stft = (window_size // 2) + 1 
           
           Al calcular la Transformada Rápida de Fourier (FFT) de una ventana 
           de 1024 muestras temporales, matemáticamente obtenemos 1024 números 
           complejos. Sin embargo, para una señal real (como el audio), el 
           espectro es simétrico. La segunda mitad es un espejo redundante 
           de la primera. (513 locaciones frecuenciales) 
           
           ------------------------------- NOTA -------------------------------
           
           Hay cosas que se que repito y reexplico, hay que tener en cuenta que
           he llevado a cabo esto a lo largo de varios meses, con lo que no viene 
           mal recordar ciertos conceptos importantes.

           --------------------------------------------------------------------
           
          
################################# OPERACIONES #################################

1. F_audio.melscale_fbanks(): Es el banco de filtros Mel. La matriz de filtros 
                              triangulares que usábamos en 'audio_processor.py'
                              para transformar magnitudes lineales de frecuencia 
                              a una representación en bandas Mel.
 
2. torch.pinverse(mel_fb): Calcula la Pseudo-inversión de Moore-Penrose. 
                           
                           Recordemos que para calcular el espectrograma Mel desde 
                           el espectrograma lineal, en 'audio_processor.py'
                           realizábamos la siguiente operación: 
                               
                               Y_{mel} = M * X_{fft}
                               
                           Si M fuera una matriz cuadrada n x n, simplemente calcularíamos 
                           su inversa matemática M^{-1} de tal manera que podemos 
                           volver al espectro lineal:
                               
                               X_{fft} = M^{-1} * Y_{mel}
                               
                           Sin embargo, los bancos de filtros que hemos creado 
                           en el decoder constituyen matrices RECTANGULARES 
                           (51 x 64), por tanto, trataremos de encontrar una 
                           proyección aproximada a una inversa para poder regresar
                           al espectrograma lineal, es decir, queremos buscar 
                           una matriz M^+ tal que: 
                               
                               \hat{X}_{fft} = M^+ * Y_{mel}
                           
                           Donde \hat{X}_{fft} es la solución mínima para el MSE
                           entre el espectrograma Mel de entrada y el producto 
                           entre el espectrograma estimado y los filtros:
                               
                               \arg\min_{\hat{X}_{fft}} = ||M \cdot \hat{X}_{fft} - Y_{mel}||^2
                         
                           Todo esto Pytorch lo calcula mediante 'DESCOMPOSICIÓN
                           DE VALORES SINGULARES' (SVD). 
                           
                           ----------------------- NOTA -----------------------
                           
                           No he investigado en que consiste la SVD, trataré de 
                           hacerlo una vez construido el scaffolding completo.
                           
                           ----------------------------------------------------
                           
                           Para ello, Usamos una aproximación analítica instantánea 
                           llamada Pseudo-inversa de Moore-Penrose. La pseudoinversa
                           de Moore-Penrose de una matriz M de dimensión m x n 
                           tal que rg(M) = n (n > m) viene dada por la fórmula: 
       
                               M^+ = (M^t*M)^{-1}*M^t
                        
~~~~~~~~~~~~~~~~~~~~~~~~~~ Real Soun Synthesis [3] ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
~~~~~~~~~~~~~~~~~~~ Discrete Tiem Signal Processing [10.3] ~~~~~~~~~~~~~~~~~~~~
   
'''

class NoiseSynthesizer(nn.Module):
   
    
    def __init__( self, 
                 window_size = 2048, 
                 hop_size = 512, 
                 n_mels = 64, 
                 sample_rate = 32000):
        
        super().__init__()
        
        self.window_size = window_size
        self.hop_size = hop_size
        
        self.n_stft = (window_size // 2) + 1  
        
        # MATRIZ DE TRANSFORMACIÓN FÍSICA
        # Proyecta las 64 bandas Mel logarítmicas de vuelta al mundo Lineal de la STFT
                
        mel_fb = F_audio.melscale_fbanks(
            
            n_freqs = self.n_stft,
            f_min = 0.0,
            f_max = 16000,
            n_mels = n_mels,
            sample_rate = sample_rate,
            
        ) 
        
        inverse_mel = torch.pinverse(mel_fb)
        
        # La guardamos como un parámetro no entrenable (buffer) en la VRAM
        
        self.register_buffer('inverse_mel', inverse_mel)
        
        
    def forward(self, mel_filter_coeffs, target_audio_length):
        
        '''
        
        Esta es la función a la que llamamos para procesar los datos con 
        los métodos diseñados en __init__:
            
            
        ############################# PARÁMETROS ##############################
        
        1. mel_filter_coeffs: Curvas de volumen generadas por el Decoder.
                              ([Batch, 51, 64])
     
        2. target_audio_length: Longitud de muestras del audio original para
                                la reconstrucción. (25600)
                                
        
        ############################ OPERACIONES ##############################
            
        1. batch_size: Establecemos el tamaño de Batch. (Batch)
        
        2. device: Indicamos el dispositivo del ordenador donde vamos a realizar
                   las operaciones y a guardar información importante para cálculos
                   sucesivos, de esta forma nos podremos referir a device siempre
                   que queramos usar el mismo entorno (dispositivo) por temas 
                   de corrección computacional. PyTorch espera que los tensores
                   que interactúan entre sí estén en el mismo dispositivo.
                   
        3. linear_filter = torch.matmul(): Expandimos estadísticamente las bandas
                                           Mel obtenidas del decoder sobre el 
                                           espacio lineal temporal de la STFT.
                                           
                                           ([Batch, 51, 64] x [64, 1025] = 
                                           [Batch, 51, 1025])
                                           
        4. torch.relu(): Aplicamos ReLU para asegurarnos de que no existen 
                         valores negativos, así habilitamos la corrección del 
                         filtro. ([Batch, 51, 1025])
                         
        5. linear_filter.transpose(1, 2): Trasponemos el tensor para adaptarlo 
                                          a la forma de la STFT [Batch, FRQ, T].
                                          ([Batch, 1025, 51])
                                          
        6. noise: Generamos el espectro de ruido blanco entre como una Normal
                  univariable ~N(0,1), es sobre este que aplicaremos la 
                  síntesis sustractiva con la información extraida del decoder.
                    
        7. noise_stft: Calculamos la STFT del ruido blanco generado para poder 
                       operarlo en el mismo dominio de la frecuencia en la que 
                       hemos calculado el filtro.
        
        8. noise_stft.size(2) == linear_filter.size(2): Por seguridad frente a 
                                                        redondeos de PyTorch,
                                                        comprobamos que el número 
                                                        de frames es igual en la 
                                                        STFT del ruido y en el 
                                                        filtro. 
                       
        9. filtered_stft: Es una multiplicación física, el ruido caótico adopta 
                          la forma dictada por los coeficientes de filtro construidos 
                          por el decoder. Al ser noise_stft complejo y linear_filter
                          real, LA FASE SE MANTIENE INTACTA.
                          
        10. audio_generado: Invertimos la STFT para obtener la onda de audio 
                            final.
        
        11. InverseMelScale: Si usamos el método InverseMelScale dentro del método 
                             forward de la red destrozará la velocidad al tener 
                             que calcularla en cada iteración del modelo, es por 
                             ello que la precomputamos la pseudo-inversa de  
                             Moore-Penrose de los espectrogramas Mel capturados 
                             por el decoder.
                             
                             
        '''
        
        batch_size = mel_filter_coeffs.size(0) 
        device = mel_filter_coeffs.device 
        
        
        linear_filter = torch.matmul(mel_filter_coeffs, self.inverse_mel)
        
        # Aseguramos que no haya valores negativos por el cálculo de mínimos cuadrados
        
        linear_filter = torch.relu(linear_filter)
       
        linear_filter = linear_filter.transpose(1, 2) 
       
        noise = 2 * torch.rand(batch_size, target_audio_length, device=device) - 1
        
        # noise = torch.randn(batch_size, target_audio_length, device=device)  
      
        # PROBAR ALTERNATIVA: 
            
        # noise = torch.rand(batch_size, target_audio_length, device=device) * 2 - 1 
       
        noise_stft = torch.stft(
        
            noise, 
            n_fft = self.window_size, 
            hop_length = self.hop_size, 
            win_length = self.window_size,
            window = torch.hann_window(self.window_size).to(device), # Evitar fallos
            return_complex = True,
            center = True,
            pad_mode = 'constant'
        )
        
       
        if noise_stft.size(2) == linear_filter.size(2):
           
           '''
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ DDSP ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
       
           La síntesis sustractiva mediante multiplicación real-compleja garantiza
           la diferenciabilidad del modelo. 
           
           '''
            
           filtered_stft = noise_stft * linear_filter
           
           # filtered_stft = noise_stft[:, :, :min_frames] * linear_filter[:, :, :min_frames]
            
           audio_generado = torch.istft(
        
               filtered_stft, 
               n_fft = self.window_size, 
               hop_length = self.hop_size, 
               win_length = self.window_size,
               window = torch.hann_window(self.window_size).to(device),
               center = True,
               length = target_audio_length
        )
            
           return audio_generado
        
        
        else:
            
            raise ValueError(f"Frames incompatibles: noise_stft={noise_stft.size(2)}")
            
            sys.exit()
        

    