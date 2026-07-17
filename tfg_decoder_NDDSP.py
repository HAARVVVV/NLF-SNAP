# -*- coding: utf-8 -*-
"""
Created on Mon Feb 9 16:06:58 2026

@author: andre
"""

import torch
import torch.nn as nn



class NDDSP_Decoder(nn.Module):
    
    def __init__(self, num_percs = 5, latent_dim =16, num_frames = 51, 
                 filter_bands = 64): # Lo fijamos aquí para asegurarnos
        
        '''
    
        En "tfg_decorder.py" aplicamos el enfoque estándar para la reconstrucción
        estipulada por un VAE, usamos 'CONVOLUCIONES TRANSPUESTAS' para 'INVERTIR'
        el proceso del encoder, este proceso quizás no nos proporciona la calidad
        de resultado deseada, ya que estamos reconstruyendo y pegando frames, lo 
        que puede suponer problemas de continuidad entre frames y de fases 
        desalineadas. 
        
        Bajo el paradigma DDSP, el decoder no generará directamente el sonido, 
        sino que generaremos 'INSTRUCCIONES', las cuales serán comandadas a un 
        sintentizador harmónico + ruido. 
    
        En este proyecto, al esta generando percusiones, hemos tomado la decisión 
        de diseño de obviar la componente armónica completamente, solo haremos 
        uso del 'SINTETIZADOR DE RUIDO'. 
    
        Este 'decoder_NDDSP.py' (Noise-DDSP) devolverá los coeficientes espectrales 
        para construir un 'FILTRO LTI', el cual que se solapará sobre la señal  
        generada por el sintetizador de ruido 'tfg_NoiseSynth.py'.
    
        Recordemos que en el encoder acabamos con el cálculo de la madia y varianza
        de una Gaussiana Multivariante del espacio latente desconocida, de la cual,
        tras aplicar la reparametrización:
               
            z = \mu + \sigma \cdot \epsilon, 
           
        Obtenemos los 16 valores asignados a la distribución aún desconocida.    
        Ahora, podemos juntar esos 16 valores con los 8 pesos de la etiqueta 
        determinada recogidos del Embedding, obteniendo un tensor de 24 valores. 
        (H')
    
            H' = [z; e(c)] \in \\mathbb{R}^24
            

        ############################# PARÁMETROS ##############################
           
        1. latent_dim: Dimensión del espacio latente, usamos la misma que la 
                       fijada en el encoder. (16)
                         
        2. num_frames: Queremos construir un audio de las mismas características
                       que un audio preprocesado de dimensión [Batch, 1, 64, 51]. 
                       (51)
    
        3. filter_bands: de la misma forma que 'num_frames'. (64)                                  
           
    
        ############################# OPERACIONES #############################
           
        1. nn.Embedding(): Generamos un embedding distinto al del encoder, (la 
                           matriz de pesos de las etiquetas), para poder 
                           llamarlos con el índice correspondiente. 
                              
        2. self.expand(): Vamos a construir una secuencia de métodos para expandir
                          la información del espacio latente del en dimensiones
                          espaciales mayores: 
                             
                          1. nn.Linear(_,_): Implementa una transformación afín 
                                             que toma un vector de entrada de 
                                             dimensión _1_ y lo proyecta sobre 
                                             uno de dimensión _2_.
                                                 
                                                 x' = x*W^t + b 
                                 
                          2. nn.LayerNorm(): Mecaninsmo de normalización similar
                                             a 'nn.BatchNorm2d()' pero que opera
                                             sobre las dimensiones espaciales por 
                                             cada capa de forma independiente. 
                                             Tiene el objetivo de estabilizar la 
                                             distribución interna de las capas 
                                             intermedias y facilitar el entrenamiento 
                                             del decoder.
           
        3. nn.GRU(): La GRU (Gated Recurrent Unit) es una capa con 'MEMORIA'. 
                     Cuando calculamos el coeficiente del filtro de un frame, 
                     "recuerda" la forma del coeficiente de los frames vecinos, 
                     dotando de coherencia temporal al filtro construido. 
                     
                     
                         ##################### PARÁMETROS #####################
                        
                         1. input_size: Indica la dimensión del vector de entrada
                                        en cada frame temporal.
                            
                         2. hidden_size: El tamaño de la memoria que la red 
                                         mantiene y actualiza a lo largo de la 
                                         secuencia.
                                         
                                         Por tanto la salida también tendrá 128
                                         valores por frame.
                                         
                         3. batch_first: (= True) le marcamos que batch va 
                                         primero en la ordenación del tensor. 
                        
                            
                     La GRU procesa cada paso de tiempo calculando una serie de 
                     multiplicaciones de matrices y funciones de activación sigmoide
                     y  tangente hiperbólica (\tanh). Utiliza dos gates para
                     controlar el flujo de la información:
           
                         1. Update Gate, z_t: Decide cuánta memoria del pasado se 
                                              debe conservar para este milisegundo.
                                
                                              z_t = \sigma(W_z x_t + U_z h_{t-1})
                                
                        2. Reset Gate, r_t: Decide cuánto del pasado inmediato se 
                                            debe "olvidar" porque el sonido ha
                                            cambiado bruscamente.
                               
                                            r_t = \sigma(W_r x_t + U_r h_{t-1})
                                
                     Gracias a esta matemática, la salida de la GRU es [Batch, 51, 128], 
                     51 pasos temporales de 128 valores perfectamente cohesionados 
                     y acústicamente lógicos.
    
                         M = (M0, ..., M50), M(t) \in \mathbb{R}^128
                        
        4. nn.Sigmoid(): La GRU nos da 128 variables abstractas por cada frame 
                         temporal. Pero nuestro ecualizador final (el sintetizador 
                         de ruido que haremos en 'tfg_NoiseSynth.py') no entiende 
                         de variables sobre escalas diferentes; necesita bandas 
                         reales. Es por ello que, a continuación, proyectamos 
                         cada frame temporal, a bandas espectrales mediante una
                         capa lineal.
                         
                         Ahora, para asegurarnos de que la red escupe coeficientes 
                         de filtro más convenientes, aplicamos una sigmoide final.
                         (\in [0,1])
                         
                         ------------------------ NOTA ------------------------
                         
                         La sigmoide quizás comprime demasiado el rango dinámico;
                         si luego notamos que el sintetizador queda demasiado 
                         “plano”, podríamos explorar una parametrización positiva
                         diferente, cómo 'SOFTPLUS', 'EXP', 'RELU', 'LEAKYRELU'.
                         
                         ------------------------------------------------------
       ~~~~~~~~~~~~~~~~~~~~ Learning Phrase Representation ~~~~~~~~~~~~~~~~~~~~
       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ DDSP ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
       '''
       
       
        super().__init__()
        
        self.num_frames = num_frames
        self.filter_bands = filter_bands
        
        self.class_embedding = nn.Embedding(num_embeddings = num_percs, embedding_dim = 8)
        self.input_dim = latent_dim + 8  # 16 + 8 = 24
        
        self.expand = nn.Sequential(
            
            nn.Linear(self.input_dim, 256), 
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2),
            
            nn.Linear(256, num_frames * 128),
            nn.LeakyReLU(0.2)
        )

        
        self.gru = nn.GRU(input_size = 128, hidden_size = 128, batch_first = True)
        
        self.filter = nn.Linear(128, filter_bands)
        self.softplus = nn.Softplus()
        # self.sigmoid = nn.Sigmoid()

    def forward(self, z, c):
         
        '''
        
        Esta es la función a la que llamamos para procesar los datos con 
        los métodos diseñados en __init__:
            
            
        ############################# PARÁMETROS ##############################
        
        1. z: Son los tensores del espacio latente que obtenemos de aplicar
              la reparametrización en 'tfg_model.py' a los parámetros de la 
              distribución gaussiana multivariante obtenida de 'tfg_encoder.py'
              tendrá la forma de un tensor [Batch, 16]. 
        
q        2. c: Es el tensor de etiquetas. ([Batch, 16])
        
        
        ############################ OPERACIONES ##############################
        
        1. .view(): es un método que sirve para cambiar la forma geométrica de un 
                    tensor sin alterar sus datos. .size(0) para indicar que sigamos 
                    manteniendo el numero de batch.
        
        
        '''
        
        c_emb = self.class_embedding(c) # [batch, 8]
        merged = torch.cat([z, c_emb], dim=1) # [batch, 24]
        
        x = self.expand(merged) # [batch, 24] -> [Batch, 256] -> [Batch, 51 * 128]
        
        x = x.view(x.size(0), self.num_frames, 128) # [Batch, 51, 128]
        
        out, _ = self.gru(x) # [Batch, 51, 128]
        
        filter_out = self.filter(out) # SOLO ACTÚA SOBRE LA ÚTLIMA DIMENSIÓN
                                      # [Batch, 51, 128] -> [Batch, 51, 64]
        
        filter_out = self.softplus(filter_out) # [Batch, 51, 64]
        
        return filter_out