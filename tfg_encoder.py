"""
Created on Sat Dec 20 17:33:06 2025

@author: andre
"""

import torch
import torch.nn as nn

class Conditional_Encoder(nn.Module): 
    
    '''
    
    Creamos la función del Encoder a través de nn.module, 
    definamos a continuación como actúa.   
    
    '''
    
    '''
    https://d2l.ai/d2l-en.pdf ---> d2l
    
    [29, 119, 133, 136, 137]
    '''
    
    def __init__(self, n_mels = 64, num_percs = 5, latent_dim = 16):
        
        # self.n_mels=64
        # self.num_percs=5
        # self.latent_dim=16
        
        # Vamos a empezar con latent_dim = 16, podremos cambiar y probar más 
        # adelante
        
        # nn.Module.__init__(self)
        
        super().__init__() # Implementamos nn.Module en el propio init     
        
        '''
        
        Veamos las distintas etapas que sigue el pipeline: 
            
            
        ######################## self.conv_blocks #############################
        
        1. nn.Conv2d: Las primeras capas del encoder van a ser las denominadas
                      convoluciones en 2 dimensiones. Una convolución en 2d
                      consiste en una operación entre dos matrices, la primera 
                      (M) es aquella que queremos 'convolucionar' y la posterior 
                      (K) siendo el denominado 'kernel' o filtro que usamos para 
                      la convol. En esta operación combinamos de forma lineal 
                      los elementos de M con sus vecinos inmediatos, tomando como 
                      coeficientes los valores de K. Es decir, sea M una matriz de
                      n x m y K una de dimensión m x k, entonces:
                          
                          N[i,j] = \sum_{q=0}^{m-j}(\sum_{p=0}^{n-i} (K[i-n, j-m]* M[i,j]))
                          
                      ¿De qué nos sirve esto? En redes convolucionales, donde, 
                      evidentemente, se aplican muchas convoluciones; como en 
                      ciertos sistemas de roconocimiento y procesado de imágenes,
                      lo que conseguimos por cada capa convolución es resumir la 
                      información contenida en la M, de una 'forma dictada por K', 
                      lo que consigue que la red capture patrones espaciales 
                      pequeños al principio, y patrones más complejos al apilar 
                      varias capas. Es exactamente lo que queremos hacer con el 
                      procesamiento de nuestros esepctrogramas Mel, que, premeditadamente, 
                      también escogimos porque vienen representados como matrices
                      (M), entonces, lo que aprenderemos en el entrenamiento son  
                      los números que constituyen el filtro (K), los cuales 
                      llamamos 'PESOS', y son, literalmente, el constituyente 
                      fundamental para que una red 'APRENDA' mediante 'BACKPROPAGATION'.  
                     
                     
                      #################### PARÁMETROS #########################
        
                      1. kernel_size: Es la dimensión del filtro (K). Queremos 
                                      enfocarnos en la extracción de características 
                                      lo mas locales posible, este parámetro es 
                                      necesario para capturar transientes percusivas. 
                                      (3)
        
        
                      2. padding: Indica la dimensión del marco de 'CEROS' que 
                                  añadimos a la matriz de la convolución (M). 
                                  Queremos extraer toda la información posible 
                                  de las matrices a convolucionar, en nuestro 
                                  caso, las frecuencias GRAVES y los MOMENTOS 
                                  INICIALES son fundamentales para los sonidos 
                                  percusivos que queremos tratar. Sin padding 
                                  estaríamos omitiendo esta información inicial
                                  al no centrar el filtro de la convolución en 
                                      dichos momentos, es por ello que añadimos este 
                                  borde sin información adicional inútil para 
                                  poder capturar adecuadamente estos momentos. (1)
                    
                    
                      3. stride: Tamaño del salto del filtro sobre la matriz 
                                 entre interaciones. Un stride >1 genera un pequeño 
                                 downsampling, lo que nos permite reducir la 
                                 dimensión, omitir información redundante de 
                                 valores vecinos y optimizar la velocidad 
                                 del proceso. (2)
        
        
                      4. dim_conv = Es la fórmula de dimensión de una convolución. 
                                    Con kernel_size = 3, padding = 1, stride = 2, 
                                    podemos reducir la dimensión inicial del 
                                    espectrograma Mel de entrada de 64 x 51 a su
                                    'mitad'. (1ª capa -> 32 x 25)
        
                                    La dimensión de la matriz resultante de una 
                                    convolución viene dada por: 
                                  
                                        
        dim_conv = \floor(((dim_M + 2 * padding - (kernel-1)-1)/stride) + 1)
        
                  
                      
        ~~~~~~ A guide to convolution arithmetic for deep learning [12] ~~~~~~~
        ~~~~~~~~~~~~~~ DEEP LEARNING BOOK (BISHOP) [10, 290] ~~~~~~~~~~~~~~~~~~ 
        
        
        ------------------------------------------------------------------------
        
        2. nn.BatchNorm2d(): A medida que avanzamos dentro de la red neuronal,
                             ciertos batches de un 'CANAL' (neurona) pueden tomar 
                             valores extremos en comparación con otros, esto, 
                             a la hora de entrenar el modelo, puede provocar que 
                             las funciones de pérdida y/o activación sean más 
                             difíciles de tratar y más inestables.
                             
                             Es por esto que aplicamos la normalización por lotes,
                             que nos ayudaa mantener distribuciones de los datos
                             normalizadas, y por ende más manejables, para 
                             estabilizar el entrenamiento y optimizar la red.
                             
                             Con nn.BatchNorm2d() para cada canal de forma 
                             independiente, tomando como entradas tensores de 
                             forma [Batch, Channel, H, W], estimamos unas media 
                             (mu) y  varianza(sigma^2) a partir del batch que 
                             estemos tratando. Estos valores los utilizamos para 
                             estandarizar las activaciones bajo  una nueva 
                             distribución. La fórmula de un dato (x) normalizado
                             (y) es: 
                                 
                                 y = (x-\mu_{batch})/(\sqrt{\sigma^2}_{batch} + \epsilon)
                                 
                             Una vez acabada esta normalización en el canal por 
                             cada batch, la red puede añadir por canal dos parámetros 
                             aprendibles extras, un reescalado (\gamma) y un 
                             desplazamiento (\beta) con los que simplemente 
                             podemos modificar la normalización anterior 
                             creando una versión móvil y escalable de la misma 
                             en función de dichos parámetros, adaptándos a los
                             datos del canal, evitando así tanto romper el descenso
                             de gradiente como la propia pérdida de información.
                             El reescalado de estos nuevos parámetros viene 
                             dado por: 
                                 
                                 y = \gamma * y + \beta
                                 
                             Es decir, con la normalización pro lotes las capas 
                             continuan aprendiendo desde un estado más estable,
                             facilitando enormemente el entrenamiento.
        
        
        ~~~~~~~~~~~~~~ DEEP LEARNING BOOK (BISHOP) [7, 227] ~~~~~~~~~~~~~~~~~~~
        ~~~~~~~ Batch Normalization: Accelerating Deep Network [---] ~~~~~~~~~~
       
        
       EJEMPLO DE APLCIACION DE BATCHNORM2D:
           
       1. https://dcain.etsin.upm.es/~carlos/bookAA/05.7_RRNN_Convoluciones_CIFAR_10_INFORMATIVO.html
       2. https://keepcoding.io/blog/batch-normalization-red-convolucional/
       
       ------------------------------------------------------------------------
        
        3. LeakyReLU(): Es la 'FUNCIÓN DE ACTIVACIÓN' que usaremos. es decir,
                        aquella función que determina el paso de información 
                        entre neuronas. Estas se encargan de transformar la salida 
                        lineal de una capa para introducir 'NO LINEALIDAD'. Es esta 
                        no linealidad la que hace de esta composición masiva de 
                        funciones que estamos explicando parte de una 'RED PROFUNDA'. 
                        Si fuera puramente lineal no podría 'aprender' los poatrones
                        complejos que estamos buscando aprender.
                        
                        LeakyReLU() es una función de activación derivada de 
                        ReLU(). ReLU() tiene la siguiente forma:
                            
                            ReLU(x) = \max(0,x)
                        
                        RelU es muy simple y mantiene gradientes fuertes en 
                        la región positiva durante el entrenamiento. Sin embargo 
                        contribuye altamente al problema de las 'NEURONAS MUERTAS',
                        lo que puede provocar un estancamiento en el descenso de 
                        gradiente, sobretodo después de aplciar nn.BatchNorm2d(),
                        al haber creado distribuciones más compactas en los datos 
                        de  cada batch. 
                        
                        Es por ello que seleccionamos LeakyReLU() como mejor
                        alternativa, tiene la siguiente fórmula:
                            
                            LeakyReLU(x) = \max(\alpha x, x)
                            
                        La cual en lugar de colapsar todos los valores negativos
                        sobre 0, los escala co una constante alpha. Así,
                        permitimos que los valores negativos sigan existiendo 
                        pero más acotados, evitamos las neuronas muertas. 
                        Mantenemos la no-linealida y permitimos que fluya el 
                        gradiente. (alpha = 0.2) (ReLU era TERRIBLE)
        
        
       ------------------------------------------------------------------------
     
        3. AdaptiveAvgPool2d(): Reduce las 'DIMENSIONES ESPACIALES' de los tensores 
                                de cada canal a la dimensión objetivo, conservando
                                los canales de entrada, creando una representación
                                mas compacta mediante medias. 
                                
                                Al promediar los valores de las matrices de cada 
                                tensor, la red 'pierde sensibilidad' a la posición
                                exacta de eventualidades, en este caso, por
                                ejemplo, de una 'TRANSIENTE', le prestará entonces
                                más atención a las características globales de 
                                los datos y como interactúan entre si. Desacopla 
                                la parte final del encoder de pequeñas variaciones 
                                en los frames temporales del espectrograma.
                                
                                Además, nos ayuda enormemente en el entrenamiento
                                unificando el tamaño de salida antes de la parte 
                                final del encoder al reducir considerablemente 
                                el número de parámetros de las capas posteriores, 
                                lo que se traduce en menos memoria, menos riesgo 
                                de sobreajuste y una 'fully connected layer' 
                                mucho más ligera.
                                
                                
        ~~~~~~~ Deep Learning and the Information Bottleneck Principle ~~~~~~~~
        ~~~~~~~~~~ DEEP LEARNING BOOK (BISHOP) [receptive field] ~~~~~~~~~~~~~~
        
        
        IMPORTANTISIMO: 
            
            
        "Unsupervised Representation Learning with Deep Convolutional Generative
         Adversarial Networks"
         
        '''
        
        self.conv_blocks = nn.Sequential( # Definimos los bloques convolucionales
                                          # dentro de un pipeline para evitar 
                                          # problemas con pytorch
            
            # Extraemos patrones básicos
            
            nn.Conv2d(in_channels=1, out_channels = 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            
            # Reducimos más la imagen, extraemos características cada vez más 
            # profundas
            
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            # Captamos las características globales 
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            nn.AdaptiveAvgPool2d((2, 2)) # Reducción final de dimensión
                                         # 8 x 7 x 64 -> 3584 valores a 
                                         # 2*2*64 -> 256 valores
                                             
        )
        
        self.flatten_dim = 64 * 2 * 2 # Guardamos la dimensión final de valores
        
        
        '''
        
        ############################## EMBEDDING ##############################
        
        ¿Qué debemos hacer ahora? hemos enseñado a la red los diversos sonidos 
        y sus características a través de los datos. Nuestro objetivo es permitir
        a la red generar sonidos dentro de las etiquetas de los tipos de percusiones 
        que previamente hemos definido, ['HIHAT', 'KICK', 'SNARES', 'TOMS', 'PERCS'].
        
        Ahora mismo podemos generar sonidos pero, para construir un CVAE, debemos 
        condicionar esta generación bajo una etiqueta específica. Es aquí cuando
        definimos el 'EMBEDDING', o lo que es lo mismo, los vectores latentes 
        de clasificación. 
        
        nn.Embedding() es un método que, sobre la 'MATRIZ DEL EMBEDDING', 
        compuesta de hiperparámetros de peso (en nuestro caso 5 x 8), simplemente 
        toma la fila de vectores correspondiente a la etiqueta del sonido que 
        estamos tratando en ese momento. Por ejemplo, si estamos con un 'HIHAT',
        nn.Embedding sabrá directamente que los pesos correspondientes a dicha 
        etiqueta serán los de la primera fila de la matriz del embedding. 
        
        Cada etiqueta queda representada no por un único número, sino por un pequeño
        vector continuo entrenable. De hecho, es una representación mucho más 
        útil y 'lógica' que si usasemos directamente el número de la etiqueta,
        la red podría interpretar erróneamente que existe una relación natural 
        entre los tipos de percusión, lo cual no es cierto, todas las categorias 
        operan bajo el mismo espacio. De hecho, si dos etiquetas acaban siendo 
        'acústicamente' más parecidas, la red puede aprender embeddings más 
        cercanos entre sí en ese espacio, es decir, podemos trasladar similitudes
        acústicas a similitudes matemáticas sobre el espacio del Embedding.
        
        De esta manera es como podemos dotar de 'significado' a los números que 
        estamos generando y poder entrenarlos acorde a la etiqueta designada, 
        es este frangemnto que hace a nuestro autoencoder 'condicional'.
        
        Ahora, incorporamos la etiqueta del sonido mediante un embedding aprendido
        con el tensor de valores de la convolución. Llamaremos a este vector:
        
            H = [c(x); e(c)]; concatenación: 
                
                1. c(x): Caracterización convolucional. (256)
                
                2. e(c): Vector del embedding de las etiquetas. (8)
            
        '''
        
        self.class_embedding = nn.Embedding(num_embeddings = num_percs, 
                                            embedding_dim = 8)
        
        total_dim = self.flatten_dim + 8 # -> 264, dimensión total que pasará
                                         # a las capas finales.
        
        '''
        
        ########################## ESPACIO LATENTE ############################
        
        Vamos a construir ahora el 'ESPACIO LATENTE', la representación final 
        que aprenderá la red. Para ello transformaeros la representación 
        acústico-condicional compacta que obtenemos del embedding en los 
        parámetros de una 'DISTRIBUCIÓN LATENTE'.
        
        Al obligar a la red a comprimir toda la información en solo 16 números,
        La estamos forzando a descubrir las 'VARIABLES LATENTES', esdecir, 
        la información fundamental oculta que compone un sonido.
        
        Para ello tomamos H bajo una distribución gaussiana multivariante, en  
        nuestro caso, la tratamos como 16 componentes arbitrarias, cada una
        como una Gaussiana Normal. 
        
        Queremos averiguar cual es dicha distribución latente (una normal, ¿qué
        parámetros \mu, \sigma la gobiernan?), para ello calculamos:
            

        1. mu: Aplica una transformación lineal sobre H produciendo un vector 
               que representa la media de la distribución latente aproximada.
               (\mu \in \mathbb{R}^16)
               
                   \mu = W_\mu * H + b_\mu (nn.linear)
               
                   1. W_\mu: matriz 16 x 264 pesos para entrenamiento, son 
                             diferentes a los de W_\logvar.
                        
                   2. H: Vector resultado bloque de convoluciones + embedding.
               
                   3. b_\mu: Bias, hiperparámetros condicionales adicionales
                             también sujetos a entrenamiento, es diferente a 
                             b_\logvar.
                         
                         
        2. logvar: aplica otra transformación lineal distinta, con pesos distintos,
                   y produce un segundo vector que representa el logaritmo de 
                   la varianza de la distribución latente a modelar. 
                   
                       \logvar = W_\logvar * H + b_\logvar
                   
                       1. W_\logvar: matriz 16 x 264 pesos para entrenamiento, 
                                     son diferentes a los de W_\mu.
                                     
                       2. H: Vector resultado bloque de convoluciones + embedding.
                   
                       3. b_\logvar: Bias, hiperparámetros condicionales adicionales
                                     también sujetos a entrenamiento, es diferente 
                                     a b_\logvar.
                             
                   En lugar de predecir directamente la varianza (sigma^2), 
                   nos es más conveniente calcular la logvar al ser una 
                   transformación lineal que trabaja con números negativos. 
                   Además, en la 'DIVERGENCIA DE KULLBACK-LEIBER', una de las 
                   componentes fundamentales de la 'FUNCIÓN DE PÉRDIDA' que 
                   usaremos para el entrenamiento, utilizaremos tanto mu como 
                   logvar, con lo que tenerla ya precalculada nos será altamente
                   útil. A partir de \lgovar es inmedianto recuperar la varianza:
                        
                       \var = \sigma^2 = \exp(\logvar)
                   
                       
        ~~~~~~~~~~~~~~~~~~~ Auto-Encoding Variational Bayes ~~~~~~~~~~~~~~~~~~~
        
        '''
        
        self.mu = nn.Linear(total_dim, latent_dim)
        self.logvar = nn.Linear(total_dim, latent_dim)
    
        
    def forward(self, x, c):
            
        ''' 
        
        Esta es la función a la que llamamos para procesar los datos con 
        los métodos diseñados en __init__ .

        
        ########################### PARÁMETROS ################################

        1. x: Son los tensores de espectrogramas Mel que extraemos de varios
              audios preprocesados en 'audio_processor.py', esta entrada tomará 
              la forma de un tensor de 4 DIMENSIONES: ([Batch, 1, 64, 51])
        
              1. [Batch]: Indica el número de espectrogramas Mel que procesamos 
                          en simultaneo. 
        
              2. [1,64,51]: Dimensión de los elemento del Batch, 
        
                           1. 64 bandas Mel. (Dimensión espacial 1)
                     
                           2. 51 frames temporales.  (Dimensión espalcial 2)
                     
                           3. 1 canal, el número de neuronas donde viven los 
                              tensores, empezamos exclusivamente con una única
                              entrada (estrictamente necesario para poder procesar
                              los tensores). 
        
        2. c: Es el tensor de etiquetas de clase esta entrada tomará la forma     
              [Batch].
        
        
        ########################### OPERACIONES ###############################
        
        1. feauture: Aplicamos el tray de convoluciones y normalizaciones ya visto.
        
        2. .view(): Aplanamos la matriz procesada de los conv_blocks a un vector 
                    de 256 características. .view() en PyTorch sirve para cambiar 
                    la geométria dimensional de un tensor sin alterar los datos.
        
        3. .size(): Lo hacemos para indicar que sigamos manteniendo el numero de 
                    batch, es decir, mantenemos los audios que se procesan al mismo 
                    tiempo distinguibles entre si. ([Batch, 256])
            
        4. .class_embedding: Extraemos el vector del embedding pertintente de la 
                             matriz del embedding (la fila correspondiente a la 
                             etiqueta).
                            
        5. .cat(): Concatenamos el vector features con el c_emb. [Batch, 264]
       
        6. .mu()/.logvar(): Calculamos media y Logvar. Son estos valores los que
                            el encoder devolverá. ([Batch, latent_dim])        
        
        '''
            
        features = self.conv_blocks(x) # Aplicamos las convoluciones
        features = features.view(features.size(0), 256) 
            
        c_emb = self.class_embedding(c)         
        merged = torch.cat([features, c_emb], dim = 1) 
          
        mu = self.mu(merged)                        
        logvar = self.logvar(merged)                
        
        return mu, logvar


# ---------------------------------- PRUEBA -----------------------------------

# encoder = ConditionalEncoder(n_mels=64, num_percs=5, latent_dim=16)

# dummy_audio = torch.randn(4, 1, 64, 51) # 4 audios, 1 canal, 64 mels, 51 frames
# dummy_labels = torch.tensor([0, 1, 1, 1]) # 4 etiquetas 

# mu, logvar = encoder(dummy_audio, dummy_labels)

# print(f"Dim Mu: {mu.shape}")           # torch.Size([4, 16])
# print(f"Dim Log_Var: {logvar.shape}")   # torch.Size([4, 16])