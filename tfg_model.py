"""
Created on Sat Jan 3 14:35:43 2026

@author: andre
"""

import torch
import torch.nn as nn
import tfg_encoder
import tfg_decoder_NDDSP


class CVAE(nn.Module):
    
      '''
    
      Después calcular los vectores de dimensión 16 y sus media (\mu) y logvarianza 
      (\log \sigma^2) determinadas, obtenidas de una distribución gaussiana 
      multivariante en base a dichos 16 valores, para contrarrestar el comportamiento 
      estocástico al tomar una muestra arbitraria de esta distribución (z) 
      aplicaremos uan 'REPARAMETRIZACIÓN'. 
    
      Sea z una variable aleatoria de la distribución Gaussiana multivariante 
      definida N(\mu, \sigma^2), aplicar el backpropagation con esta muestra 
      aleatoria introduciría una componente estocástica en el descenso de gradiente 
      (dificulta enormemente el cálculo de la derivada) retrasando el aprendizaje
      de la red, lo que hacemos con esta reparametrización es entonces intrducir 
      una variable aleatoria externa ~N(0,1) de tal manera que, en lugar de tomar
      z como una variable aleatoria, la podemos reexpresar mediante una 
      transformación lineal DETERMINISTA. Aín así, el proceso global sigue siendo
      ESTOCÁSTICO, pero esta aleatoriedad la aislamos sobre una variable auxiliar
      \epsilon:

          z = f(\mu, \sigma, \epsilon) = \mu  + \sigma \cdot \epsilon (prod 1 a 1)
                
      Esta reformulación permite calcular derivadas parciales continuas con respecto
      a \mu y \sigma, habilitando la retropropagación del error al no tener ya 
      la componente estocástica dentro de mu y sigma. Esta reparametrización es 
      la invención matemática exacta que permitió que los autoencoders variacionales
      existieran. Sin este truco,la red neuronal sería físicamente incapaz de 
      'aprender' bajo las condiciones que la hemos impuesto.
      
      Todo esto viene implementado en la función reparametrize.
      
      ---------------------------------- NOTA ---------------------------------
    
      La teoría correspondiente al mecanismo exacto de los CVAE, la reparametrización
      y todos los conceptos probabilísticos pertinentes vendrán explicados bien 
      dentro de la memoria, no quería excederme (más aún) dentro del código. 
      
      -------------------------------------------------------------------------
    
      ~~~~~~~~~~~~~~~~~ "Auto-Encoding Variational Bayes" [2.4] ~~~~~~~~~~~~~~~~~
    
      ~~~~~~~~~~~~~~~~~~~~ Goodfellow "Deep Learning" [20] ~~~~~~~~~~~~~~~~~~~~~~
   
      '''
    
      def __init__(self, num_percs = 5, latent_dim = 16):
        
          super().__init__()
        
          # Instanciamos las dos mitades de la red, el encoder y el decoder. 
        
          self.encoder = tfg_encoder.Conditional_Encoder(num_percs=num_percs, latent_dim=latent_dim)
          self.decoder = tfg_decoder_NDDSP.NDDSP_Decoder(num_percs=num_percs, latent_dim=latent_dim)

      def reparameterize(self, mu, logvar):
        
          std = torch.exp(0.5 * logvar) # \logvar = \log(\sigma^2) = 2 * \log(\sigma) 
                                        # \sigma = torch.exp(0.5 * logvar)
        
          eps = torch.randn_like(std) # \epsilon ~ N(0,1), eps == std ([batch,16])
         
          return mu + eps * std


      def forward(self, x, c):
        
          '''
        
          Ahora si construimos la función que conecta el encoder y el decoder,
          es decir, ya tenemos el camino completo del tensor de audio.
          
          ########################## PARÁMETROS ###############################

          1. x: Tensor del Espectrograma Mel. ([Batch, 1, 64, 51])
        
          2. c: Tensor de etiquetas de clase.([Batch])
        
        
          Devolvemos new_x como el nuevo tensor  generado por el decoder de la 
          red, devolveremos además mu y logvar para calcular la función de 
          pérdida mediante el 'ELBO' y la 'DIVERGENCIA DE KL'. ([Batch, 1, 64, 51])
        
          '''
        
          mu, logvar = self.encoder(x, c) # Usamos el encoder
          z = self.reparameterize(mu, logvar) # Truco de la Reparametrización
          new_x = self.decoder(z, c) # Usamos el decoder
     
          return new_x, mu, logvar 