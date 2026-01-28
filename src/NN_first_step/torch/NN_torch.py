import torch as tc

device = tc.device("mps" if tc.mps.is_available() else "cpu")

class PricingModel(tc.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        # Definiamo i "mattoni"
        self.layer1 = tc.nn.Linear(input_size, hidden_size)
        self.layer2 = tc.nn.Linear(hidden_size, hidden_size) # Stessa dimensione in/out per i layer interni
        self.layer3 = tc.nn.Linear(hidden_size, hidden_size)
        self.layerout = tc.nn.Linear(hidden_size, output_size) # Imbuto finale
        
        self.activation = tc.nn.LeakyReLU() # Il nostro "condimento"

    def forward(self, x):
        # 1. Primo strato + Attivazione
        x = self.layer1(x)
        x = self.activation(x)
        
        # 2. Secondo strato + Attivazione
        x = self.layer2(x)
        x = self.activation(x)
        
        # 3. Terzo strato + Attivazione
        x = self.layer3(x)
        x = self.activation(x)
        
        # 4. Output finale -> NIENTE attivazione (Lasciamo il valore grezzo)
        x = self.layerout(x)
        
        return tc.relu(x)  # Assicuriamoci che il payoff sia non negativo