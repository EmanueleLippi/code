import numpy as np 
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
import time
from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Optional

class FBSNN(ABC):
    def __init__(self, 
    Xi: tf.Tensor, T: float, M: int, N: int,
    D: int, layers: List[int]):
        self.Xi = Xi #Tensore di punti iniziali
        self.T = T #Tempo finale
        self.M = M #Numero di traiettorie
        self.N = N #Numero di step temporali
        self.D = D #Numero di dimensioni
        self.layers = layers #Lista di layer --> D+1 --> 1

        #Creo la rete neurale
        self.weights, self.biases = self.initialize_NN(layers)

        #Definisco la sessione di TensorFlow --> Sessione: Esecuzione del grafo computazionale
        # SPIEGAZIONE SESSIONE TENSORFLOW:
        # In TensorFlow 1.x, il codice funziona in due fasi:
        # 1. Costruzione del grafo (il progetto): definisci le operazioni (es. tf.placeholder, tf.matmul) ma non le esegui.
        # 2. Esecuzione (la sessione): è il "motore" che preleva il grafo e lo esegue sull'hardware.
        #
        # Questa riga specifica `self.sess = tf.Session(...)`:
        # - Alloca le risorse (memoria, link a GPU/CPU).
        # - Crea il ponte tra il Python (astratto) e il C++ (esecutivo).
        # - Mantiene lo stato delle variabili (es. i pesi self.weights) tra una `sess.run()` e l'altra.
        #
        # Parametri usati:
        # - allow_soft_placement=True: se un'op non esiste per GPU, usa la CPU senza crashare.
        # - log_device_placement=True: stampa nel terminale dove sta girando ogni operazione.
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        
        #Definisco i placeholder --> Variabili che verranno riempite durante l'addestramento
        # SPIEGAZIONE PLACEHOLDER:
        # In TF 1.x, i placeholder sono come "buchi" nel grafo. Non contengono dati di per sé,
        # ma definiscono la forma (shape) e il tipo (dtype) dei dati che entreranno.
        #
        # self.learning_rate: Tasso di apprendimento (scalare).
        # self.t_tf: Tiene traccia del tempo. Deve essere una matrice (M, N+1, 1).
        # self.W_tf: Tiene traccia del moto Browniano (input stocastico).
        # self.Xi_tf: Punto di partenza (costante per tutta la simulazione).
        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        self.t_tf = tf.placeholder(tf.float32, shape=[M, N+1, 1])
        self.W_tf = tf.placeholder(tf.float32, shape=[M, N+1, D])
        self.Xi_tf = tf.placeholder(tf.float32, shape=[1, D])

        #Definisco il grafo computazionale --> Operazioni che verranno eseguite
        # SPIEGAZIONE GRAFO COMPUTAZIONALE:
        # È come una ricetta. Definisce cosa fare con i dati, ma non li esegue ancora.
        # In TF 1.x, tutto è un grafo statico. Una volta definito, non puoi cambiare le operazioni.
        #
        # self.loss: La funzione di costo (errore) che vogliamo minimizzare.
        # self.X_pred, self.Y_pred: Le soluzioni predette dalla rete.
        # self.Y0_pred: La soluzione al tempo t=0 (il nostro target da confrontare con Xi).
        self.loss, self.X_pred, self.Y_pred, self.Y0_pred = self.loss_function(self.t_tf, self.W_tf, self.Xi_tf)

       #Definisco l'ottimizzatore --> Algoritmo che aggiorna i pesi della rete
       # SPIEGAZIONE OTTIMIZZATORE:
       # È il motore che corregge gli errori. In questo caso, usiamo Adam.
       # - learning_rate: Quanto "grande" è il passo che fa per correggere gli errori.
       # - minimize(self.loss): Dice all'ottimizzatore: "Trova i pesi che rendono self.loss più piccolo possibile".
        self.optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss)

       #Inizializzo le variabili --> Assegno valori iniziali ai pesi e ai bias
       # SPIEGAZIONE INIZIALIZZAZIONE:
       # I pesi non possono partire da 0, altrimenti la rete non impara nulla.
       # Usiamo Xavier initialization: distribuisce i pesi in modo che la varianza rimanga costante tra i layer.
        init = tf.global_variables_initializer()
        self.sess.run(init)

    # SPIEGAZIONE INIZIALIZZAZIONE RETE NEURALE:
    #
    # SCOPO:
    # Costruisce le variabili TensorFlow (Tensori) che rappresentano i parametri
    # apprendibili della rete: Pesi (Weights, W) e Bias (b).
    #
    # FUNZIONAMENTO:
    # 1. Input: 'layers' è una lista di interi che definisce l'architettura.
    #    Es: [D+1, 256, 256, 1] significa:
    #    - Input layer: D+1 neuroni (t + X)
    #    - Hidden layers: 2 strati da 256 neuroni
    #    - Output layer: 1 neurone (u)
    # 2. Iterazione: Per ogni connessione tra due layer adiacenti (l -> l+1):
    #    - Crea una matrice di pesi W di dimensione [N_in, N_out] usando Xavier Init.
    #    - Crea un vettore di bias b di dimensione [1, N_out] inizializzato a zero.
    # 3. Return: Restituisce le liste di tutte le variabili W e b create.
    def initialize_NN(self, layers):
        weights = [] # Lista per salvare i tensori dei pesi
        biases = []  # Lista per salvare i tensori dei bias
        num_layers = len(layers) 
        
        # Ciclo su tutti i layer tranne l'ultimo (perché i pesi connettono l -> l+1)
        for l in range(0, num_layers-1):
            # Inizializzazione Pesi (W):
            # Usa Xavier initialization per stabilità numerica.
            W = self.xavier_init(size = [layers[l], layers[l+1]])
            
            # Inizializzazione Bias (b):
            # Inizializzati a zero. tf.Variable li rende modificabili dall'ottimizzatore.
            b = tf.Variable(tf.zeros([1, layers[l+1]], dtype=tf.float32))
            
            weights.append(W)
            biases.append(b)

        return weights, biases

       
    # SPIEGAZIONE XAVIER/GLOROT INITIALIZATION:
    # 
    # LOGICA GENERALE:
    # L'inizializzazione Xavier (o Glorot) serve a risolvere il problema della scomparsa o esplosione
    # dei gradienti nelle reti neurali profonde. L'obiettivo è mantenere la varianza dei segnali di
    # attivazione circa costante attraverso i vari layer della rete durante la propagazione in avanti,
    # e mantenere costante la varianza dei gradienti durante la retropropagazione.
    # Se i pesi sono troppo piccoli, il segnale svanisce (vanishes); se troppo grandi, esplode.
    # Xavier trova un bilanciamento ideale basandosi sulle dimensioni di input e output del layer.
    #
    # FUNZIONAMENTO SPECIFICO DEL CODICE:
    # 1. Input: Riceve 'size', una tupla (in_dim, out_dim) che rappresenta le connessioni del layer
    #    (es. neuroni strato precedente -> neuroni strato successivo).
    # 2. Calcolo Deviazione Standard (stddev): Applica la formula di Glorot per la distribuzione Normal:
    #    stddev = sqrt(2 / (fan_in + fan_out))
    #    dove fan_in = in_dim e fan_out = out_dim.
    # 3. Generazione Pesi: Crea un tensore di forma [in_dim, out_dim] usando una distribuzione 
    #    normale troncata ('truncated_normal') con media 0 e la stddev calcolata.
    #    La normale troncata scarta valori molto distanti dalla media (oltre 2 deviazioni standard),
    #    garantendo stabilità numerica ed evitando pesi outlier troppo grandi.
    # 4. Return: Restituisce una tf.Variable inizializzata con questi valori, pronta per essere
    #    ottimizzata durante il training.
    def xavier_init(self, size: Tuple[int, int]) -> tf.Variable:
        in_dim = size[0] #Dimensione input
        out_dim = size[1] #Dimensione output
        xavier_stddev = np.sqrt(2/(in_dim + out_dim)) #Deviazione standard
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    # SPIEGAZIONE FEED-FORWARD NEURAL NETWORK:
    #
    # LOGICA GENERALE:
    # Questa funzione implementa la propagazione in avanti (forward pass) di una rete neurale
    # completamente connessa (Dense Neural Network o MLP).
    # L'input X attraversa i vari strati (hidden layers). In ogni strato, avviene una trasformazione
    # lineare (prodotto matriciale Input * Pesi + Bias) seguita da una funzione di attivazione non lineare.
    # L'ultimo strato solitamente non ha funzione di attivazione (o ne ha una diversa) per permettere
    # alla rete di predire valori in un range arbitrario (regressione).
    #
    # FUNZIONAMENTO SPECIFICO DEL CODICE:
    # 1. Input: Riceve i dati di input X e le liste di tensori per pesi (weights) e bias.
    # 2. Hidden Layers (Ciclo for):
    #    Itera attraverso tutti gli strati tranne l'ultimo.
    #    Calcola H = sin(H * W + b).
    #    Nota: L'uso di sin() come funzione di attivazione è tipico nelle Physics-Informed Neural Networks (PINNs)
    #    o nella risoluzione di equazioni differenziali, poiché il seno è una funzione infinitamente derivabile,
    #    al contrario di ReLU che non è derivabile nello zero (o tanh/sigmoid che possono saturare).
    # 3. Output Layer:
    #    L'ultimo strato applica solo la trasformazione lineare Y = H * W + b.
    #    Non c'è funzione di attivazione (attivazione lineare) per non limitare il range dell'output.
    # 4. Return: Restituisce il tensore Y che rappresenta la predizione della rete.
    def neural_net(self, X: tf.Tensor, weights: List[tf.Variable], biases: List[tf.Variable]) -> tf.Tensor:
        num_layers = len(weights) + 1 #Numero di layer: utilizziamo il numero di pesi piu' 1 per contare anche l'input

        H = X #Definisco H come X, inizialmente sono uguali ma poi H diventerà il risultato
        for l in range(0, num_layers-2): #Ciclo per tutti i layer tranne l'ultimo
            W = weights[l] #definisco il peso alla layer l-esima
            b = biases[l] #definisco il bias alla layer l-esima
            H = tf.sin(tf.add(tf.matmul(H, W), b)) #Applico la funzione di attivazione sin
        W = weights[-1] #definisco il peso alla layer finale
        b = biases[-1] #definisco il bias alla layer finale
        Y = tf.add(tf.matmul(H, W), b)
        return Y #Restituisco Y --> Output della rete
        
    # SPIEGAZIONE APPROSSIMAZIONE SOLUZIONE E GRADIENTE:
    #
    # LOGICA GENERALE:
    # Questa funzione utilizza la rete neurale per approssimare la soluzione u(t, x) dell'equazione differenziale.
    # Inoltre, calcola il gradiente spaziale Du = ∇u(t, x) rispetto alla variabile di stato X.
    # In molti problemi FBSDE, il termine di diffusione Z è legato proprio al gradiente della soluzione u (Z = ∇u * sigma).
    #
    # FUNZIONAMENTO SPECIFICO DEL CODICE:
    # 1. Input: Prende in ingresso il tempo t e lo stato X.
    # 2. Concatenazione: Unisce t e X in un unico tensore di input [t, X] per la rete neurale.
    #    Questo perché la soluzione u dipende sia dal tempo che dallo spazio.
    # 3. Forward Pass: Chiama self.neural_net per ottenere il valore u.
    # 4. Calcolo Gradiente: Utilizza la differenziazione automatica (tf.gradients) per calcolare
    #    le derivate parziali di u rispetto a X.
    #    tf.gradients(u, X)[0] restituisce il vettore gradiente ∇_x u.
    # 5. Return: Restituisce la coppia (u, Du).
    def net_u(self, t: tf.Tensor, X: tf.Tensor) -> tf.Tensor:
        u = self.neural_net(tf.concat([t, X], 1), self.weights, self.biases) #Calcolo u
        Du = tf.gradients(u, X)[0] #Calcolo Du 

        return u, Du #Restituisco u e Du
    
    # SPIEGAZIONE GRADIENTE DELLA CONDIZIONE TERMINALE:
    #
    # LOGICA GENERALE:
    # Nei problemi FBSDE, la condizione finale dell'equazione Backward è spesso data da una funzione 
    # g(X_T) al tempo terminale T. La condizione per il processo Z al tempo T è data dal gradiente
    # di questa funzione g rispetto allo stato X: Z_T = ∇g(X_T) * sigma.
    # Questa funzione calcola tale gradiente, che serve come "target" o condizione al contorno
    # finale durante l'addestramento della rete.
    #
    # FUNZIONAMENTO SPECIFICO DEL CODICE:
    # 1. Input: Riceve lo stato X (tipicamente al tempo finale T).
    # 2. Calcolo g(X): Chiama la funzione self.g(X) (definita altrove nel codice, rappresenta la condizione terminale).
    # 3. Differenziazione Automatica: Usa tf.gradients per calcolare le derivate parziali di g(X) rispetto a X.
    #    Questo evita di dover calcolare a mano il gradiente di g, rendendo il codice flessibile per diverse g.
    # 4. Return: Restituisce il gradiente ∇g(X).
    def Dg_tf(self, X: tf.Tensor) -> tf.Tensor:
        return tf.gradients(self.g_tf(X), X)[0] #Calcolo Dg ovvero il gradiente di g rispetto a X

    # SPIEGAZIONE FUNZIONE DI COSTO (SOLUZIONE FBSDE):
    #
    # LOGICA GENERALE:
    # Questa funzione è il cuore dell'algoritmo. Risolve l'equazione differenziale stocastica forward-backward (FBSDE)
    # minimizzando la discrepanza tra due modi di calcolare la soluzione Y:
    # 1. Tramite la rete neurale: Y = u(t, X)
    # 2. Tramite l'evoluzione stocastica (schema di Eulero-Maruyama) dell'equazione Backward.
    # L'idea fondamentale è che se la rete u(t, X) è corretta, deve soddisfare l'equazione stocastica passo-passo.
    #
    # STRUTTURA A DUE PARTI:
    # PARTE 1: Evoluzione temporale (Ciclo for)
    #    Simula le traiettorie del processo Forward X e Backward Y nel tempo.
    #    Ad ogni passo, impone che il valore predetto dalla rete al tempo t+1 (Y1) coincida con quello
    #    ottenuto evolvendo l'equazione differenziale dal tempo t (Y1_tilde).
    # PARTE 2: Condizione Terminale
    #    Al tempo finale T, impone che la soluzione Y coincida con la condizione terminale nota g(X_T).
    #
    # FUNZIONAMENTO SPECIFICO DEL CODICE (RIGA PER RIGA):
    def loss_function(self, t: tf.Tensor, W: tf.Tensor, Xi: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        loss = 0 # Inizializzo loss a 0. Accumulerà gli errori commessi ad ogni passo temporale.
        X_list = [] # Lista per salvare la storia delle traiettorie di X (memoria per visualizzazione/analisi).
        Y_list = [] # Lista per salvare la storia delle traiettorie di Y.

        # PARTE INIZIALE (Tempo t=0)
        t0 = t[:,0,:] # Tempo iniziale (per tutte le traiettorie M).
        W0 = W[:,0,:] # Valore iniziale del moto Browniano.
        X0 = tf.tile(Xi, [self.M, 1]) # Punto di partenza X0 duplicato per le M traiettorie simulate (M x D).
        
        # Calcolo Y0 e Z0 al tempo 0 usando la rete neurale.
        # Questa è la stima iniziale della soluzione che vogliamo ottimizzare.
        Y0, Z0 = self.net_u(t0, X0) 

        X_list.append(X0) # Salvo X0 nello storico.
        Y_list.append(Y0) # Salvo Y0 nello storico.

        # PARTE 1: CICLO TEMPORALE (Eulero-Maruyama)
        for n in range(0, self.N):
            # Estrazione valori al passo successivo n+1
            t1 = t[:, n+1, :] # Tempo t_{n+1}
            W1 = W[:, n+1, :] # Moto Browniano W_{n+1}
            
            # --- EQUAZIONE FORWARD (X) ---
            # X_{n+1} = X_n + mu(X_n)*dt + sigma(X_n)*dW
            # Calcolo la nuova posizione X1 usando lo schema di Eulero.
            # self.mu_tf(...): termine di drift (velocità deterministica).
            # self.sigma_tf(...): termine di diffusione (volatilità).
            # (W1 - W0): incremento del moto Browniano dW.
            X1 = X0 + self.mu_tf(t0,X0,Y0,Z0)*(t1-t0) + tf.squeeze(tf.matmul(self.sigma_tf(t0,X0,Y0),tf.expand_dims(W1-W0,-1)), axis=[-1])
            
            # --- EQUAZIONE BACKWARD (Y) - Target "Fisico" ---
            # Y_{n+1}_tilde = Y_n + phi(...) * dt + Z_n * dW
            # Calcolo dove "dovrebbe" essere Y al passo successivo secondo l'equazione differenziale.
            # Y1_tilde è il nostro target stocastico basato sulla fisica del problema.
            Y1_tilde = Y0 + self.phi_tf(t0,X0,Y0,Z0)*(t1-t0) + tf.reduce_sum(Z0*tf.squeeze(tf.matmul(self.sigma_tf(t0,X0,Y0),tf.expand_dims(W1-W0,-1))), axis=1, keepdims = True)
            
            # --- PREDIZIONE RETE NEURALE ---
            # Calcolo la predizione della rete per il nuovo stato X1 al tempo t1.
            Y1, Z1 = self.net_u(t1, X1)
            
            # --- CALCOLO LOSS LOCALE ---
            # La rete deve essere coerente con l'equazione: Y predettto (Y1) deve essere uguale a Y calcolato (Y1_tilde).
            # Sommo l'errore quadratico medio su tutte le traiettorie.
            loss += tf.reduce_sum(tf.square(Y1 - Y1_tilde))

            # Aggiornamento variabili per il passo successivo
            t0 = t1
            W0 = W1
            X0 = X1
            Y0 = Y1
            Z0 = Z1

            # Salvataggio nello storico
            X_list.append(X0)
            Y_list.append(Y0)

        # PARTE 2: CONDIZIONE TERMINALE
        # Alla fine del tempo T (dopo N passi), la soluzione Y deve rispettare la condizione al contorno g(X_T).
        # loss_terminal_Y: Penalizza la differenza tra Y finale predetto e g(X_finale).
        loss += tf.reduce_sum(tf.square(Y1 - self.g_tf(X1)))
        
        # loss_terminal_Z: Penalizza la differenza tra Z finale (gradiente della rete) e il gradiente di g.
        # Questo aiuta a stabilizzare l'apprendimento del gradiente Z.
        loss += tf.reduce_sum(tf.square(Z1 - self.Dg_tf(X1)))

        # Pack dei risultati per il ritorno
        X = tf.stack(X_list, axis=1) # Tensore [M, N+1, D]
        Y = tf.stack(Y_list, axis=1) # Tensore [M, N+1, 1]

        # Ritorna:
        # loss: scalare da minimizzare
        # X, Y: traiettorie complete (per plot/analisi)
        # Y[0,0,0]: il valore iniziale Y0 della prima traiettoria (spesso il valore che cerchiamo, es. prezzo opzione).
        return loss, X, Y, Y[0,0,0]


    # SPIEGAZIONE GENERAZIONE BATCH (SIMULAZIONE STOCASTICA):
    #
    # LOGICA GENERALE:
    # Questa funzione genera i dati per una iterazione di training ("minibatch").
    # Nel contesto delle equazioni differenziali stocastiche, i "dati" non sono fissi, ma vengono
    # generati simulando nuove traiettorie del moto Browniano ogni volta.
    # Questo permette alla rete di esplorare l'intero spazio delle possibili evoluzioni del sistema.
    #
    # FUNZIONAMENTO SPECIFICO DEL CODICE:
    # 1. Recupero Parametri: Ottiene T, M (numero traiettorie), N (step temporali), D (dimensioni).
    # 2. Inizializzazione: Crea array di zeri per gli incrementi temporali (Dt) e del moto Browniano (DW).
    # 3. Calcolo Incrementi:
    #    - dt = T / N (passo temporale costante).
    #    - DW: Genera incrementi casuali secondo la distribuzione normale N(0, dt).
    #      La varianza è dt, quindi la deviazione standard è sqrt(dt).
    #      Nota: Il primo punto (indice 0) rimane a 0 perché il processo parte dall'origine relativa.
    # 4. Somma Cumulativa (Cumsum):
    #    - t: Costruisce la griglia temporale sommando i dt.
    #    - W: Costruisce le traiettorie del moto Browniano sommando gli incrementi DW.
    # 5. Return: Restituisce i tensori completi t e W di dimensione [M, N+1, ...].
    def fetch_minibatch(self):
        T = self.T
        M = self.M
        N = self.N
        D = self.D

        Dt = np.zeros((M, N+1, 1))
        DW = np.zeros((M, N+1, D))
        
        dt = T/N

        Dt[:,1:,:] = dt
        DW[:,1:, :] = np.sqrt(dt)*np.random.normal(size=(M,N,D))

        t = np.cumsum(Dt, axis=1)
        W = np.cumsum(DW, axis=1)

        return t, W

    # SPIEGAZIONE METODO TRAIN (ADDESTRAMENTO):
    #
    # CONTESTO GENERALE:
    # Questo metodo gestisce il ciclo di addestramento della rete neurale. 
    # In questo framework FBSDE, l'addestramento non avviene su un dataset fisso (come immagini o testi),
    # ma su traiettorie stocastiche generate dinamicamente ad ogni iterazione.
    # L'obiettivo è minimizzare la loss function definita precedentemente, che misura quanto bene
    # la rete soddisfa l'equazione differenziale stocastica e le condizioni al contorno.
    # 
    # UTILIZZO SPECIFICO:
    # Viene chiamato dall'utente dopo aver istanziato la classe. Esegue 'epochs' iterazioni 
    # di discesa del gradiente (Adam optimizer).
    def train(self, epochs: int, learning_rate: float):
        # Inizializzo il timer per monitorare quanto tempo impiegano le iterazioni.
        # Utile per capire la velocità di addestramento (iterazioni/secondo).
        start_time = time.time()
        
        # CICLO DI TRAINING
        # Itera per il numero di epoche specificato.
        for epoch in range(epochs):
            
            # 1. GENERAZIONE DEI DATI (BATCH)
            # Ad ogni epoca, generiamo un nuovo set di traiettorie del moto Browniano.
            # self.fetch_minibatch() restituisce:
            # - t_batch: griglia temporale [M, N+1, 1]
            # - W_batch: traiettorie del moto Browniano [M, N+1, D]
            # Questo è cruciale: la rete non memorizza dati, ma impara la dinamica dell'equazione
            # vedendo sempre nuove realizzazioni del processo stocastico.
            t_batch, W_batch = self.fetch_minibatch()
            
            # 2. DIZIONARIO DI ALIMENTAZIONE (FEED DICT)
            # Mappiamo i dati Python (numpy array) ai Placeholder di TensorFlow definiti in __init__.
            # - self.Xi_tf: punto iniziale (fisso).
            # - self.t_tf: i tempi generati.
            # - self.W_tf: i moti Browniani generati.
            # - self.learning_rate: il passo di apprendimento passato come argomento.
            tf_dict = {self.Xi_tf: self.Xi, self.t_tf: t_batch, self.W_tf: W_batch, self.learning_rate: learning_rate}
            
            # 3. ESECUZIONE DELLO STEP DI OTTIMIZZAZIONE
            # Questa è l'istruzione chiave. sess.run esegue il grafo computazionale.
            # Chiediamo di eseguire 'self.train_op': questa operazione calcola i gradienti della loss
            # rispetto ai pesi e li aggiorna (minimizza l'errore).
            # Non ci serve il valore di ritorno qui, ci interessa l'effetto collaterale (aggiornamento pesi).
            self.sess.run(self.train_op, tf_dict)

            # 4. MONITORAGGIO E LOGGING (Ogni 100 epoche)
            # Per non rallentare troppo il processo, calcoliamo e stampiamo le metriche solo ogni 100 passi.
            # (Nota: nel codice originale era 10, ma spesso si aumenta per velocità).
            if epoch % 10 == 0:
                # Calcoliamo il tempo trascorso per queste 10 iterazioni.
                elapsed = time.time() - start_time
                
                # Eseguiamo nuovamente sess.run, ma questa volta chiediamo valori specifici per monitorare:
                # - self.loss: il valore attuale dell'errore (deve scendere).
                # - self.Y0_pred: la stima del valore iniziale (es. prezzo opzione). Deve convergere.
                # - self.learning_rate: per conferma.
                # Nota: passiamo lo stesso tf_dict, quindi ricalcola su QUESTO batch corrente.
                loss_value, Y0_value, learning_rate_value = self.sess.run([self.loss, self.Y0_pred, self.learning_rate], tf_dict)
                
                # Stampiamo i risultati formattati:
                # - %.3e: notazione scientifica con 3 decimali (utile per loss molto piccole).
                # - %.3f: float con 3 decimali (utile per Y0).
                print("Epoch: %d, Loss: %.3e, Y0: %.3f, Learning Rate: %.3e, Time: %.2fs" % (epoch, loss_value, Y0_value, learning_rate_value, elapsed))
                
                # Resettiamo il timer per misurare il prossimo blocco di iterazioni.
                start_time = time.time()
    
    # SPIEGAZIONE METODO PREDICT (INFERENZA/TEST):
    #
    # CONTESTO GENERALE:
    # Questo metodo viene utilizzato DOPO l'addestramento per valutare il modello su nuovi dati 
    # o per esaminare le traiettorie predette. A differenza di 'train', qui non avviene nessuna
    # ottimizzazione dei pesi (i pesi sono congelati).
    # Serve a rispondere alla domanda: "Data una certa evoluzione del moto Browniano W e un tempo t,
    # quali sono le traiettorie X (Forward) e Y (Backward) che la rete prevede?"
    #
    # UTILIZZO SPECIFICO:
    # Riceve in input dei dati di test (spesso generati allo stesso modo di fetch_minibatch, 
    # ma tenuti separati come test set) e restituisce le matrici delle soluzioni.
    def predict(self, Xi_star: np.ndarray, t_star: np.ndarray, W_star: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        
        # 1. PREPARAZIONE DEI DATI
        # Creiamo il dizionario per alimentare i placeholder.
        # Notiamo che NON passiamo 'learning_rate' perche' non stiamo addestrando (non usiamo train_op).
        # Xi_star: punto di partenza.
        # t_star: griglia temporale di test.
        # W_star: realizzazioni del moto Browniano di test.
        tf_dict = {self.Xi_tf: Xi_star, self.t_tf: t_star, self.W_tf: W_star}

        # 2. ESECUZIONE DELLA PREDIZIONE
        # Eseguiamo la sessione chiedendo in output:
        # - self.X_pred: la traiettoria del processo Forward.
        # - self.Y_pred: la traiettoria del processo Backward (soluzione dell'EQ).
        # Questi nodi del grafo usano i pesi ormai addestrati per calcolare l'output.
        #
        # Nota: Potremmo fare una sola chiamata sess.run([self.X_pred, self.Y_pred], tf_dict)
        # per efficienza, ma separarli funziona ugualmente.
        X_star = self.sess.run(self.X_pred, tf_dict)
        Y_star = self.sess.run(self.Y_pred, tf_dict)

        # 3. RITORNO DEI RISULTATI
        # Restituisce le matrici numpy con i valori predetti.
        # Shape tipica: [M, N+1, D] o [M, N+1, 1]
        return X_star, Y_star

    # METODI ASTRATTI PER LA DEFINIZIONE DEL PROBLEMA:
    # Questi metodi devono essere implementati nelle sottoclassi per definire 
    # la specifica equazione differenziale (PDE) che si vuole risolvere.
    
    # SPIEGAZIONE PHI (GENERATORE BACKWARD):
    # Rappresenta il termine di "drift" dell'equazione Backward, o il termine non lineare della PDE.
    # Nell'equazione: dY_t = -phi(t, X_t, Y_t, Z_t)dt + Z_t dW_t
    # Corrisponde alla funzione f(t, x, y, z) nei paper accademici.
    # Input: Tempo t, Stato X, Valore Y, Gradiente Z.
    @abstractmethod
    def phi_tf(self, t, X, Y, Z):
        pass

    # SPIEGAZIONE G (CONDIZIONE TERMINALE):
    # Rappresenta la condizione al contorno finale al tempo T.
    # Y_T = g(X_T).
    # Esempio: Nel pricing di opzioni, questa è la funzione di Payoff (es. max(X-K, 0)).
    @abstractmethod
    def g_tf(self, X):
        pass

    # SPIEGAZIONE MU (DRIFT FORWARD):
    # Rappresenta la parte deterministica dell'evoluzione dello stato X.
    # Nell'equazione: dX_t = mu(t, X_t, Y_t, Z_t)dt + sigma(...)dW_t
    # Input: Tempo t, Stato X, e talvolta anche Y, Z (nei problemi fully coupled).
    # Default: Ritorna 0 (nessun drift).
    @abstractmethod
    def mu_tf(self, t, X, Y, Z):
        M = self.M
        D = self.D
        return np.zeros([M, D])

    # SPIEGAZIONE SIGMA (DIFFUSIONE FORWARD):
    # Rappresenta la volatilità o la parte stocastica dell'evoluzione dello stato X.
    # Matrice DxD che moltiplica il moto Browniano dW.
    # Default: Ritorna matrice identità (moto Browniano standard non scalato).
    @abstractmethod
    def sigma_tf(self, t, X, Y):
        M = self.M
        D = self.D
        return tf.matrix_diag(tf.ones([M, D]))