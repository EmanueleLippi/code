Algoritmo nel notebook fully_Girsanov_stepByStep.ipynb
======================================================

Percorso notebook:
/Users/emanuelelippi/Library/Mobile Documents/com~apple~CloudDocs/Universita/Tirocinio/code/FBSDENN_Lippi/notebook/fully_Girsanov_stepByStep.ipynb

Obiettivo
---------
Il notebook implementa una FBSNN (Forward-Backward Stochastic Neural Network) per un sistema
fully-coupled, usando una strategia "step-by-step" in stile soft-init:

1) si parte con accoppiamento debole/disaccoppiato (const = 0.0),
2) si aumenta gradualmente il coupling fino a const = 1.0,
3) si fa un fine-tuning finale sul problema fully-coupled reale.

In questo modo la rete non affronta subito la dinamica più rigida, ma apprende prima una
struttura stabile dei pesi e poi si adatta progressivamente al sistema originale.


Struttura generale del codice
-----------------------------
Il notebook contiene:

- Classe astratta FBSNN:
  gestisce rete neurale, loss, simulazione dei cammini browniani, train loop.

- Classe concreta NN(FBSNN):
  definisce i termini del problema specifico:
  - mu_tf(...)   -> drift della forward SDE
  - phi_tf(...)  -> driver della backward SDE
  - sigma_tf(...) -> matrice di diffusione
  - g_tf(...)    -> condizione terminale

- Blocco main:
  - definisce parametri fisici e di training,
  - esegue curriculum su const,
  - esegue fine-tuning finale,
  - produce valutazione/predizione e plot.


Parte matematica implementata
-----------------------------
Stati:
- S, H, V, X_state (D = 4)

Forward:
- dS_t = mu(c - S_t) dt + s1 dW_t^1
- dH_t = mu(c - H_t) dt + s2 dW_t^2
- dV_t = a( X_t^2 + bH_t + c * V_from_Z ) dt + s3 dW_t^3
- dX_t = V_t dt

con
V_from_Z = 0.5 * ( e^t * const * Z_V / s3 - X_state )

Qui const controlla la forza dell'accoppiamento via Z:
- const = 0.0: accoppiamento molto ridotto (fase soft-init),
- const = 1.0: caso fully-coupled target.

Backward:
- driver costruito con i termini della soluzione analitica scelta,
- include il termine correttivo Girsanov:
  0.5 * exp(t) * (Z_V/s3) * a * c * (const - 1.0) * (Z_V/s3)

Questo termine è mantenuto fuori dal fattore exp(-t), coerentemente con la formula usata.


Aspetti tecnici importanti (TensorFlow v1)
------------------------------------------
Il grafo è statico, quindi il valore di const deve essere dinamico tramite placeholder.

Per questo nel notebook è presente:
- self.const_tf = tf.placeholder(tf.float32, shape=[])

e const viene letto in mu_tf/phi_tf come:
- const = tf.cast(self.const_tf, tf.float32)

Durante train/predict/evaluate, const viene passato nel tf_dict.
Questo garantisce che ogni stage del curriculum usi davvero il valore desiderato.


Loss e simulazione
------------------
Per ogni iterazione:
1) fetch_minibatch() genera cammini browniani W e griglia temporale t.
2) loss_function(...) propaga forward/backward con schema Euler.
3) si minimizza la loss con Adam.

Nel train loop:
- print ogni 10 iterazioni: Loss, Y0, tempo, learning rate.


Curriculum (soft-init step-by-step)
-----------------------------------
Nel main:

- const iniziale nei params: 0.0
- coupling_step = 0.2
- livelli: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

Per ciascun livello:
- model.const aggiornato (valore di riferimento),
- train su più sottofasi LR:
  (800, 1e-3), (1200, 5e-4), (1200, 1e-4)

Quindi ogni livello fa un mini ciclo di ottimizzazione coarse-to-fine.


Fine-tuning finale
------------------
Dopo il curriculum:
- const fissato a 1.0
- due fasi finali:
  - 2000 iterazioni a 1e-5
  - 1500 iterazioni a 1e-6

Serve per stabilizzare la soluzione sul problema target fully-coupled.


Logging e diagnostica
---------------------
Il notebook include metriche aggregate per ogni fase:

1) train(...) ora ritorna un dizionario con:
   - const, lr, n_iter,
   - last_loss, last_y0.

2) evaluate(const_value, n_batches=5):
   - esegue valutazione su mini-batch indipendenti,
   - ritorna mean/std di loss e Y0.

3) Per ogni stage stampa:
   [StageSummary] const=..., lr=..., iters=..., eval_loss=...±..., eval_Y0=...±..., time=...

4) Nel fine-tuning stampa:
   [FinalSummary] ...

5) Alla fine stampa un log compatto completo:
   phase=..., const=..., lr=..., iters=..., eval_loss=..., eval_y0=...

Questo serve per decidere se aumentare iterazioni o cambiare LR in alcuni livelli.


Interpretazione pratica dei risultati
-------------------------------------
Segnali utili:

- Buono:
  - eval_mean_loss scende con continuità stage dopo stage,
  - eval_std_loss non esplode,
  - eval_mean_y0 è stabile o converge.

- Da migliorare:
  - perdita che si blocca presto nei livelli alti (const 0.6+),
  - oscillazioni forti di Y0,
  - std loss alta persistente.

In questi casi conviene:
- aumentare iterazioni nei livelli alti,
- aggiungere una fase extra con LR più piccolo (es. 5e-5),
- eventualmente ridurre il coupling_step (es. 0.1) vicino a const=1.0.


Perché questo approccio è utile
-------------------------------
Il problema fully-coupled può essere numericamente rigido.
Il curriculum su const:
- facilita l'inizializzazione dei pesi,
- riduce instabilità iniziale,
- rende più robusta la convergenza verso la dinamica reale.

In sintesi, il notebook implementa un addestramento "continuation/homotopy-like":
si deforma progressivamente il problema da più semplice (disaccoppiato)
a quello reale (fully-coupled), riusando sempre i pesi già appresi.
