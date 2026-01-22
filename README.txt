# Progetto BSDE Solver con Regressione B-Spline
=================================================

Questo progetto implementa un risolutore per Equazioni Differenziali Stocastiche BACKWARD (BSDE) utilizzando l'algoritmo di Longstaff-Schwarz 
potenziato con regressione su base B-Spline. Il focus principale è la valutazione di contratti finanziari e derivati (Call Option, Contratti Logaritmici, Payoff Quadratici).

## Struttura del Progetto

Il progetto è organizzato come segue:

/code
  ├── /src                     # Codice sorgente principale
  │    ├── /lsm
  │         ├── /basis         # Modulo per le B-Spline (Fit e valutazione derivata)
  │         ├── /bsde          # Solver BSDE (Generale e Lineare)
  │         ├── /modelli       # Formule chiuse (Black-Scholes) per benchmark
  │         └── /longstaff_schwarz # Engine classico LSM per Opzioni Americane
  │
  ├── /test                    # Script di testing e validazione
  │    ├── test_BSDE.py        # Test su Call Option Europea (vs Black-Scholes)
  │    ├── test_BSDE_log.py    # Test su Contratto Logaritmico (vs Soluzione Analitica)
  │    ├── test_BSDE_quadratic.py # Test su Payoff Quadratico (Stress test concavità)
  │    └── test_LSM_visual.py  # Test visuale per LSM classico
  │
  ├── /result                  # Cartella di output (generata automaticamente)
  │    ├── /bsde               # Risultati CSV e grafici Call Option
  │    ├── /bsde_log           # Risultati CSV e grafici Log Contract
  │    └── /bsde_quadratic     # Risultati CSV e grafici Quadratic
  │
  └── main.py                  # Entry point unico per lanciare simulazioni rapide

## Requisiti

Il codice richiede un ambiente Python 3 con le seguenti librerie:
- numpy
- scipy
- matplotlib

## Guida all'Utilizzo

### 1. Esecuzione Rapida (Consigliata)
Utilizzare lo script `main.py` per lanciare simulazioni preconfigurate.

Esempio: Call Option (Default)
$ python main.py

Esempio: Contratto Logaritmico
$ python main.py --model log

Esempio: Payoff Quadratico
$ python main.py --model quad --steps 100 --paths 50000

Parametri opzionali:
--S0, --K, --r, --sigma, --T : Parametri finanziari
--paths, --steps, --knots    : Parametri di simulazione (Monte Carlo + Spline)

### 2. Esecuzione dei Test di Validazione
Ogni script nella cartella `test/` è autonomo e salva i risultati (CSV + PNG) nella cartella `result/`.

- Validazione Call Option:
  $ python test/test_BSDE.py

- Validazione Log Contract (Verifica stabilità su funzioni concave):
  $ python test/test_BSDE_log.py

- Validazione Quadratic (Verifica stabilità su forte convessità):
  $ python test/test_BSDE_quadratic.py

## Interpretazione dei Risultati

Il solver calcola:
1. Y_0: Il prezzo del derivato al tempo t=0.
2. Z_t: La strategia di copertura (hedging) lungo tutto il percorso.

Nei grafici generati (cartella `result/`):
- Il grafico a sinistra mostra la nuvola di punti (Scatter) dei valori Z_t stimati dal solver sovrapposta alla curva teorica (Linea Rossa). Una buona 
    sovrapposizione indica che il modello sta imparando correttamente la dinamica di hedging.
- Il grafico a destra (dove presente) o l'istogramma dei residui mostra la qualità dell'approssimazione numerica.

## Note Tecniche
L'uso delle B-Spline cubiche permette di ottenere una stima analitica della derivata spaziale (dY/dX) necessaria per calcolare Z_t, 
offrendo una stabilità superiore rispetto alle differenze finite o alla regressione polinomiale globale.
