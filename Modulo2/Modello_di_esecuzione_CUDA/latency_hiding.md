### DEFINIZIONI | Latency, Throughput e Concurrency
...

### Latency Hiding nelle GPU

Cosa è il Latency Hiding?

    Una tecnica che permette di mascherare i tempi di attesa dovuti ad operazioni ad alta latenza (come gli accessi alla memoria globale) attraverso l'esecuzione concorrente di più warp all’interno di un SM.

Si ottiene schedulando un warp diverso quando un warp va in stallo (es. accesso memoria), per massimizzare l'uso delle unità di calcolo di ogni SM. 

NB: i Warp Scheduler dell’SM selezionano costantemente (ad ogni ciclo di clock) i warp pronti all'esecuzione. __occorre che abbiano sempre warp eleggibili ad ogni ciclo__

Vantaggi del Latency Hiding
- Migliore Utilizzo delle Risorse: Le unità di elaborazione della GPU sono mantenute costantemente attive.
- Maggiore Throughput: Completamento di un maggior numero di operazioni nello stessa unità di tempo.
- Minore Latenza Effettiva: Minimizza l'impatto delle operazioni ad alta latenza.

### Esempio completo
![alt text](immagini/Esempio%20di%20Esecuzione%20di%20Blocchi%20e%20Warp%20su%20un%20SM.png)

NOTA: La gestione dei warp e dei warp scheduler è automatica (dettagli trasparenti al programmatore). Il programmatore deve solo __garantire un elevato numero di warp in esecuzione per massimizzare l'efficienza__.

### DOMANDA FONDAMENTALE

    Come stimare il numero di warp attivi necessari per mascherare la latenza?

#### Legge di Little

    Warp Richiesti = Latenza × Throughput

- Latenza: Tempo di completamento di un'istruzione (in cicli di clock).
- Throughput: Numero di warp (e, quindi, di operazioni) eseguiti per ciclo di clock.
- Warp Richiesti: Numero di warp attivi necessari per nascondere la latenza.

abbastanza intuitivo

esempi ...