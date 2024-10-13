Quest'anno più focus su tecniche, strategie e architetture SOFTWARE per velocizzare i calcoli. Meno hardware.

# CALCOLO PARALLELO
Obiettivo: accelerare l'elaborazione delle informazioni. Eseguire dei calcoli nel minor tempo possibile, in più, col dispendio di energia minore possibile.
    -> una strategia possibile: parallelismo
    
    In poche parole, c'è esigenza di velocizzare i calcoli.

---

Le CPU vanno bene in ogni situazione, ma questo non significa che esse siano l'architettura ideale per OGNI problema. Architetture __specializzate__ verso il problema, lo risolvono meglio.

Una GPU è un esempio di questo tipo. Ad esempio: ogni pixel di uno schermo non ha dipendenze verso gli altri pixel 
    -> calcolo parallelo -> faccio il render tutto insieme dello schermo piuttosto che fare il render sequenziale a partire dal pixel in alto a sinistra 

Altro esempio di applicazione del calcolo parallelo è quello del training e dell'inferenza di un intelligenza artificiale. 

    DEF Inferenza: quando viene utilizzata l'intelligenza artificiale; faccio la domanda e mi aspetto la risposta. 

Non c'è particolare differenza tra le operazioni effettive (aritmetica binaria, operazioni logiche, ...) che una CPU riesce a fare rispetto ad una GPU. Semplicemente la differenza tra le due architetture sta nell'efficenza con cui si riesce a risolvere un problema.

---
Per andare più veloce si può sicuramente incrementare la frequenza di clock. Tuttavia questa non è una soluzione scalabile in termini di potenza. 

    POTENZA proporzionale con (circa) CUBO DELLA FREQUENZA  

Inoltre c'è un limite fisico della velocità della luce, una frequenza troppo alta non lascia propagare i segnali.

    A 4GHz un segnale percorre circa 7,5cm

La soluzione al giorno d'oggi per andare più forte è quello di incrementare il numero di CPU (core), in questo modo il consumo di potenza non esplode.
    
    POTENZA in un sistema multi-core cresce linearmente con il numero di core

__Conclusione: aumentare la frequenza non è più fattibile__

I core tuttavia vanno sfruttati! Non è banale "spalmare" un task su più core (dipendenze tra dati).

## OBIETTIVI DEGLI AVANZAMENTI ARCHITETTURALI NEI COMPUTER
- __Ridurre la Latenza__: tempo necessario per completare un'operazione (tipicamente espresso in nanosecondi o microsecondi). La Latenza è il tempo necessario per avviare e completare un'operazione (È il tempo che l'acqua impiega a percorrere l'intero tubo)
- __Aumentare la Banda (Bandwidth)__: quantità massima di dati che possono essere trasferiti per unità di tempo (espressa in MB/s, GB/s o Tbps). La Bandwidth misura la capacità di trasferire dati (Un tubo più grande che può far passare più acqua)
- __Aumentare il Throughput__: Numero di operazioni completate per unità di tempo (espresso in MFLOPS o GFLOPS). Il Throughput misura il numero di operazioni processate per unità di tempo (Il volume d'acqua che scorre nel tubo in un certo periodo)

Dei colli di bottiglia non ovvi sono le velocità con cui si riesce a trasferire/recuperare i dati da fare processare (Un processore velocissimo è inutile se non si riesce ad alimentarlo sufficentemente).

---
COMPUTAZIONE PARALLELA
Paradigma per elaborare i dati in cui più operazioni vengono eseguite simultaneamente. 

Sfide:
- dipendenze tra i dati
- mantenere l'ordine delle operazioni

LEGGE DI AMDAHL
    non mi interessa la formula...

La Legge di Amdahl è un principio fondamentale nel calcolo parallelo che descrive il limite delle prestazioni ottenibili quando si parallelizza una parte di un programma.

    SPEEDUP = tempo per codice base / tempo per codice ottimizzato = T_1/T_p; con p numero di processori paralleli.

__Conclusione__: prima di andare a cercare di parallelizzare del codice, è opportuno fare un profiling per capire quanto di quel codice è parallelizzabile in primo luogo.

PROGRAMMAZIONE PARALLELA
- I calcoli vengono suddivisi in TAKS che possono essere eseguiti contemporaneamente.
- I TASK indipendenti, senza dipendenze di dati, offrono il maggiore potenziale di parallelismo.
- I programmi paralleli possono, e spesso lo fanno, contenere anche parti sequenziali.

### DIPENDENZE TRA DATI
- Vincolo: ottenere gli stessi risultati della versione sequenziale.
- Una dipendenza di dati si verifica quando un'istruzione richiede i dati prodotti da un'istruzione precedente.
- Le dipendenze limitano il parallelismo, poiché impongono un ordine di esecuzione.
- L'analisi delle dipendenze è cruciale per implementare algoritmi paralleli efficienti

__OSS__: PARALLELISMO Simile a ESECUZIONE FUORI ORDINE.
Oltre al pipelining, nei processori vi è un altra tecnica di accelerazione, l'esecuzione fuori ordine. In pratica, vengono decodificate molte istruzioni alla volta e quest' ultime vengono distribuite sulle unità di elaborazioni disponibili. I risultati vengono poi recuperati e riordinati. Il risultato è che le istruzione specificate nel programma vengono eseguire fuori ordine e parallelamente ma il risultato è lo stesso, ottenuto però più velocemente.

## SISTEMI MULTI-CORE E MANY-CORE
Più processori (core) per fare calcoli al posto di incrementare la frequenza.  Ogni core ha la propria memoria (cache di livello alto), i core poi possiedono anche una  memoria condivisa(cache di basso livello). Se si ha un cache miss si va alla ram attraverso la bus interface.

### GPU
Le GPU hanno un puttanaio di core. Sistema MANY-core. Ogni GPU ha decine di migliaia di core, più semplici però rispetto ai core di una CPU.

#### DIFFERENZE TRA CORE CPU e GPU:
Nonostante i termini "multicore" e "many-core" siano usati per etichettare le architetture CPU e GPU, un core CPU è molto diverso da un core GPU!

#### Core CPU
- Unità di controllo complessa per gestire flussi di istruzioni variabili
- Ampia cache per ridurre la latenza di accesso alla memoria
- Unità di predizione delle diramazioni sofisticate (molti salti)
- Esecuzione fuori ordine per ottimizzare l'utilizzo delle risorse
#### Core GPU
- Unità di controllo semplificata per gestire operazioni ripetitive
- Cache più piccola, compensata da alta larghezza di banda di memoria
- Minor enfasi sulla predizione delle diramazioni (pochi salti)
- Esecuzione in-order per massimizzare il throughput

### ARCHITETTURE ETEROGENEE
Un'architettura eterogenea è una struttura di sistema che integra diversi tipi di processori o core di elaborazione all'interno dello stesso computer o dispositivo.
- Ruoli
    - CPU (Host): gestisce l'ambiente, il codice e i dati
    - GPU (Device): co-processore, accelera calcoli intensivi (hardware accelerator)
    - ... esistono anche altri tipi di acceleratori, per esempio per AI, oppure nei telefoni per fotografia.
- Connessione: GPU collegate alla CPU tramite bus PCI-Express
- Struttura del Software: Applicazioni divise in codice host (CPU) e device (GPU)

Perchè le Architetture Eterogenee?
- CPU Multi-Core: 
    - Ottimizzato per Latenza
    - Eccellente per task sequenziali
    - Meno efficiente per parallelismo massiccio
- GPU Many-Core:
    - Ottimizzato per Throughput
    - Eccellente per parallelismo massiccio
    - Meno efficiente per task sequenziali

Una architettura Eterogenea (CPU+GPU) 
- best of both worlds
- Ottimizzazione flessibile
- Migliori prestazioni complessive

### APPROCCI AL PARALLELISMO
- Parallelismo a Livello di Dati (Data-Level Parallelism, DLP): Esegue la stessa operazione su più elementi di dati contemporaneamente
    - SIMD: Esecuzione della stessa istruzione su set diversi di dati.
    - Vettorizzazione: Ottimizza le operazioni su array di dati per l'esecuzione parallela.
    - GPU Computing: Utilizza le GPU per elaborare grandi dataset in parallelo (es. CUDA, OpenCL).
- Parallelismo a Livello di Istruzione (Instruction-Level Parallelism, ILP): Parallelizzazione delle istruzioni all'interno di un singolo thread.
    - Pipelining: Sovrapposizione di fasi di esecuzione di istruzioni diverse.
    - Superscalarità: Esecuzione simultanea di più istruzioni indipendenti nello stesso ciclo di clock.
        - Più pipeline, n instruction fetch contemporaneamente, n decode ..., ecc.
    - Out-of-Order Execution: Il processore esegue le istruzioni indipendenti prima, ottimizzando le risorse.
        - micro-op eseguite non nel loro ordine naturale ma appena è disponibile una risorsa, per poi riordinare
- Parallelismo a Livello di Thread (Thread-Level Parallelism, TLP): Esecuzione simultanea di più thread.
    - Multithreading: Esegue più thread concorrentemente su uno o più core del processore.
        - distribuisco i thread su piu core
    - Multiprocessing: Distribuisce l'elaborazione su più processori o core fisici.
    - Hyper-Threading: Simula core aggiuntivi per eseguire più thread su un singolo core fisico.
        - cambio di contesto a livello di core e non di processo