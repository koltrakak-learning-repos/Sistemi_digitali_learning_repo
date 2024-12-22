- Una GPU permette l'esecuzione simultanea anche di più kernel.

- Più blocchi possono essere assegnati (scheduling) ed eseguiti sullo stesso SM contemporaneamente. L'assegnazione dei blocchi, però, dipende dalla disponibilità delle risorse dell'SM (registri, memoria condivisa) e dai limiti architetturali di ciascun SM (max blocks, max threads, etc.)
    - Se un SM è pieno su di esso non vengono schedulati nuovi blocchi finchè non si libera
    - Ad es. 

- I blocchi di un kernel vengono distribuiti tra i vari SM di una GPU dal gigathread engine (scheduler globale)
    - Una volta assegnati a un SM, i thread di un blocco eseguono esclusivamente su quell'SM. Un blocco e i relativi thread non cambiano mai SM durante la loro esecuzione. 

- La sincronizzazione è possibile all'interno di un thread block, ma non tra thread block diversi.
    - la funzione _syncthreads()_ sincronizza i thread di un blocco nel punto in cui quella funzione è stata chiamata

- Il modello SIMT può essere visto come una evoluzione di SIMD in quanto, se non c'è divergenza, i thread di un warp eseguono tutti la stessa istruzione su dati diversi, analogamente a SIMD. L'evoluzione sta nell'ammissibilità della divergenza a livello di warp; thread di uno stesso warp possono eseguire istruzioni appartenenti a percorsi diversi. 
    - A livello di blocco invece è normale che warp diversi eseguano ognuno una istruzione diversa (ogni warp ha il proprio PC)
        - Si parla di warp divergence, la divergenza occorre solo a livello di warp

- **Warps** (unità minima di esecuzione in un SM): 32 thread che eseguono la stessa istruzione all'interno di una SMSP
    - Un warp viene assegnato a una sub-partition, solitamente in base al suo ID, dove rimane fino al completamento.
    - Una sub-partition gestisce più warp, anche appartenenti a blocchi diversi, in maniera concorrente con un limite fisso (es., Turing 8 warp, Volta 16 warp).

- Siccome i blocchi vengono assegnati ad un SM e li rimangono fino alla loro terminazione, lo stesso vale anche per i relativi warp del blocco schedulato. Questo significa che una volta che il gigathread engine a schedulato un blocco su un determinato SM, quel SM si dovrà occupare di schedulare i warp gestendo le risorse di calcolo e stati dei warp.

- In conclusione un singolo SM in base alla compute capability ed alla generazione di GPU, ha dei limiti architetturali sulla quantità massima di risorse che riesce a gestire in maniera concorrente:
    - Max Blocchi per SM (32)
    - Max Warp per SM (64)
        - In particolare un warp viene assegnato a una sub-partition, dove rimane fino al completamento. Una sub-partition gestisce un “pool” di warp concorrenti di dimensione fissa (es., Turing 8 warp, Volta 16 warp).
        - il warp scheduler del SMSP schedula uno di questi warp
    - Max Threads per SM (2048)

- **Warp: Contesto di Esecuzione**
    - Il contesto di esecuzione locale di un warp in un SM contiene:
        - Program Counter (PC): Indica l’indirizzo della prossima istruzione da eseguire.
        - Call Stack: Struttura dati che memorizza le informazioni sulle chiamate di funzione, inclusi gli indirizzi di ritorno, gli argomenti, array e strutture dati più grandi.
        - Registri: Memoria veloce e privata per ogni thread, utilizzata per memorizzare variabili e dati temporanei.
        - Memoria Condivisa: Memoria veloce e condivisa tra i thread di un blocco utile per comunicare.
        - Thread Mask: Indica quali thread del warp sono attivi o inattivi durante l'esecuzione di un'istruzione.
        - Stato di Esecuzione: Informazioni sullo stato corrente del warp (es. in esecuzione/in stallo/eleggibile).
        - Warp ID: Identificatore che consente di distinguere i warp e calcolare l’offset nel register file per ogni thread nel warp.
    - L'SM mantiene on-chip il contesto di ogni warp per tutta la sua durata, quindi il cambio di contesto (scheduling ed esecuzione di un altro warp) è senza costo.

- I thread di un warp hanno una memoria privata (registri nel SMSP) e una memoria condivisa (shared memory del SMSP condivisa con la L1 cache)

- **Classificazione blocchi warp e thread**
    - Un blocco viene considerato attivo quando gli vengono allocate risorse di calcolo di un SM come registri e memoria condivisa
        - NON significa che tutti i suoi warp siano in esecuzione simultaneamente sulle unità di elaborazione.
        - I warp contenuti in un thread block attivo sono chiamati warp attivi.
        - Il numero di blocchi/warp attivi in ciascun istante è limitato dalle risorse dell'SM

    - Esistono varie tipologie di warp attivo (che ricorda, non vuol dire warp in esecuzione)
        - Warp Selezionato: Un warp in esecuzione attiva su un'unità di elaborazione (FP32, INT32, Tensor Core, etc.).
        - Warp in Stallo: Un warp in attesa di dati o risorse, impossibilitato a proseguire l'esecuzione.
            - Cause comuni:
                - latenza di memoria
                - dipendenze da istruzioni
                - sincronizzazioni.
                - carenza di risorse
        - Warp Eleggibile: Un warp pronto (ma ancora non scelto) per l'esecuzione, con tutte le risorse necessarie disponibili.
            - Condizioni per l'eleggibilità:
                - Disponibilità Risorse: I thread del warp devono essere allocabili sulle unità di esecuzione disponibili (unità di elaborazione, registri e memoria condivisa libera).
                - Prontezza Dati: Gli argomenti dell'istruzione corrente devono essere pronti (es. dati dalla memoria).
                - Nessuna Dipendenza Bloccante: Risolte tutte le dipendenze con le istruzioni precedenti.

    - I thread di un warp attivo si classificano in
        - thread attivi: in esecuzione
        - thread inattivi: disabilitati da una maschera calcolata in HW

- **Warp scheduling**
    - Funzionamento generale dello scheduling dei warp:
        - Processo di Schedulazione: I warp scheduler all'interno di un SM selezionano i warp eleggibili ad ogni ciclo di clock e li inviano alle dispatch unit, responsabili dell’assegnazione effettiva alle unità di esecuzione.
        - Gestione degli Stalli: Se un warp è in stallo, il warp scheduler seleziona un altro warp eleggibile per l'esecuzione, garantendo consentendo l'esecuzione continua e l'uso ottimale delle risorse di calcolo
            - concorrenza e latency hiding
        - Limiti architettonici:
            - Il numero di warp attivi su un SM è limitato dalle risorse di calcolo. (Esempio: 64 warp concorrenti su un SM Kepler).
            - Il numero di warp selezionati ad ogni ciclo è limitato dal numero di warp scheduler del SM. (Esempio: 4 su un SM Kepler)

- **Latency Hiding**
    - Quando un warp è in stallo (es. accesso memoria), l'SM passa immediatamente all'esecuzione di altri warp pronti (cambio di contesto senza costo).
    - I Warp Scheduler dell’SM selezionano costantemente (ad ogni ciclo di clock) i warp pronti all'esecuzione
    - **occorre che abbiano sempre warp eleggibili ad ogni ciclo**.

- **Legge di Little**
    - ci aiuta a calcolare quanti warp (approssimativamente) devono essere in esecuzione concorrente per ottimizzare il latency hiding e mantenere le unità di elaborazione della GPU occupate.
    - **Warp Richiesti = Latenza × Throughput**


- **Warp Divergence**
    - esecuzione sequenziale da parte dei thread di un warp dei due blocchi dell'if
    - riduce il parallelismo e l'utilizzo delle unità di calcolo all'interno del SM (parallelismo ridotto al massimo di 32 volte)
    - I passi condizionali in **differenti warp non causano divergenza**



- **Partizionamento delle Risorse nell'SM**
    - Ogni SM ha una quantità limitata di registri e memoria condivisa:
        - Register File: Un insieme di registri a 32 bit, partizionati tra i **thread attivi**.
        - SMEM: Una quantità fissa di memoria condivisa, partizionata tra i **blocchi di thread attivi**.
    - Il numero di thread block e warp che possono risiedere simultaneamente su un SM dipende dalla:
        - Disponibilità di Risorse: Quantità di registri e memoria condivisa disponibili sull'SM.
        - Richiesta del Kernel: Quantità di registri e memoria condivisa richiesti dal kernel per l'esecuzione.
        - Un maggiore utilizzo di registri per thread riduce il numero di **warp** che possono essere allocati su un SM.
        - Un maggiore utilizzo di memoria condivisa per thread block riduce il numero di **blocchi** che possono essere allocati su un SM.
    - è bene ridurre al massimo il numero di registri e la quantità di SMEM utilizzata all'interno del kernel
    - Una terza risorsa limitata all'interno di un SM è la quantità di thread allocabili. Bisogna anche dimensionare la dimensione dei blocchi quindi  

- Che cos'è l'occupancy?
    - è il rapporto tra i warp attivi e il numero massimo di warp supportati per SM (vedi compute capability)
    - rappresenta il grado di utilizzo delle risorse di calcolo dell'SM
        - **Occupancy Ottimale**: Quando raggiunge un livello sufficiente per **nascondere la latenza**. Un ulteriore aumento potrebbe degradare le prestazioni a causa della riduzione delle risorse disponibili per thread.
        - **Occupancy Bassa**: Risulta in una scarsa efficienza nell'emissione delle istruzioni, poiché non ci sono abbastanza warp eleggibili per nascondere la latenza tra istruzioni dipendenti.

- Linee Guida per le Dimensioni di Griglia e Blocchi
    - Mantenere il numero di thread per block multiplo della dimensione del warp (32).
    - Evitare dimensioni di block piccole: Iniziare con almeno 128 o 256 thread per block.
    - Regolare la dimensione del blocco in base ai requisiti di risorse del kernel.
    - Mantenere il numero di blocchi molto maggiore del numero di SM per esporre sufficiente parallelismo al dispositivo (latency hiding).
    - trial & error