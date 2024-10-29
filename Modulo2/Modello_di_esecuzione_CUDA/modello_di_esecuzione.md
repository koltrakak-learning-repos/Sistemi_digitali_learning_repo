### SM, Thread Blocks e Risorse
Parallelismo Hardware
- Più SM per GPU permettono l'esecuzione simultanea di migliaia di thread (anche da kernel differenti).
Distribuzione dei Thread Blocks
- Quando un kernel viene lanciato, i blocchi di vengono automaticamente e dinamicamente distribuiti dal GigaThread Engine (scheduler globale) agli SM.
- Le variabili di identificazione e dimensione gridDim, blockIdx, blockDim, e threadIdx sono rese disponibili ad ogni thread e condivise nello stesso SM.
- Una volta assegnati a un SM, i thread di un blocco eseguono esclusivamente su quell'SM.
Gestione delle Risorse
- Più blocchi possono essere assegnati allo stesso SM contemporaneamente.
- Lo scheduling dei blocchi dipende dalla disponibilità delle risorse dell'SM (registri, memoria condivisa) e dai limiti architetturali di ciascun SM (max blocks, max threads, etc.).
Parallelismo Multi-Livello
- Parallelismo a Livello di Istruzione: Le istruzioni all'interno di un singolo thread sono eseguite in pipeline.
    - obiettivo: 1 istruzione per clock
- Parallelismo a Livello di Thread: Esecuzione concorrente di gruppi di thread (warps) sugli SM

### gigathread engine = scheduler globale
prende i blocchi generati dal lancio dei kernel e li schedula ai vari SM dinamicamente ed automaticamente. Applicando anche una politica di load balancing verso  vari SM.
- schedulazione "di primo livello": blocchi verso gli SM

__NB__: una volta che un blocco viene assegnato ad uno SM il blocco rimane li fino al suo completamento.

### due livelli di parallelismo
Ogni blocco schedulato dal giga thread engine viene elaborato parallelamente in maniera indipendente -> ordine di esecuzione dei blocchi non definibile

Una volta che gli SM esauriscono le risorse dopo avere accettato quanti più blocchi possibilie, l'esecuzione dei blocchi rimanenti è __sequenziale__. Ovvero, bisogna aspettare la terminazione dell'elaborazione di un blocco all'interno dell'SM per mettere in esecuzione uno in attesa.

__NB__: Non __c'è comunicazione tra thread di blocchi diversi__ anche se scheduati sullo stesso SM. La memoria condivisa è condivisa solo intra-block e non inter-block.

### SIMD vs SIMT

__SIMD__: È un modello di esecuzione parallela comunemente utilizzato nelle CPU dove una __singola istruzione opera simultaneamente su più elementi__ usando unità di elaborazione vettoriale.
- Utilizza registri vettoriali che possono contenere più elementi (es. 4 float, 8 int16, 16 byte).
- Il programma segue un flusso di controllo centralizzato -> __un unico thread__.
- Limitazioni:
    - Larghezza vettoriale fissa nell'hardware (es. AVX-512 consente 512 bit), limitando gli elementi per istruzione.
    - __Divergenza non è ammessa in SIMD__; se sono richieste condizioni (if/else), si usano maschere esplicite che indicano su quali elementi il calcolo deve essere eseguito

__SIMT__: Modello ibrido adottato in CUDA che __combina parallelismo a livello di più thread con esecuzione tipo SIMD__.
- Caratteristiche Chiave:
    - A differenza del SIMD, non ha un controllo centralizzato delle istruzioni. Ogni thread possiede un proprio Program Counter (PC), registri e stato indipendenti (maggiore flessibilità).
    - __Supporta divergenza__ del flusso di controllo (thread possono avere percorsi di esecuzione indipendenti).
- Implementazione
    - In CUDA, i thread sono organizzati in __gruppi di 32 chiamati warps__ (unità minima di esecuzione in un SM).
    - I thread in un warp iniziano insieme allo stesso indirizzo del programma (PC), ma possono divergere.
    - __Divergenza in un warp causa esecuzione seriale dei percorsi diversi__, riducendo l’efficienza (da evitare).
    - La divergenza è __gestita automaticamente dall'hardware__, ma con un impatto negativo sulle prestazioni.

La caratteristica fondamentale è che permette di scrivere degli if -> divergenza, gestita internamente dall'hardware. Tuttavia, questo ha un costo, il ramo if e il ramo else vengono eseguite sequenzialmente in questo ordine

### Warp: L'Unità Fondamentale di Esecuzione nelle SM
Come gia detto sopra, i warp sono raggruppamenti di 32 thread ottenuti suddivendendo un blocco. __Fisicamente__ i thread eseguiti parallelamente sono quelli di uno warp non quelli di un blocco. La vista __logica__ invece, è quella in cui sono i blocchi a venire gestiti direttamente.

    Lo SM gestisce warp e non blocchi.

Riassumendo abbiamo:
- Distribuzione dei Thread Block
    - Quando si lancia una griglia di thread block, questi vengono distribuiti tra i diversi SM disponibili.
- Partizionamento in Warp
    - I thread di un thread block vengono suddivisi in warp di 32 thread (con ID consecutivi).
- Esecuzione SIMT
    - I thread in un warp eseguono la stessa istruzione su dati diversi, con possibilità di divergenza.
- Esecuzione Logica vs Fisica
    - I singoli Thread vengono eseguiti in parallelo logicamente, ma non sempre fisicamente per motivi legati alla carenza di risorse oppure alla divergenza.
- Scheduling Dinamico (Warp Scheduler)
    - L'SM gestisce dinamicamente l'esecuzione di un numero limitato di warp, switchando efficientemente tra di essi. 
        - Scheduling dei warp.

__NB__: Quando la dimensione del blocco di thread non è un multiplo della dimensione del warp, i thread inutilizzati all'interno dell'ultimo warp vengono __disabilitati__, ma consumano comunque risorse all'interno dell'SM.
- Best practice → dimensione blocco multiplo di 32 (warp size) in modo da non incorrere mai in sprechi di risorse di questo tipo.
__NB_2__: I thread di un warp appartengono sempre allo stesso thread block e non vengono mai suddivisi tra diversi thread block.

Un warp viene assegnato a una __sub-partition__, solitamente in base al suo ID, dove rimane fino al completamento. Una sub-partition gestisce un “pool” di warp concorrenti di dimensione fissa (es., Turing 8 warp, Volta 16 warp).

Riassumendo:
- Sugli SM vengono schedulati più blocchi
- All'interno delle partizioni di questo SM vengono schedulati i vari warp appartenenti ai blocchi gestiti dallo SM. 

### Organizzazione dei thread e dei warp
- Punto di Vista Logico: Un blocco di thread è una collezione di thread organizzati in un layout 1D, 2D o 3D.
- Punto di Vista Hardware: Un blocco di thread è una collezione 1D di warp. I thread in un blocco sono organizzati in un layout 1D e ogni insieme di 32 thread consecutivi (con ID consecutivi) forma un warp.
    - Il runtime CUDA si occupa automaticamente di linearizzare gli indici multidimensionali, raggruppare i thread in warp e di gestire il mapping hardware con le risorse.

__OSS__: a che cosa serve identificare i thread di un warp? Ad associarli alle relative risorse hardware come i registri all'interno dell SM

### Contesto di esecuzione dei warp
- Il contesto di esecuzione locale di __un warp__ in un SM contiene:
    - i due qua sotto si distinguano nelle architetture >= di volta in quanto diventa contesto per thread e non più per warp
        - Program Counter (PC): Indica l’indirizzo della prossima istruzione da eseguire.
        - Call Stack: Struttura dati che memorizza le informazioni sulle chiamate di funzione, inclusi gli indirizzi di ritorno, gli argomenti, array e strutture dati più grandi.
    - Registri: Memoria veloce e privata per ogni thread, utilizzata per memorizzare variabili e dati temporanei.
    - Memoria Condivisa: Memoria veloce e condivisa tra i thread di un blocco, utile per comunicare e sincronizzarsi.
    - Thread Mask: Indica quali thread del warp sono attivi o inattivi durante l'esecuzione di un'istruzione.
    - Stato di Esecuzione: Informazioni sullo stato corrente del warp (es. in esecuzione/in stallo/eleggibile).
    - Warp ID: Identificatore che consente di distinguere i warp e calcolare l’offset nel register file per ogni thread nel warp.
- L'SM mantiene on-chip il contesto di ogni warp per tutta la sua durata, quindi __il cambio di contesto__ (da un warp all'altro) __è senza costo__.

OSS: Il warp scheduler effettua una specie di cambio di contesto nell'eseguire i vari warp da lui gestiti. Questo cambio di contesto è poi __senza costo__ in quanto lo stato dei thread viene salvato all'interno del SM. 

### Classificazioni
- Thread Block Attivo
    - Un thread block viene considerato attivo (o residente) __quando gli vengono allocate risorse di calcolo di un SM__ come registri e memoria condivisa (__non significa che tutti i suoi warp siano in esecuzione simultaneamente__).
    - I warp contenuti in un thread block attivo sono chiamati __warp attivi__.
    - Il numero di blocchi/warp attivi in ciascun istante è limitato dalle risorse dell'SM (vedi compute capability).
- Tipi di Warp Attivi
    - Warp Selezionato (Selected Warp): Un warp in esecuzione attiva su un'unità di elaborazione (FP32, INT32, Tensor Core, etc.).
    - Warp in Stallo (Stalled Warp): Un warp in attesa di dati o risorse, impossibilitato a proseguire l'esecuzione.
        - Cause comuni: latenza di memoria, dipendenze da istruzioni, sincronizzazioni.
    - Warp Eleggibile/Candidato (Eligible Warp): Un warp __pronto__ (ma ancora non scelto) per l'esecuzione
        - Condizioni per l'eleggibilità:
            - Disponibilità Risorse: I thread del warp devono essere allocabili sulle unità di esecuzione disponibili.
            - Prontezza Dati: Gli argomenti dell'istruzione corrente devono essere pronti (es. dati dalla memoria).
            - Nessuna Dipendenza Bloccante: Risolte tutte le dipendenze con le istruzioni precedenti
- Stati dei Thread
    - Un warp contiene sempre 32 thread, ma __non tutti potrebbero essere attivi__.
    - Thread Attivo (Active Thread)
        - Esegue l'istruzione corrente del warp.
        - Contribuisce attivamente all'esecuzione SIMT.
    - Thread Inattivo (Inactive Thread)
        - Divergenza: Ha seguito un percorso diverso nel warp per istruzioni di controllo flusso, come salti condizionali.
        - Terminazione: Ha completato la sua esecuzione prima di altri thread nel warp.
        - Padding: I thread di padding sono utilizzati in situazioni in cui il numero totale di thread nel blocco non è un multiplo di 32, per garantire che il warp sia completamente riempito.
    - Lo stato di ogni thread è tracciato attraverso una thread mask o maschera di attività (un registro hardware)



## Scheduling dei warp
è presente un warp-scheduler per ogni partizione di uno SM

l'obiettivo dello warp-scheduler è semplicemente di occupare il più possibile le risorse disponibili. Quindi il primo warp che trova disponibile lo schedula

la capacità di eseguire effettivamente nuove istruzioni (appartenenti ai warp all'interno di una partizione) nell'SM ad ogni ciclo di clock è limitata dalla quantità di warp-scheduler e unità di dispatch presenti.