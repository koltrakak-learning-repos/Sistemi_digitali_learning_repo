### Veriﬁca dei Kernel CUDA 
è importante controllare che il risultato prodotto dall'esecuzione parallela del kernel sia corretto. Per fare ciò una strategia possibile è preparare una funzione che faccia un confronto elemento per elemento dei risultati prodotti dalla GPU con quelli prodotti da una esecuzione sequenziale dalla CPU (più facilmente verificabile in termini di correttezza) e verifichi che questi risultati siano analoghi.

Oppure, si può lanciare il kernel con un thread e un blocco in modo da emulare un'esecuzione sequenziale.

## Gestione degli Errori in CUDA
Il problema:
- __Asincronicità__: Molte chiamate CUDA sono asincrone, rendendo difficile __associare un errore alla specifica chiamata che lo ha causato__.
- __Complessità di Debugging__: Gli errori possono manifestarsi in punti del codice distanti da dove sono stati generati.

__Soluzione__:

    // Fornisce file, riga, codice e descrizione dell'errore.

    #define CHECK(call) {
        const cudaError_t error = call;

        if (error != cudaSuccess) {
            printf("Error: %s:%d, ", __FILE__, __LINE__);
            printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));
            exit(1);
        }
    }

Da usare per ogni chiamata a funzione CUDA.

## Profiling

### metodo timer CPU
Funzione per misurare il tempo corrente:

    #include <time.h>
        double cpuSecond() {
        struct timespec ts;
        timespec_get(&ts, TIME_UTC);

        return ((double)ts.tv_sec + (double)ts.tv_nsec * 1.e-9);
    }

Funzione del Timer della CPU
- La funzione utilizza timespec_get() per ottenere il tempo corrente del sistema.
    - il campo tv_sec sono i secondi in formato epoque di questo momento
    - il campo tv_nsec sono i nanosecondi rimanenti da tv_sec di questo momento
- Restituisce il tempo in secondi (double), combinando secondi e nanosecondi.
- La precisione è nell'ordine dei nanosecondi.

Utilizzo:

    double iStart = cpuSecond();    

    kernel_name <<<grid, block >>>(argument list);
    //NB: chiamate ai kernel sono asincrone -> fondamentale sincroinizzare
    cudaDeviceSynchronize(); 

    double iElaps = cpuSecond() - iStart; 

PRO:
- Facile da implementare e utilizzare.
- Efficace per kernel lunghi e misure approssimative.

CONTRO:
- Impreciso per kernel molto brevi (< 1 ms).
- Include overhead non relativo all'esecuzione del
kernel (es., lancio dei kernel, sincronizzazione, ...).
- Non fornisce dettagli sulle fasi interne del kernel.
- Precisione influenzata dal carico dell'__host__.

### metodo NVIDIA-PROFILER
Per schede Nvidia con con 5.0 <= Compute Capability < 8.0 -> __nvprof__. Uno strumento da riga di comando per raccogliere informazioni sull'attività di CPU e GPU dell'applicazione, inclusi __kernel__, __trasferimenti di memoria__ e __chiamate all'API CUDA__.

    $ nvprof [nvprof_args] <application> [application_args]

### metodo NVIDIA Nsight Compute
Strumento di profilazione e analisi approfondita per __singoli kernel CUDA__. Fornisce metriche dettagliate sulle prestazioni a livello di kernel.

Permette di:
- Analizzare l'utilizzo delle risorse GPU.
- Identificare colli di bottiglia nelle prestazioni dei kernel.
- Offre report dettagliati che possono essere utilizzati per ottimizzare il codice a livello di kernel.

Come si usa?

    $ ncu --set full <application> [application_args]

## Nvidia Nsight Systems vs. Compute
In Sintesi:
- Nsight Systems è uno strumento di analisi delle prestazioni a livello di sistema per identificare i colli di bottiglia delle prestazioni in __tutto il sistema__, inclusa la CPU, la GPU e altri componenti hardware.

- Nsight Compute è uno strumento di analisi e debug delle prestazioni a livello di kernel per ottimizzare le prestazioni e l'efficienza di __singoli kernel__ CUDA.

Scegliere lo Strumento Giusto:
- Nsight Systems: Perfetto per ottenere una panoramica delle prestazioni dell'applicazione nel suo complesso, identificare aree di interesse (CPU bound vs. GPU bound) e analizzare le interazioni tra CPU e GPU.

- Nsight Compute: Ideale per analisi approfondite di kernel specifici, ottimizzazione di codice CUDA e identificazione di colli di bottiglia a basso livello.

## Analisi del profiling nell'esercizio di somma degli array
- Gestione memoria domina:
    - Allocazione (cudaMalloc): 55%
    - Trasferimenti (cudaMemcpy - operazioni di memoria): 20%
- Esecuzione kernel GPU trascurabile: 0.0% (244,222μs)
- Operazioni ausiliarie minime:
    - cudaFree: 3%
    - cudaDeviceSynchronize: ~0%

    Conclusioni: Prestazioni limitate dalla gestione memoria, non dal calcolo GPU. Ottimizzazione dovrebbe concentrarsi su riduzione allocazioni e trasferimenti dati.

### Ottimizzazione della Gestione della Memoria in CUDA
Sfide:
- Trasferimenti lenti: I trasferimenti di dati tra host e device attraverso il bus PCIe rappresentano un collo di bottiglia.
- Allocazione sulla GPU: L'allocazione di memoria sulla GPU è un'operazione relativamente lenta.

Best Practices:
- Minimizzare i Trasferimenti di Memoria
    - I trasferimenti di dati tra host e device hanno un'__alta latenza__.
    - Raggruppare i dati in buffer più grandi per ridurre i trasferimenti e __sfruttare la larghezza di banda__.
- Allocazione e Deallocazione Efficiente
    - L'allocazione di memoria sulla GPU tramite cudaMalloc è un'operazione relativamente lenta.
    - Allocare la memoria una volta all'inizio dell'applicazione e riutilizzarla quando possibile.
    - Liberare la memoria con cudaFree quando non serve più, per evitare perdite e sprechi di risorse.
- Sfruttare la Shared Memory (lo vedremo)
    - La shared memory è una memoria on-chip a bassa latenza accessibile a tutti i thread di un blocco.
    - Utilizzare la shared memory per i dati frequentemente acceduti e condivisi tra i thread di un blocco per ridurre l'accesso alla memoria globale più lenta.
