--- COME LA GRAFICA è DIVENTATA CALCOLO PARALLELO
...

PASSAGGIO DA GRAFICA 2D a GRAFICA 3D
...

Blocchi Fondamentali della Grafica 3D Moderna
• Triangolazione
    ○ Scomposizione di oggetti 3D in triangoli
    ○ Scopo: Semplificare la rappresentazione di forme complesse
• Rasterizzazione
    ○ Conversione di forme vettoriali in pixel
    ○ Scopo: Renderizzare oggetti 3D su uno schermo 2D
• Texture Mapping
    ○ Applicazione di immagini 2D su superfici 3D
    ○ Scopo: Aggiungere dettagli e realismo.
• Shading
    ○ Calcolo dell'illuminazione e del colore delle superfici
    ○ Scopo: Simulare l'interazione della luce con gli oggetti 3D

RAY TRACING:
tecnica di rendering per simulare il comportamento fisico della luce. Per ogni pixel si spara un raggio di luce. Il raggio di luce successivamente viaggia nello spazio 3D per e interagisce con gli oggetti al suo interno. Il ray tracing calcola
il colore di un pixel in base a tutti i raggi di luce (riflessi/rifratti) che colpiscono quel punto. Chiaramente una operazione molto complessa ma altamente parallelizzabile!
La GPU si presta bene.

--- GPU
- che cos'è?
    • Hardware progettato per l'elaborazione parallela, ottimizzato per gestire e accelerare il rendering grafico e le operazioni di calcolo intensivo.
- Funzioni Principali
    • Rendering grafico
        ○ Trasformazione di modelli 3D in immagini 2D
        ○ Calcolo di illuminazione, ombreggiatura e effetti visivi in tempo reale
    • Calcolo Parallelo Massivo
        ○ Esecuzione simultanea di migliaia di operazioni
        ○ Ottimizzazione per task che richiedono calcoli ripetitivi su grandi set di dati
GP-GPU
General Purpose computing on Graphics Processing Unit. Utilizzo delle GPU per calcoli NON grafici. Le GPU, nate per l'accelerazione grafica, mostravano un potenziale non
sfruttato per altri tipi di calcolo. Intuizione chiave: La struttura parallela delle GPU poteva essere applicata a problemi al di fuori del dominio grafico (applicazioni-general purpose), come il
rendering scientifico o simulazioni fisiche.

GPU DEDICATA vs GPU INTEGRATA e SOC
    -> dedicata     =   chip separato                   -> migliori prestazioni, memoria (VRAM) separata, più consumi
    -> integrata    =   integrata all'interno della CPU -> prestazioni moderate, memoria condivisa con il sistema, meno consumi    
Quando una GPU è dedicata ma è presente insieme al resto in un unico chip si parla di System On a Chip. Tipico nelle console.

--- DIFFERENZE TRA CPU e GPU 
UTILIZZO
    CPU (Central Processing Unit)
        • Esegue il sistema operativo e la maggior parte dei programmi generali (Navigazione web, multitasking quotidiano, etc)
        • Adatta per compiti che richiedono calcoli complessi e operazioni logiche / di controllo.
            ->  Operazioni di controllo intese come if / while / input e output, ecc...

    GPU (Graphics Processing Unit)
        • Inizialmente progettata per il rendering grafico (videogiochi, modellazione 3D, etc).
        • Ora utilizzata anche per applicazioni di calcolo parallelo come l'apprendimento automatico, la simulazione scientifica, e la crittografia.
            -> Adatta per calcoli semplici ma massivi
ARCHITETTURA
    CPU 
        • Core: Da 4-16 core per consumer CPU, fino a 128 core per CPU server di fascia alta.
        • Frequenza: Tipicamente tra 2.5 GHz e 5 GHz.
        • Gestione delle Istruzioni: Predizione delle diramazioni (branch prediction), esecuzione fuori ordine (out-of-order excecution), lunga pipeline di elaborazione
    GPU 
        • Core: Da centinaia a migliaia di core (es. Nvidia RTX 4090 ha 16384 CUDA cores)
        • Frequenza: Tipicamente tra 1 GHz e 2 GHz.
        • Architettura parallela: Design SIMT (Single Instruction, Multiple Threads) per NVIDIA GPUs.
MEMORIA
    CPU
        • Tipo di memoria: DDR4 o DDR5.
            -> RAM DDR (Double data rate) = operazioni sia sul fronte di salita che di discesa del clock.
        • Larghezza di banda: Fino a 100 GB/s con DDR5.
        • Cache: Cache di grandi dimensioni per core (fino a 72 MB di L3 cache).
        • Accesso alla memoria: Accesso diretto e condiviso con il sistema operativo e altre applicazioni.
    GPU
        • Tipo di memoria: GDDR6, GDDR6X, HBM2.
        • Larghezza di banda: Fino a 1000 GB/s con HBM2.
        • Memoria dedicata: Memoria dedicata solo alla GPU, separata dalla RAM del sistema.
        • Dimensione: 4-24 GB per GPU di consumo, fino a 48 GB o più per GPU professionali.
PRESTAZIONI
    CPU
        • Maggiore velocità per singolo thread (bassa latenza).
        • Eccellente per carichi di lavoro sequenziali
        • Migliore gestione di flussi di controllo complessi e operazioni logiche.
        • Bassa latenza per operazioni singole e accesso alla memoria.
    GPU
        • Maggiore capacità di calcolo parallelo massivo.
        • Eccellente per carichi di lavoro altamente parallelizzabili (es. rendering grafico, machine learning).
        • Throughput superiore per operazioni numeriche semplici e ripetitive.
SVANTAGGI
    CPU
        • Parallelismo limitato: Meno efficiente per operazioni altamente parallele.
        • Scalabilità: Difficoltà nell'aumentare il numero di core (vincoli termici e di consumo energetico).
        • Costo/prestazione: Costose per ottenere alte prestazioni in task paralleli.
    GPU
        • Versatilità limitata: Non adatta a tutti i tipi di carichi di lavoro.
        • Latenza: Più alta per operazioni singole rispetto alla CPU.
        • Complessità di programmazione: Richiede competenze specifiche per l'ottimizzazione.
        • Consumo energetico: Elevato sotto carichi intensivi.

--- ARCHITETTURE ETEROGENEE
Cosa è una Architettura Eterogenea? Un'architettura eterogenea è una struttura di sistema che integra diversi tipi di processori all'interno dello stesso computer.
    • Ruoli
        ○ CPU (Host): gestisce l'ambiente, il codice e i dati                               -> riduce la latenza
        ○ GPU (Device): co-processore, accelera calcoli intensivi (hardware accelerator)    -> massimizza il throughput
    • Connessione: GPU collegate alla CPU tramite bus PCI-Express
    • Struttura del Software: Applicazioni divise in codice host (CPU) e device (GPU)

NOMENCLATURA
Host    -> CPU
Device  -> GPU

Approccio ottimale:
    • Combinare CPU e GPU per massimizzare le prestazioni
    • Basso parallelismo + Dati limitati → CPU
    • Alto parallelismo + Dati massicci → GPU

STRUTTURA DEL SOFTWARE
Codice host (CPU):
    • Gestisce il flusso generale del programma
    • Si occupa di operazioni di controllo e I/O
    • Prepara e invia dati e istruzioni alla GPU
Codice device (GPU):
    • utilizzato per sezioni "compute intensive"
    • Contiene funzioni specializzate chiamate "kernel"
        -> NB: Chiamate Kernel, non centra nulla con il kernel del sistema operativo, è semplicemente una funzione eseguita dalla GPU.
    • Esegue calcoli intensivi in parallelo
    • Elabora grandi quantità di dati simultaneamente
    • Al termine della sezione compute intensive, i risultati vengono restituiti alla CPU per ulteriori elaborazioni, ritornando ad una esecuzione sequenziale

DOMANDA FONDAMENTALE
Come possiamo supportare l'esecuzione congiunta di CPU e GPU in un'applicazione? 
    -> CUDA!

--- CUDA
Cos'è CUDA?
    • Piattaforma di calcolo parallelo general-purpose ideato da NVIDIA (lanciata nel 2007)
    • Modello di programmazione per GPU NVIDIA
Obiettivo
    • Sfruttare la potenza di calcolo parallelo delle GPU
    • Semplificare lo sviluppo per sistemi CPU-GPU
CUDA come Ecosistema
    • Compilatore (nvcc), Profiler/Debugger (Nsight), Librerie Ottimizzate, Strumenti di Sviluppo (SDK, CUDA Toolkit)
Come si accede a CUDA?
    • API (CUDA Runtime API, CUDA Driver API)
    • Estensioni a C, C++, Fortran
    • Integrazione con framework di alto livello (TensorFlow, PyTorch)
Vantaggio chiave
    • Accesso alla GPU per calcoli generali, non solo grafica

--- ANATOMIA DI UN PROGRAMMA CUDA
• Struttura del Codice Sorgente
    - unico file sorgente in cui sono presenti: sia codice host, sia codice device (estensione .cu)
• Componenti Principali
    - Codice Host
        ○ Codice C/C++ eseguito sulla CPU.
        ○ Gestisce la logica dell'applicazione
        ○ Alloca memoria sulla GPU
        ○ Trasferisce dati tra CPU e GPU
        ○ Lancia i kernel GPU
        ○ Gestisce la sincronizzazione
    - Codice Device:
        ○ Codice CUDA C eseguito sulla GPU.
        ○ Contiene i kernel (funzioni parallele)
        ○ Esegue operazioni computazionali intensive in parallelo

PRIME PARTICOLARITà
    - __global__: Qualificatore CUDA per funzioni eseguite sulla GPU e chiamate dalla CPU.
    - Lancio dei kernel con <<<1, 10>>> prima delle tonde: Configurazione di esecuzione (1 blocco, 10 thread). Avvia 10 istanze parallele del kernel sulla GPU
    - cudaDeviceSynchronize(): metodo builtin, attende il completamento di tutte le operazioni GPU
        -> non l'unico builtin presente 

COSA SIGNIFICA PROGRAMMARE IN CUDA C?
Pensare in Parallelo
    • Decomposizione del Problema: Identificare le parti del problema che possono essere eseguite in parallelo per sfruttare al meglio le risorse della GPU.
    • Architettura della GPU: Le GPU sono composte da migliaia di core in grado di eseguire thread in parallelo. CUDA fornisce gli strumenti per organizzare e gestire 
      questi thread.
    • Scalabilità: Progettare algoritmi che si adattano a diversi numeri di thread (e GPU).
    • Gerarchia di Thread: Organizzare il lavoro in blocchi e griglie per massimizzare l'efficienza.
    • Gerarchia di Memoria: Utilizzare strategicamente memoria globale, condivisa e locale per ridurre i tempi di accesso.
    • Sincronizzazione: Gestire la coordinazione tra thread e il trasferimento dati tra CPU e GPU senza conflitti.
    • Bilanciamento del Carico: Distribuire il lavoro in modo uniforme per evitare colli di bottiglia.
Scrittura di codice in CUDA C
    • Si scrive codice sequenziale in C
    • Si estende a migliaia di thread, permettendo di pensare in termini sequenziali mentre si sfrutta il calcolo parallelo della GPU.
