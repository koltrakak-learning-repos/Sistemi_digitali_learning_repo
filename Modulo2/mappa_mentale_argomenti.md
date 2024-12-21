### Modello di programmazione CUDA
- Distinzione tra Host e Device
    - Chip distinti collegati da bus PCIe
    - Memorie fisiche e spazi di indirizzamento distinti
        - Sequenza tipica di operazioni riguardanti la gestione della memoria 
            1. Alloca memoria su device per input e output 
            2. trasferimento dati di input da host a device sulla memoria allocata
            3. Lancio di kernel
            4. trasferisco output sull'host
            5. elaboro output
        - Importante minimizzare trasferimenti tra host e device

- Struttura e regole per sviluppare applicazioni parallele su GPU
    - Gerarchia dei thread
        - blocchi e griglie multidimensionali
    - Gerarchia delle memorie
    - API CUDA

- Tecniche di mapping e dimensionamento
    - Mapping con indice globale
    - Dimensionamento dei blocchi
        - Almeno un po' di thread se no sottoutilizzo delle risorse quando un blocco viene schedulato su un SM
        - multipli di 32
    - Dimensionamento della griglia in maniera dinamica 
    
### Modello di esecuzione CUDA




### Modello di memoria CUDA
- Accesso alla Memoria
    - Istruzioni ed operazioni di memoria sono emesse ed eseguite per warp (32 thread).
    - Ogni thread fornisce un indirizzo di memoria quando deve leggere/scrivere la memoria globale, e la dimensione della richiesta del warp dipende dal tipo di dato.
        - es: 32 thread x 4 byte per int, 32 thread x 8 byte per double.
    - La richiesta (lettura o scrittura) è servita da una o più **transazioni di memoria**.
        - Una transazione è un'operazione atomica di lettura/scrittura tra la memoria globale e gli SM della GPU.
        - La memoria globale è accessibile in lettura e scrittura tramite transazioni di memoria da 32, 64 o 128 byte
    - Alla fine di una transazione di **lettura** dalla memoria globale, i dati richiesti vengono caricati nei registri della SMSP associata al warp che ha emesso la richiesta. In una transazione di scrittura invece dai registri si scrive in memoria globale
        - Bisogna anche tenere a mente che in mezzo a questo percorso le cache L2 e L1 dell'SM vengono caricate/sovrascritte per agevolare accessi (sia in lettura che in scrittura) successivi.
    - Tutti gli accessi alla memoria globale passano attraverso la cache L2
    - Molti accessi passano anche attraverso la cache L1, a seconda del tipo di accesso e dell'architettura GPU.
        - Fattori che influenzano il passaggio dei dati attraverso la Cache L1:
            - Compute Capability del device (Compute Capability ≥ 6.0: Cache L1 è abilitata di default.).
            - Opzioni del compilatore nvcc (Abilitazione: -Xptxas -dlcm=ca).

- Pattern di Accesso alla Memoria
    - Gli accessi possono essere classificati in pattern basati sulla **distribuzione degli indirizzi IN UN WARP**.
    - L'obiettivo è raggiungere le migliori prestazioni nelle operazioni di lettura e scrittura tramite accessi dei warp **allineati e coalescenti**

- Accessi Allineati alla Memoria
    - L'indirizzo iniziale di una transazione di memoria è un multiplo della dimensione della transazione stessa.
    - Gli accessi non allineati richiedono più transazioni, sprecando banda di memoria.

- Accessi Coalescenti alla Memoria
    - Si verificano quando tutti i 32 thread in un warp accedono a un blocco contiguo di memoria.
    - Se gli accessi sono contigui, l'hardware può combinarli in un **numero ridotto di transazioni** verso posizioni consecutive nella DRAM