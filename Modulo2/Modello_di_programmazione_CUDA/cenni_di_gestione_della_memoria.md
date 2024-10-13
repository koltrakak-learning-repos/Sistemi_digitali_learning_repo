## MODELLO DI MEMORIA CUDA
Il modello CUDA presuppone un sistema con un host e un device, ognuno con la __propria memoria__.

La comunicazione tra la memoria dell'host e quella del device avviene tramite il bus seriale PCIe (Peripheral Component Interconnect Express), che permette di __trasferire dati tra (le memorie di) CPU e GPU__.

I kernel CUDA operano sulla __memoria del device__.

CUDA Runtime fornisce funzioni per:
- Allocare memoria sul device.
- Rilasciare memoria sul device quando non più necessaria.
- Trasferire dati bidirezionalmente tra la memoria dell'host e quella del device.

Ad esempio:
- cudaMalloc: Alloca memoria sulla GPU.
- cudaMemcpy: Trasferisce dati tra host e device.
- cudaMemset: Inizializza la memoria del device.
- cudaFree: Libera la memoria allocata sul device.

__NB__: le operazioni CUDA C agiscono sulla memoria globale della GPU

__NB_2__: È responsabilità del programmatore gestire correttamente l'allocazione, il trasferimento e la deallocazione della memoria
per ottimizzare le prestazioni.

__NB_3__: non c'è un modo speciale per distinguere puntatori a memoria host rispetto a memoria device se non con nomi diversi.

### GERARCHIE DI MEMORIA
In CUDA, esistono diversi tipi di memoria, ciascuno con caratteristiche specifiche in termini di accesso, velocità, e visibilità. Per ora, ci concentriamo su due delle più importanti:
    
__Global memory__:   
- Accessibile da tutti i thread su tutti i blocchi
- Più grande ma più lenta rispetto alla shared memory
- Persiste per tutta la durata del programma CUDA
- È adatta per memorizzare dati grandi e persistenti

__Shared Memory__:
- Condivisa tra i thread all'interno di un singolo blocco
- Più veloce, ma limitata in dimensioni
- Esiste solo per la durata del blocco di thread
- Utilizzata per dati temporanei e intermedi

__NB__: le funzioni CUDA viste sopra, operano principalmente sulla GLOBAL Memory.

# FIRME DELLE VARIE FUNZIONI
- cudaMalloc()
    - cudaError_t cudaMalloc(void** devPtr, size_t size)
    - __devPtr__ è una puntatore passato per riferimento. Alla fine della funzione punterà alla locazione di memoria allocata sulla GPU.
        - Inizialmente è un normale puntatore di tipo host.
    - cudaError_t: codice di errore (cudaSuccess se l'allocazione ha successo).
- cudaMemcpy()
    - cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind)
    - dst: Puntatore alla memoria di destinazione.
    - src: Puntatore alla memoria sorgente.
        - const, non viene modificata la memoria puntata
    - count: Numero di byte da copiare.
    - kind: Direzione della copia.
        - cudaMemcpyHostToHost: Da host a host
        - cudaMemcpyHostToDevice: Da host a device
        - cudaMemcpyDeviceToHost: Da device a host
        - cudaMemcpyDeviceToDevice: Da device a device
    - Funzione __sincrona__: blocca l'host fino al completamento del trasferimento.
    - Per prestazioni ottimali, minimizzare i trasferimenti tra host e device.
- cudaMemset()
    - cudaError_t cudaMemset(void* devPtr, int value, size_t count)
        - value: Valore da impostare in ogni byte della memoria.
        - count: Numero di byte della memoria da impostare al valore specificato
    - Utilizzo: Comunemente utilizzata per azzerare la memoria (impostando value a 0).
    - Gestione: L'inizializzazione deve avvenire dopo l'allocazione della memoria tramite cudaMalloc.
    - Efficienza: È preferibile usare cudaMemset per grandi blocchi di memoria per ridurre l'overhead
- cudaFree()
    - cudaError_t cudaFree(void* devPtr)
    - devPtr: Puntatore alla memoria sul device che deve essere liberata. Questo puntatore deve essere stato
precedentemente restituito tramite la chiamata cudaMalloc.
    -  Gestione: È responsabilità del programmatore assicurarsi che ogni blocco di memoria allocato con
cudaMalloc sia liberato per evitare perdite di memoria (memory leaks) sulla GPU.
    - Efficienza: La deallocazione della memoria può avere un overhead significativo, pertanto è consigliato
minimizzare il numero di chiamate.