## Che cos'è un modello di programmazione? 
Un insieme di regole e concetti per la strutturazione del codice. 

CUDA fornisce un'astrazione per programmare le GPU, offrendo
concetti come thread, blocchi e griglie.
    
Definisce la struttura e le regole per sviluppare applicazioni parallele su GPU. Elementi fondamentali:
- Gerarchia di Thread: Organizza l'esecuzione parallela in thread, blocchi e griglie, ottimizzando la scalabilità su diverse GPU.
- Gerarchia di Memoria: Offre tipi di memoria (globale, condivisa, locale, costante, texture) con diverse prestazioni e scopi, per ottimizzare l'accesso ai dati.
- API: Fornisce funzioni e librerie per gestire l'esecuzione del kernel, il trasferimento dei dati e altre operazioni essenziali.

### THREAD CUDA
Un thread CUDA rappresenta un'unità di esecuzione elementare nella GPU. Ogni thread CUDA esegue una porzione di un programma parallelo, chiamato __kernel__. Sebbene migliaia di thread vengano eseguiti concorrentemente sulla GPU, ogni singolo thread segue un percorso di esecuzione sequenziale all’interno del suo contesto.

Cosa Fa un Thread CUDA?
- Elaborazione di Dati: Ogni thread CUDA si occupa di un piccolo pezzo del problema complessivo, eseguendo calcoli su un sottoinsieme di dati.
- Esecuzione di Kernel: Ogni thread esegue lo stesso codice del kernel ma opera su dati diversi, determinati dai suoi identificatori univoci (threadIdx,blockIdx).
- Stato del Thread: Ogni thread ha il proprio stato, che include il program counter, i registri, la memoria locale e altre risorse specifiche del thread.

I thread CUDA non hanno bisogno di cambio di contesto, inoltre condividono memoria (vedroemo meglio in seguit).

### FLUSSO TIPICO DI ELABORAZIONE CUDA
1. Inizializzazione e Allocazione Memoria (Host)
    - Prepara dati e alloca memoria su CPU e GPU.
2. Trasferimento Dati (Host → Device)
    - Copia input dalla memoria CPU alla GPU.
3. Esecuzione del Kernel (Device)
    - GPU esegue calcoli paralleli.    
4. Recupero Risultati (Device → Host)
    - Copia output dalla memoria GPU alla CPU.
5. Post-elaborazione (Host)
    - Analizza o elabora ulteriormente i risultati sulla CPU.
6. Liberazione Risorse
    - Libera memoria allocata su CPU e GPU.

I passi 2-5 possono essere ripetuti più volte in un'applicazione complessa.