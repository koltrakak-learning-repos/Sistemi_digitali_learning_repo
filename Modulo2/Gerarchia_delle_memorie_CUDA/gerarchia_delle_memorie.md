### Gerarchia di memoria in CUDA
Importante la distinzione tra memorie on e off chip. Chiaramente la prima è la più veloce ma più piccole (ad es. shared memory)

- Registri
    - Tipicamente 32-bit per registro
    - Memoria on-chip più veloce (massima larghezza di banda e minima latenza) sulla GPU (accesso ∼1 ciclo di clock)
    - Strettamente privati per thread (non condivisi) con durata limitata all'esecuzione del kernel.

- Memoria locale (nice name)
    - Memoria **off-chip** (DRAM). **Nome ambiguo**, fisicamente collocata nella stessa posizione della memoria globale.
    - Privata per thread
    - Utilizzata per variabili che non possono essere allocate nei registri a causa di limiti di spazio (array locali, grandi strutture).

- SMEM e chache L1
    - Ogni SM ha memoria on-chip limitata (es. 48-228 KB), condivisa tra shared memory e cache L1 (in alcune GPU, separate).
    - Partizionata fra tutti i thread block residenti in un SM.
    - Memoria è ad alta velocità, con elevata bandwidth e bassa latenza rispetto a memoria locale e globale.
    - Ottimizza la condivisione e comunicazione tra thread di un blocco. **Ciclo di vita legato al blocco di thread**, rilasciata al completamento del blocco.
    - Richiede sincronizzazione esplicita (__syncthreads) per prevenire data hazard

- Memoria costante
    - Spazio di memoria di sola lettura, off-chip (DRAM), accessibile a tutti i thread di un kernel.
    - Dichiarata con scope globale, visibile a tutti i kernel nella stessa unità di compilazione.
    - Inizializzata dall'host (readable and writable) e non modificabile dai kernel (read-only).

- Memoria texture (don't care)

- Memoria globale
    - Scope e lifetime globale (da qui global memory): Accessibile da ogni thread in ogni SM per tutta la durata dell'applicazione.
    - allocabile sia staticamente che dinamicamente
    - Memoria principale off-chip (DRAM) della GPU, accessibile tramite transazioni da 32, 64, o 128 **BYTE**.
    - Fattori chiave per l'efficienza:
        - Coalescenza: Raggruppare accessi di thread adiacenti a indirizzi contigui.
        - Allineamento: Indirizzi di memoria allineati a 32, 64, o 128 byte (per non fare più transazioni del necessario).
    - NB: Accessibile da tutti i thread di tutti i kernel, attenzione per la sincronizzazione.
        - Non c'è sincronizzazione fra blocchi, ogni blocco deve operare sui suoi dati

Bel riassunto in slide 43

**NB**: Molte delle memorie sono programmabili, sta quindi al programmatore cercare di usare le memorie (on chip) più vicine agli SM per avere la maggiore bandwith possibile.