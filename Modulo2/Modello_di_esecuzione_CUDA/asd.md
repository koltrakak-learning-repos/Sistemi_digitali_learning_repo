### che cos'è un modello di esecuzione e a che cosa serve?
...

### Streaming multiprocessor
Una GPU è composta da tanta roba, ad esempio:
- __memory controller__ -> controllore per gli accessi alla memoria, multipli per suddivisione dell'accesso alla memoria globale (del device)
- giga thread engine -> scheduler dei blocchi verso i multiprocessori della GPU
- streaming multiprocessor: Una GPU è composta da tanti SM (16 nell'esempio, al giorno d'oggi anche 144). Ogni SM si occuperà di lanciare i thread definiti nella griglia

UN SM è composto da tanta roba:
- CUDA CORES, unità di elaborazione composta da ALU e FPU (floating point unit, alu per float)
- Register file
- warp scheduler; 
    - decide come assegnare i thread ai cuda core
    - warp = gruppi di 32 thread con cui vengono suddivisi i blocchi

- dispatch unit     
    - simile alla control unit di una CPU, gestisce l'attivazione di ALU, FPU (ad es. se arriva una istruzione di somma la DU attiva tutte le unità per eseguire l'istruzione)
- load/store units

A quanto pare il concetto di CUDA core cambia, adesso per esempio fa riferimento alle unità di elaborazione per float.

