    CUDA adotta una gerarchia a due livelli per organizzare i thread basata su blocchi di thread e griglie di blocchi.

### Struttura Gerarchica:
- Grid (griglia):   
    - composta da (array di) blocchi
    - rappresenta l'intera computazione di un kernel
        - Contiene tutti i thread che eseguono il singolo kernel
        - Ad ogni kernel è associata una e una sola griglia
    - Condivide lo stesso spazio di memoria globale.
    - È organizzata in una struttura 1D, 2D o 3D.
- Blocco:
    - Un thread block è un gruppo di thread eseguiti __logicamente in parallelo__.
    - Ha un ID univoco all'interno della sua griglia.
    - I thread di un blocco possono sincronizzarsi (non automaticamente) e condividere memoria.
        - Possibilità di sincronismo nell'accesso ai dati condivisi.
    - I thread di blocchi diversi __non possono cooperare__.          
        - i thread appartenenti allo stesso blocco possono comunicare tra di loro (memoria condivisa). Lo stesso non vale per i thread in blocchi diversi.
    - I blocchi sono organizzati in una struttura 1D, 2D o 3D.
- Thread:
    - Ha un proprio ID univoco all'interno del suo blocco.
    - Ha accesso alla propria memoria privata (registri).

### Perché una Gerarchia di Thread?
- Mappatura Intuitiva
    - La gerarchia di thread (grid, blocchi, thread) permette di scomporre problemi complessi in unità di lavoro parallele più piccole e gestibili, __rispecchiando spesso la struttura intrinseca del problema stesso__.
- Organizzazione e Ottimizzazione
    - Il programmatore può controllare la dimensione dei blocchi e della griglia per __adattare l'esecuzione alle caratteristiche specifiche dell'hardware e del problema__, ottimizzando l'utilizzo delle risorse.
- Efficienza nella Memoria
    - I thread in un blocco condividono dati tramite memoria on-chip veloce (es. shared memory), riducendo gli accessi alla memoria globale più lenta, migliorando dunque significativamente le prestazioni.
- Scalabilità e Portabilità
    - La gerarchia è scalabile e permette di adattare l'esecuzione a GPU con diverse capacità e numero di core. Il codice CUDA, quindi, risulta più portabile e può essere eseguito su diverse architetture GPU.
- Sincronizzazione Granulare
    - I thread possono essere sincronizzati solo all'interno del proprio blocco, evitando costose sincronizzazioni globali che possono creare colli di bottiglia.

## IDENTIFICAZIONE DEI THREAD
Ogni thread ha un'identità unica definita da coordinate specifiche all'interno della gerarchia grid-block. Queste coordinate sono essenziali per l'esecuzione dei kernel e __l'accesso corretto ai dati__.

Gli identificatori sono equivalenti a variabili built-in preinizializzate all'interno di un kernel a runtime. 

### Variabili di Identificazione (Coordinate)
1. __blockIdx__ (indice del blocco all'interno della griglia)
    - Componenti: blockIdx.x, blockIdx.y, blockIdx.z
2. __threadIdx__ (indice del thread all'interno del blocco)
    - Componenti: threadIdx.x,threadIdx.y,threadIdx.z

Entrambe sono variabili built-in di tipo __uint3__ pre-inizializzate dal CUDA Runtime e accessibili solo all'interno del kernel. __uint3__ è un built-in vector type di CUDA con tre campi (x,y,z) ognuno di tipo unsigned int.

### Variabili di Dimensioni
1. __blockDim__ (dimensione del blocco in termini di thread)
    - Tipo: dim3 (lato host), uint3 (lato device, built-in)
    - Componenti: blockDim.x,blockDim.y,blockDim.z
2. __gridDim__ (dimensione della griglia in termini di blocchi)
    - Tipo: dim3 (lato host), uint3 (lato device, built-in)
    - Componenti: gridDim.x,gridDim.y,gridDim.z

Identificare i thread permette di associare una specifica porzione dei dati ad un determinato thread. 

# Guarda esempi sulle slide per capire bene come funzionano queste variabili. 