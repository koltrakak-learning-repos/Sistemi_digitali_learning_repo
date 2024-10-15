## CHE COS'é UN KERNEL CUDA?
- Un kernel CUDA è una funzione che viene eseguita in parallelo sulla GPU da migliaia o milioni di thread.
- Nei kernel viene definita la logica di calcolo per un singolo thread e l'accesso ai dati associati a quel thread.
- Ogni thread esegue lo stesso codice kernel, ma opera su diversi elementi dei dati

### Sintassi della chiamata Kernel CUDA
kernel_name <<<gridSize,blockSize>>>( argument list );

Con gridSize e blockSize si definisce:
- Numero totale di thread per un kernel.
- Il layout dei thread che si vuole utilizzare.

### Qualificatori
un kernel ha un prefisso che specifica dove verrà eseguito il kernel:
- __\_\_global\_\___: eseguito sul Device, chiamato dall’Host
- __\_\_device\_\___: eseguito sul Device, chiamato dal Device
- __\_\_host\_\___: eseguito sull'Host, chiamato dall’Host

Ad es: \_\_global\_\_ void kernelFunction(int *data, int size);

## Restrizioni dei Kernel CUDA
1. Esclusivamente Memoria Device ( \_\_global__ e \_\_device__ )
    - Accesso consentito solo alla memoria della GPU. Niente puntatori a memoria host.
2. Ritorno void ( \_\_global__ )
    - I kernel non restituiscono valori direttamente. La comunicazione con l'host avviene tramite la memoria.
3. Nessun supporto per argomenti variabili ( \_\_global__ e \_\_device__ )
    - Il numero di argomenti del kernel deve essere definito staticamente al momento della compilazione.
4. Nessun supporto per variabili statiche ( \_\_global__ e \_\_device__ )
    - Tutte le variabili devono essere passate come argomenti o allocate dinamicamente.
5. Nessun supporto per puntatori a funzione ( \_\_global__ e \_\_device__ )
    - Non è possibile utilizzare puntatori a funzione all'interno di un kernel.
6. Comportamento asincrono ( \_\_global__ )
    - I kernel vengono lanciati in modo asincrono rispetto al codice host, salvo sincronizzazioni esplicite.

### Compute capability
La Compute Capability di NVIDIA è un numero che identifica le caratteristiche e le __capacità__ di una GPU NVIDIA in termini di funzionalità supportate e limiti hardware.

Ad esempio: 
- Il numero massimo totale di thread per blocco è 1024 per la maggior parte delle GPU (compute capability >= 2.x). Un blocco può essere organizzato in 1, 2 o 3 dimensioni, ma ci sono limiti per ciascuna dimensione. Esempio:
    - Il prodotto delle dimensioni x, y e z non può superare 1024 (queste limitazioni potrebbero cambiare in futuro).
    - x: 1024 , y: 1024, z: 64
- Anche le griglie hanno un numero massimo di blocchi che possono contenere
    - Max grid x-dimension: 2^31-1
    - Max grid y/z-dimension: 65535

