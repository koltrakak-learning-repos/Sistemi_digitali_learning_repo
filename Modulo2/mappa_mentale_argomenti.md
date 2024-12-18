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
    