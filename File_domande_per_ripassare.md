**1. Che cos'è un'architettura eterogenea? Perchè l'abbimo esplorata?**
    Un computer con un'architettura eterogenea è un tipo di computer che integra diversi tipi di processori o core di elaborazione al suo interno combinandone i vantaggi (e bypassando i svantaggi, le cose in cui non sono bravo le faccio fare ad un altro più bravo di me). Il computer, avendo a disposizione unità di calcolo eterogenee, può far eseguire su ognuna di esse il task per cui sono più portate velocizzando in questo modo i calcoli e ottenendo flessibilità. 

    Ad esempio: un computer con GPU e CPU, può eseguire sulla CPU i task sequenziali sfruttando la sua minore latenza delle operazioni, mentre può eseguire sulla GPU i task altamente paralleli sfruttandone l'alto throughput. A questo punto è il programmatore che decide dove far eseguire cosa per ottenere le migliori prestazioni.

## SIMD
**2. Descrivi in generale le caratteristiche del paradigma SIMD**
    SIMD è un paradigma di elaborazione di istruzioni in cui una singola istruzione SIMD elabora multipli dati eseguendo su di essi la medesima operazione. Questi dati sono memorizzati in __registri speciali detti estesi__ della CPU pensabili a come un array di registri normali.

    Questo paradigma:
    - Incrementa il parallelismo agendo a livello dei dati
    - Può essere utilizzato in combinazione con altre strategie di parallelismo (Pipelining, superscalarità, multithreading, ...)
    - Richiede un numero di modifiche all'hardware limitato rispetto a SISD con impatto modesto in termini di maggiori risorse utilizzate
        - Integrazione di ALU/unità di calcolo addizionali (una per ogni lane)
        - Integrazione dei registri estesi
        - Integrazione delle nuove istruzioni nel decoder

**3. Quali sono delle tipiche istruzioni supportate da un estensione SIMD?**
    In generale quelle classiche di un processore, con in aggiunta istruzioni per la manipolazione dei dati all'interno dei registri estesi:
    - Load e store dei registri estesi da/verso la memoria principale di blocchi di dati (Condizione necessaria per l'efficacia del paradigma SIMD è che i registri estesi possano essere letti/scritti agevolmente) 
    - somme, sottrazioni, moltiplicazioni, etc tra registri estesi
    - operazioni logiche e di confronto bitwise tra registri estesi
    - manipolazione e riarrangiamento dei dati intra e inter registro esteso (blend, packing, unpacking, ...)
    - consentire operazioni con saturazione e operazioni molto comuni come SAD (Sum of Absolute Differences) o FMA (Fused Multiply Add)

**4. Che cos'è una operazione con saturazione e come mai è utile?**
    Una operazione con saturazione è una operazione che non va in overflow/underflow è utile quando serve solo sapere se un valore è molto grande o molto piccolo.

**5. Parlami del branching in SIMD**
    Con il paradigma SIMD, il branching condizionata al valore di un registro esteso non è supportato in quanto non applicabile (se 3 lane su 8 rispettano la condizione di branching salto o non salto? non ha senso...).

    Se si desidera compiere azioni differenti sui dati all'interno di un registro esteso, si utilizzano apposite maschere prodotte da apposite istruzioni di confronto SIMD e operazioni logiche. Queste maschere filtrano le lane che rispettano le "condizioni di branch" e a cui applicare le operazioni desiderate.

## Modello di programmazione CUDA
**6. Che cos'è il modello di programmazione CUDA?**
    Il Modello di Programmazione definisce la struttura e le regole per sviluppare applicazioni parallele su GPU. In particolare definisce:
    - Gerarchia di Thread: organizza l'esecuzione parallela in thread, blocchi e griglie, ottimizzando la scalabilità su diverse GPU.
    - Gerarchia di Memoria: Offre tipi di memoria (globale, condivisa, locale, costante, texture) con diverse prestazioni e scopi, per ottimizzare l'accesso ai dati.
    - API: Fornisce funzioni e librerie per gestire l'esecuzione del kernel, il trasferimento dei dati e altre operazioni essenziali.

**7. Che cos'è un thread CUDA**
    Un thread CUDA rappresenta un'unità di esecuzione elementare nella GPU. Ogni thread CUDA si occupa di un piccolo pezzo del problema complessivo, eseguendo calcoli su un sottoinsieme di dati in maniera sequenziale. Il parallelismo si ottiene coprendo l'intero spazio dei dati del problema, lanciando contemporaneamnete migliaia di thread (SIMT). Ogni thread esegue lo stesso codice del kernel ma opera su dati diversi, determinati dai suoi identificatori univoci (threadIdx,blockIdx).

**8. Qual'è il tipico workflow in CUDA?**
    1. Inizializzazione e Allocazione Memoria su CPU e GPU 
    2. Trasferimento Dati (Host → Device)
    3. Esecuzione (asincrona) del Kernel (Device)
        - (eventuali calcoli lato host)
    4. Recupero Risultati (Device → Host)
    5. Post-elaborazione (Host)
    6. Liberazione Risorse

**9. Come vengono organizzati i thread in CUDA?**
    Abbiamo due livelli di organizzazione:
    - i thread vengono raggruppati in blocchi
    - i blocchi vengono raggruppati in griglie
    Entrambe le struttura possono poi essere ulteriormente strutturate in maniera 1D, 2D o 3D in base al problema da risolvere.

**10. Come mai c'è bisogno di una gerarchia di thread?**
    La gerarchia di thread permette di scomporre problemi complessi in unità di lavoro parallele più piccole e gestibili, rispecchiando spesso la struttura intrinseca del problema stesso. Ad esempio si possono distinguere sottoproblemi paralleli diversi in griglie diverse e la griglia può essere strutturata al suo interno in blocchi che logicamente rispecchiano la risoluzione del sottoproblema.
    
    Il programmatore poi, può controllare la dimensione dei blocchi (e della griglia) per adattare l'esecuzione alle caratteristiche specifiche dell'hardware e del problema, ottimizzando l'utilizzo delle risorse della GPU a disposizione. La gerarchia risulta quindi scalabile e permette di adattare l'esecuzione a GPU con diverse capacità e numero di core. Il codice CUDA, quindi, risulta più portabile e può essere eseguito su diverse architetture GPU.

    Inoltre, la distinzione tra blocchi e griglie permette operazioni, come sincronizzazione e allocazione di memoria condivisa, che sarebbero troppo costose a livello globale.

**11. Che cos'è un kernel CUDA?**
    Un kernel CUDA è una funzione che viene eseguita in parallelo sulla GPU da migliaia/milioni di thread. Al suo interno vi è definito che cosa il singolo thread dovrà fare, ed il mapping ai dati che dovrà elaborare.

**12. Ci sono dei limiti per quanto riguarda il dimensionamento di blocchi e griglie?**
    Si, il numero massimo totale di thread per blocco è 1024 per la maggior parte delle GPU. Inoltre, le dimensioni di griglie e blocchi (anche 3D) sono limitate. I valori variano in base a alla compute capability della GPU.

    La Compute Capability (CC) di NVIDIA è un numero che identifica le caratteristiche e le capacità di una GPU NVIDIA in termini di funzionalità supportate e limiti hardware.

# 13. Che influenza ha il dimensionamento di blocchi e griglie sulle performance? (aggiusta parlando anche di occupancy e latency hiding)
    Il dimensionamento di blocchi è griglie ha conseguenze dirette sull'utilizzo delle risorse della GPU. Ad esempio: una configurazione con una manciata di blocchi con tanti thread, non attiva tutti gli SM della GPU diminuendo il parallelismo. Al contrario una configurazione con tanti blocchi composti da una manciata di thread causa un overhead eccessivo per lo scheduler dei blocchi **e non sfrutta bene le risorse all'interno dell'SM/SMSP in cui il blocco viene schedulato (un SM è in grado di gestire blocchi composti da centinaia di thread, pensa al caso estremo di <32 thread per blocco)**(???)

    Bonus: blocchi più grandi che accedono a dati localmente vicini possono anche sfruttare meglio la cache L1 rispetto a blocchi più piccoli.

# 14. Che influenza ha il mapping dei dati ai thread sulle performance?
    Innanzitutto, è ovvio che il mapping dei dati deve essere corretto, ovvero bisogna garantire che il mapping permetta di ottenere   un risultato del calcolo parallelo uguale a quello del calcolo sequenziale. Per fare questo il mapping deve: garantire una copertura completa dei dati senza ripetizioni.

    Più interessante è il fatto che il mapping dei dati influenzi la scalabilità ed il parallelismo che si riesce ad ottenere da un kernel. Con un mapping inappropriato il kernel potrebbe non scalare al crescere della dimensione dei dati (pensa all'esempio del kernel in cui un thread  somma un'intera colonna tra due matrici) oppure potrebbe diventare proprio impossibile risolvere il problema (pensa ad un mapping con solo threadId -> si è limitati dalla dimensione del blocco).

    Altri aspetti influenzati dal mapping dei dati sono: l'accesso alla memoria che idealmente deve essere allineato e coalescente, e il bilanciamento del carico tra i thread che idealmente deve essere uniforme.

    Riassumendo il mapping deve garantire:
    - Copertura completa dei dati
    - Scalabilità per diverse dimensioni dei dati
    - Coerenza dei risultati con l'elaborazione sequenziale
    - Accesso efficiente alla memoria

**15. Esistono diversi metodi di mapping dei dati?**
    Si, noi ne abbiamo visti due:
        - metodo lineare: più comodo quando si ha a che fare con configurazioni 1D
        - metodo per coordinate: più comodo quando si ha a che fare con configurazioni 2D/3D 
    Metodi diversi possono produrre indici globali differenti per lo stesso thread, impattando in questo modo le prestazione del kernel per aspetti come la coalescenza degli accessi in memoria.



