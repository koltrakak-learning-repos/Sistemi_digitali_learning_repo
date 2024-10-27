## Paradigma di elaborazione SIMD
Una singola istruzione SIMD elabora multipli dati eseguendo su di essi la medesima operazione. Questi dati sono memorizzati  in __registri speciali/estesi__ della CPU.

### eXtended Multimedia Register
I registri XMM della CPU di una architettura SIMD sono di taglia estesa (eg, 64, 128, 256, 512 bit) rispetto ai registri dell’architettura base. Pensabili come ad una sorta di __array di registri__ normali.

Le operazioni sui registri estesi equivalgono circa ad operazioni multiple contemporanee sui registri singoli. Chiaramente se prima, ad esempio, avevamo bisogno di una ALU, adesso ne abbiamo bisogno di n per eseguire i calcoli in parallelo, lo stesso vale per tutte le altre unità funzionali.

OSS: una condizione necessaria per far si che si abbia un vantaggio utilizzando i registri estesi è che si possano caricare/scaricare  dati in questi registri velocemente e facilmente.

NB: nel caso di overflow in questo tipo di registri, si ha un grosso overhead di gestione rispetto al caso dei registri classici

### Caratteristiche SIMD
- Incrementa il parallelismo agendo a livello dei dati
- Può essere utilizzata con tutte le strategie menzionate (eg, pipelining)
- Richiede un numero di modifiche hardware limitato vs SISD come l’integrazione di ALU addizionali con impatto modesto in termini maggiori di risorse utilizzate
- Supportato da quasi tutte le ISA più diffuse (eg, x86, ARM e RISC-V)
- Tuttavia, ISA diverse hanno set di istruzioni SIMD differenti, sebbene spesso con funzionalità simili
    - Noi vedremo principalmente il set di istruzioni SIMD appartenente all'ISA x86
- __Anche all’interno della stessa ISA, possono esserci diversi set di istruzioni SIMD__. La tendenza è quella di supportare tutte le estensioni SIMD precedenti, principalmente per ragioni di compatibilità software.

### Evoluzione set di istruzioni SIMD dentro a x86 
L’ampia diffusione di questo tipo di istruzioni mediante CPU general-purpose si è avuta però solo nei primi anni 90 con l’uso massiccio di __dati multimediali__ (audio, immagini) e operazioni intensive (encoding, real-time processing, etc) su tali dati.

In ordine cronologico, le estensioni SIMD più notevoli sono state:
- MMX (MultiMedia eXtension)
    - deprecata/obsoleta
    - elaborazione di dati interi con registri a 64 bit
- SSE (Streaming SIMD Extension)
    - fino a SSE4
    - registri a 128 bit
    - supporto per dati floating point
- AVX (Advanced Vector Extensions)
    - 3 versioni, l'ultima è AVX-512
    - registri da 256 fino a 512 bit 

__Noi utilizzeremo principalmente SSE__

### x86: tipi di dati supportati
- I tipi di dati possono essere byte (8 bit), interi a 16 bit, interi a 32 bit, single precision floating-point (32 bit), double precision floating-point (64 bit)
- Per i tipi byte e interi sono supportate operazioni con dati signed e unsigned
- Non tutte le istruzioni o estensioni supportano tutti i tipi di dato (eg, MMX solo interi 8, 16, 32, 64 bit)
- In generale non c’è molta ortogonalità nell’ISA x86, ovvero, un istruzione per gli interi a 32 bit non è detto che esista per interi a 16 bit

### istruzioni tipiche per un ISA che supporta una estensione SIMD
- rendere agevole e veloce la lettura/scrittura, il packing/unpacking e interleaving/de-interleaving di dati tra memoria principale e registri estesi 
- eseguire operazioni comuni nell’ambito del image/signal processing come somme, sottrazioni, moltiplicazioni, etc
- individuare massimi e minimi, replicare elementi all’interno dei registri estesi, consentire operazioni con saturazione e operazioni molto comuni come SAD (Sum of Absolute Differences)

### Intrinsics 
Le operazioni SIMD sono disponibili come istruzioni in linguaggio assembly. E’ possibile però anche scrivere la maggior parte del codice in linguaggi di alto livello e iniettare dove necessario il codice assembly per le istruzioni SIMD. Tuttavia, una soluzione meno complicata per il programmatore consiste
nell’utilizzare “Intrinsics”.

Gli __intrinsics__ non sono altro che funzioni inline (ovvero non c'è una vera chiamata a funzione) che corrispondono a singole o sequenze di istruzioni SIMD in linguaggio assembly.

## Tipi di dato e operazioni in x86 SSE
I registri SSE a 128 bit possono essere utilizzati con i seguenti tipi di dato:
- __m128i (interi)
    - 16x8 bit – epi8  (signed), epu8 (unsigned)            ; epi = extended packed integer
    - 8x16 bit – epi16 (signed), epu16 (unsigned)
    - 4x32 bit – epi32 (signed), epu32 (unsigned)
    - 2x64 bit – epi64 (signed)
- __m128 (float)
    - 4x32 bit – tipicamente denominati ps (floating-point a singola precisione)
- __m128d (double)
    - 2x64 bit – tipicamente denominati pd (floating-point a doppia precisione) più il dato è piccolo, più il vantaggio è grande

### Lettura dalla memoria
L’istruzione seguente (SSE) carica dalla memoria 128 bit a partire dall’indirizzo __mem_addr__ che deve essere __allineato__ (guarda linee di cache) con la __taglia del registro__ (16 byte). I dati letti nel registro __m128i__ possono essere byte (16 elementi), interi a 16 bit(8 elementi), interi a 32 bit (4 elementi)

    __m128i  _mm_load_si128(__m128i const* mem_addr)    ; si = signed integer

Esistono anche istruzioni (SSE) per leggere dati floating-point a singola (32bit, 4 elementi in __m128) e a doppia precisione (64 bit, 2 elementi in __mm128d)

    __m128   _mm_load_ps (float const* mem_addr)        ; ps = packed single precision
    __m128d  _mm_load_pd (double const* mem_addr)       ; pd = packed double precision

### Scrittura in memoria
L’istruzione seguente (SSE) scrive in memoria 128 bit di interi a partire dall’indirizzo __mem_addr allineato con la taglia del registro__ (16 byte).
I dati scritti in memoria possono essere byte (16 elementi), interi a 16 bit (8 elementi), interi a 32 bit (4 elementi)

    void _mm_store_si128 (__m128i* mem_addr, __m128i a)

• Esistono anche istruzioni (SSE) per scrivere dati floating-point a singola (32bit, 4 elementi in __m128) e a doppia precisione (64 bit, 2 elementi in __mm128d)

    void _mm_store_ps (float* mem_addr, __m128 a)
    void _mm_store_pd (double* mem_addr, __m128d a)


### Allineamento
Se uso indirizzi allineati con la taglia del registro, mi trovo tutto il registro esteso in una unica cache line (linee di cache sono sempre multiple di 2 e quindi i registri estesi si infilano bene senza accavvallamenti su più cache lines).

Per questo motivo è sempre bene non leggere/scrivere mai in maniera disallineata. Altrimenti devo fare due read/write -> peggioramento delle performance.

Ad esempio array di interi in C sarebbe meglio memorizzarli ad un indirizzo multiplo di 4

Nei linguaggi di programmazione, si può forzare l'allineamento delle variabili sia nel caso di allocazione:
- Statica, mediante direttiva.
    - Ad esempio: int A[8] __\_\_attribute\_\_((aligned(16)));__ // 16 byte (128 bit) aligned
- Dinamica, mediante intrinsics appartenenti a SSE
    - void* _mm_malloc (size_t size, size_t align)
    - void _mm_free (void * mem_addr)
    - ad esempio: char *SIMD_array = (char *) _mm_malloc(1024, 16);     ...    _mm_free(SIMD_array);

Ricorda: l'aritmetica dei puntatori si basa sulla dimensione del dato puntato

## Botta di istruzioni utili per programmazione nel modello SIMD

### Caricamento e estrazione con registri estesi
- Caricamento (non efficiente) di un registro esteso con dati non necessariamente allineati
    - __m128i _mm_set_epi32 (int e0, int e1, int e2, int e3)
    - Istruzione lenta! Un'alternativa potrebbe essere il mascheramento con degli and, ma non cambia troppo.
- Estrazione (non efficiente) di dati dai registri estesi
    - E’ possibile estrarre un singolo valore da un registro esteso
    - int _mm_extract_epi32(__m128i a, const int imm8)
    - L’istruzione esiste per vari formati di dati (eg, interi a 8 bit, 16 bit, 32 bit)

### Copia con inserimento di un valore
Questa istruzione consente di copiare un registro in un altro modificando un singolo elemento del registro sorgente

### Mascheramento e branching in SIMD
Il mascheramento è utilizzato molto nel mondo SIMD. Questo perchè __non sono permessi i branch__, l'unica modalità contemplata è quella di un __unico flusso di esecuzione__ su dati diversi. Considera questo esempio: se il mio processore deve decidere se prendere un branch o meno in base al valore di un registro esteso, se 3 linee su 8 rispettano la condizione di branching e le altre 5 no, prendo il branch oppure no? Non ha senso la domanda in primo luogo.

Una soluzione, se si vuole applicare delle operazioni solo su alcuni dati di un registro esteso è usare il __mascheramento__.

Una situazione simile si incontra con la gestione dell'overflow nel caso delle somme fra registri estesi, quest'ultimo infatti in SIMD non è gestito.

### operazioni logiche
Anche le operazioni logiche bit a bit tra registri estesi vengono usate molto in questo mondo. Alcune notevoli sono:
- and   
- or 
- xor 
- andnot
    - dest = NOT(a) AND b
In generale queste istruzioni sono utili per mascheramento

__ricorda__: differenza tra shift logico e aritmetico
- Logico: faccio entrare degli zeri
- Aritmetico: faccio ancora entrare degli zeri ma devo anche mantenere il segno
    - per questo motivo lo shift aritmetico a sinistra non ha senso, corrisponde a quello logico

### operazioni di shifting

### shuffling
Le istruzioni di shuffling sono istruzioni che consentono di smistare i dati di un registro sorgente in posizioni differenti nel registro destinazione __in funzione di una maschera__

Utili quando si desiderano disporre in un determinato ordine i valori presenti in un registro esteso

### packing e unpacking
Istruzioni che impacchettano in un registro il contenuto di due registri effettuando il casting (con saturazione) a un tipo di dato di __dimensione inferiore__ (32→16, 16→8).

Impachettare è utile perchè permette un maggiore grado di parallelismo, a scapito però della precisione dei valori rappresentati.

    - Bisogna capire in che contesti può essere usato.

    __m128i _mm_packus_epi16    ; packus = pack unsigned saturation

In modo specchiato si può anche fare unpacking anche detto interleaving del contenuto di due registri estesi. Questo gruppo di istruzioni consentono di combinare il contenuto di due registri attraverso interleaving, ovvero alternando prima una lane di un registro e poi l'altra fino a riempire il registro destinazione (e tralasciando delle lane nei due registri sorgente).

per la natura dell'operazioni esistono sia istruzioni di unpacking "high", cioè che considerando prima i lane a partire dai most sig. bits, e low che fanno il contrario. 

### blending
Questo gruppo di istruzioni consentono di combinare contenuto di due registri __in funzione di un valore immediato (blend) o un registro (blendv)__

Disponibili per i tipi di dati interi a 8 (blendv) e 16 bit (blend).

### Istruzioni di confronto
Richiamando quanto detto prima, nel paradigma SIMD le istruzioni di confronto si usano in modo diverso rispetto ad un paradigma scalare. Qui elementi posti nelle stesse posizioni all’interno dei registri estesi subiscono il confronto e generano una maschera che sintetizza il risultato di quest'ultimo. Sfruttando questa maschera nelle operazioni successive si può decidere a quali lane applicare una istruzione e a quali no.

- Sono previste tre tipologie di confronti: uguale (EQ), maggiore (GT) o minore (LT)
    - EQ: per valori interi a 8, 16, 32, 64 bit
    - GT: per valori interi signed a 8, 16, 32, 64 bit
    - LT: per valori interi signed a 8, 16, 32 bit

### Istruzioni di confronto senza branch/salti?
Rispieghiamo quanto già detto sui branch meglio: 
- Come appena visto, esistono istruzioni per eseguire confronti ma non istruzioni di salto condizionato (branch)
- Questo era prevedibile essendo possibile un solo flusso di esecuzione delle istruzioni
- Per eseguire operazioni diverse sui dati all’interno di un unico flusso di esecuzione si utilizzano delle maschere (eg, l’output di una istruzione di confronto)
- Mediante le maschere è possibile filtrare dati che soddisfano (o non soddisfano) determinate condizioni

### ESERCIZIO DEL PROF
prova anche la sua strategia