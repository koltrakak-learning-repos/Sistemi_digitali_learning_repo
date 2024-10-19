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
- Tuttavia, ISA diverse hanno set di istruzioni SIMD differenti sebbene spesso con funzionalità simili
    - Noi vedremo principalmente il set di istruzioni SIMD appartenente all'ISA x86
- __Anche all’interno della stessa ISA, possono esserci diversi set di istruzioni SIMD__. La tendenza è quella di supportare tutte le estensioni SIMD precedenti, principalmente per ragioni di compatibilità software



...

noi utilizzeremo principalmente SSE

### istruzioni tipiche per un ISA che supporta SIMD

operazioni per leggere e scrivere blocchi di dati

### MISCUGLIO 

con gcc è possibile mischiare codice C e codice assembly, tuttavia fare così è un casino.

Meglio __intrinsics__, queste funzioni non fanno altro che mettere inline una porzione di codice assembly.

...

più il dato è piccolo, più il vantaggio è grande

... consiglio che non capisco bene ...

se salvo roba allineata mi trovo tutto il registro esteso in una cache line (linee di cache sono sempre multiple di 2 e quindi i 
registri estesi si infilano bene senza accavvallamenti su più cache lines)
    -> è bene salvare il registro esteso in maniera allineata con la dimensione del registro.

in leggere è bene non leggere/scrivere mai disallineato. Altriementi devo fare due read/write -> peggioramento delle performance.
Ad esempio array di interi in c sarebbe meglio memorizzarli ad un indirizzo multiplo di 4

si può avere allineamento sia con allocazione
    - statica -> direttiva
    - dinamica-> mmalloc()

### CACHE
chiedi a chatgpt per la definizione di una linea di cache e di indirizzamento all'interno di una cache

...

a volte si caricano dati che si sa che non verranno mai più utilizzanti in seguito ne tantomeno i dati vicini. Pertano è desiderabile
non salvare la cache line.

...

l'aritmetica dei puntatori si basa sulla dimensione del dato puntato