### extended multimedia register
registri più grossi presenti in una architettura SIMD (una sorta di array di registri).

operazioni su registri estesi equivalgono a operazioni multiple contemporanee sui sotto registri singoli.

OSS: una condizione necessaria per far si che si abbia un vantaggio utilizzando i registri estesi è che si possano caricare/scaricare 
dati in questi registri velocemente e facilmente.

NB: nel caso di overflow in questo tipo di registri, si ha un grosso overhead di gestione rispetto al caso dei registri classici

NB_2: chiaramente se prima avevamo bisogno di una ALU, adesso abbiamo bisogno di n ALU

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