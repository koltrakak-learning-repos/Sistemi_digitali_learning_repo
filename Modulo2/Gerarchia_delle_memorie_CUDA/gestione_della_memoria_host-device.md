### Connettività Host-Device e Throughput di Memoria
Trasferimenti host-device mediante il PCIe sono lenti e quindi da minimizzare in numero (aggregare i dati in un unico trasferimento per sfruttare la meglio la bandwith)

### Cosa importante da sapere | memoria pinned
... una sorta di buffer per i trasferimenti verso il device

... sistema operativo non può fare swap out di quelle pagine

... copia dei dati e overhead nelle operazioni

Se si desidera velocizzare i trasferimenti è possibile allocare sull'host della memoria pinned direttamente con: *cudaMallocHost()* . Evitando in questo modo la doppia copia.

...

### Memoria zero-copy
La memoria zero-copy è memoria pinned dell’host che è mappata nello spazio degli indirizzi del device

non c'è trasferimento **esplicito** dai dati tra host e device.

Con questa tecnica il device può accedere alla memoria dell'host. I dati vengono comunque trasferiti con il PCIe, ma non vengono salvati sulla GDDR.

**OCCHIO**: due entità possono accedere allo stesso posto -> sincronizzazione necessaria tra host e device.

**NB**: chiaramente con questa memoria sono limitato dalla bandwith del PCIe piuttosto che dalla bandwith della GDDR e quindi vado più lento. Utile quando la memoria del device finisce. Ma **in generale non si utilizza direttamente questa metodologia**.

### Unified Virtual Addressing (UVA)
un unico spazio di indirizzamento virtuale
...
cudaMemCopy da fare in maniera esplicita

### Unified Memory
Si basa su UVA. Adessono non c'è bisogno di fare trasferimenti in maniera esplicita

... 

Allocazione in maniera **lazy**!!!

la memoria non è pinned...

Semplifica molto il codice e fornisce un livello di astrazione superiore nella gestione della memoria. Per le massime prestazioni è il caso di usare la tecnica che conosci già. (importante allocare memoria sul processore che la usa per evitare trasferimenti, se tutto è astratta non si ha controllo su dove la memoria viene allocata)