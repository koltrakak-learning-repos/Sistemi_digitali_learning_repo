Come in C, il programmatore deve allocare e deallocare memoria manualmente. In più, in CUDA è necessario gestire esplicitamente il **trasferimento dei dati tra host e device**.

### Connettività Host-Device e Throughput di Memoria
La memoria **GDDR** della GPU offre una larghezza di banda teorica più alta (fino a 2-3 TB/s per HBM). Il collegamento **PCIe**, tra CPU e GPU, ha una larghezza di banda teorica massima di 64 GB/s.
- Significativa differenza tra la larghezza di banda della memoria GPU e quella del PCIe.
    - I trasferimenti di dati tra host e dispositivo possono rappresentare un collo di bottiglia.
- Essenziale ridurre al minimo i trasferimenti di dati tra host e dispositivo

**In breve**: Trasferimenti host-device mediante il PCIe sono lenti e quindi da minimizzare in numero (aggregare i dati in un unico trasferimento per sfruttare la meglio la bandwith ammortizzando il "costo fisso" delle funzioni di trasferimento)



## Cosa importante da sapere | MEMORIA PINNED
La memoria allocata dall’host di default è **pageable**. Questo ha tra le varie conseguenza il fatto che una determinata pagina di memoria possa essere swapped out. Qualunque tentativo di accesso a memoria paged out di norma produce un page fault che scatena il relativo swap in per un accesso sicuro alla memoria. Questo non accesso sicuro **non è garantito** con la GPU.  

    La GPU non può accedere in modo sicuro alla memoria host pageable (mancanza di controllo sui page fault)

Il trasferimento di memoria tra host e device avviene in tre step:

1. Il driver CUDA alloca temporaneamente memoria sull'host di tipo pinned (page-locked/pinned, in altre parole non soggetta a swap-out o bloccata in RAM).
2. Copia i dati dalla memoria host sorgente alla memoria pinned.
3. Trasferisce i dati dalla memoria pinned alla memoria del device

**OSS**: Chiaramente c'è un overhead di allocazione e copia preliminare della memoria dei passi 1 e 2 

La memoria pinned è quindi una sorta di buffer per dei trasferimenti sicuri verso il device.

**NB**: È possibile velocizzare i trasferimenti allocando sull'host direttamente della memoria pinned con la funzione: *cudaMallocHost()* . Evitando in questo modo l'overhead dovuto alla doppia copia.

Occhio però che, allocando troppa memoria pinned, si potrebbero degradare le prestazioni del sistema host che si ritrova con meno memoria swappabile.

**NB_2**: La memoria pinned è **più costosa da allocare/deallocare** rispetto a quella paginabile, ma accelera i trasferimenti di **grandi volumi di dati**, soprattutto se ripetuti dallo **stesso buffer**, ammortizzando il costo iniziale




### Memoria zero-copy
La memoria zero-copy è memoria **pinned** dell’host che è **mappata nello spazio degli indirizzi del device**. In questo modo il device può accedere direttamente (zero copie) alla memoria dell'host senza la necessità di copiare **esplicitamente** i dati tra le due memorie.

Non c'è trasferimento **esplicito** dai dati tra host e device. Chiaramente un trasferimento implicito sul PCIe è comunque necessario dato che le due memorie sono distinte.

**OCCHIO**: Due entità distinte (Host e device) possono accedere allo stesso posto causando corse critiche 
    -> sincronizzazione necessaria tra host e device.

**NB**: chiaramente con questa memoria sono limitato dalla bandwith del PCIe piuttosto che dalla bandwith della GDDR e quindi vado più lento. Utile quando la memoria del device finisce, o per dati piccoli/utilizzati raramente per cui il costo di trasferimento non sarebbe giustificato. **In generale non si utilizza direttamente questa metodologia**.





### Unified Virtual Addressing (UVA)
Con questa tecnica si estende l'idea della memoria zero-copy, implementando un unico spazio di indirizzamento virtuale condiviso tra CPU e GPU (la memoria fisica rimane distinta).

Rispetto al caso precedente non vi è più distinzione tra un puntatore virtuale host e uno device. Il runtime di CUDA gestisce automaticamente la mappatura degli indirizzi virtuali agli indirizzi fisici nella memoria della CPU o della GPU, a seconda delle necessità.

Si ha comunque bisogno di sapere se si sta allocando memoria su GPU o CPU dato che UVA **non gestisce la migrazione dei dati**, richiedendo trasferimenti manuali espliciti.




## Unified Memory (UM)
Si basa su UVA. Fornisce uno spazio di memoria virtuale unificato a 49 bit che permette di accedere agli stessi dati da tutti i processori del sistema usando un unico puntatore. Adessono non c'è bisogno di pensare ai trasferimenti.

### Memoria gestita
La Memoria Gestita (Managed Memory) si riferisce alle allocazioni di Unified Memory che sono gestite automaticamente dal sistema sottostante e sono interoperabili con le allocazioni specifiche del device.

Si ha Gestione Automatica: Il sistema migra automaticamente i dati tra host e device, semplificando il codice. 

**Interessante**: La allocazione della memoria managed avviene in maniera **lazy**!!! Solamente quando la memoria viene effettivamente utilizzata si ha l'allocazione vera e propria. In particolare la memoria viene allocata sul primo dispositivo che la utilizza per questioni di performance (cercare di allocarla a priori su uno tra i vari dispositivi a caso potrebbe in seguito portare a necessità di trasferimenti lenti sul PCIe causati da page fault).  


Semplifica molto il codice e fornisce un livello di astrazione superiore nella gestione della memoria. Per le massime prestazioni però, è il caso di usare allocazioni e trasferimenti espliciti (tecnica che conosci già). È infatti importante allocare memoria sul processore che la usa per evitare trasferimenti, se tutto è astrattao via non si ha controllo su dove la memoria viene allocata.