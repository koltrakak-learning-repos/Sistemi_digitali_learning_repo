## MOLTO IMPORTANTE | Pattern di accesso alla memoria globale
La maggior parte delle applicazioni GPU è limitata dalla larghezza di banda della memoria DRAM. Ottimizzare l'uso della memoria globale è quindi fondamentale per le prestazioni del kernel. Senza questa ottimizzazione, altri miglioramenti potrebbero avere effetti trascurabili.

```
Ottimizzare l'uso della memoria globale è una delle cose che impatta maggiormente le prestazioni.
```

### Modello di Esecuzione CUDA e Accesso alla Memoria
Sappiamo che la concorrenza è a livello di warp; abbiamo 32 potenziali accessi alla memoria separati? No si usano le **transazioni di memoria**.

- Istruzioni ed operazioni di memoria sono emesse ed eseguite per warp (32 thread).
- Ogni thread fornisce un indirizzo di memoria quando deve leggere/scrivere, e la dimensione della richiesta del warp dipende dal tipo di dato (es.: 32 thread x 4 byte per int, 32 thread x 8 byte per double).
- La richiesta (lettura o scrittura) è servita da una o più **transazioni di memoria**.
    - Una transazione è un'operazione atomica di lettura/scrittura tra la memoria globale e gli SM della GPU
    - Una transazione si applica sia ad una lettura che ad una scrittura tramite trasferimenti di memoria da 32, 64 o 128 byte

```
Ottimizzare l'accesso alla memoria globale significa ridurre il più possibile il numero di transazioni necessarie per il singolo warp.
```

### Caratteristiche Ottimali degli Accessi alla Memoria
- Accessi allineati alla memoria
    - L'indirizzo iniziale di una transazione di memoria è un multiplo della dimensione della transazione stessa.
    - Gli accessi non allineati richiedono più transazioni, sprecando banda di memoria (devo trasferire 128 byte quando me ne servivano solamente 64).
    - Nota: La memoria allocata tramite CUDA Runtime API, ad esempio con cudaMalloc(), **è garantita essere allineata ad almeno 256 byte**.
- Accessi coalescenti alla memoria
    - Si verificano quando **tutti i 32 thread in un warp accedono a un blocco contiguo di memoria**.
    - Se gli accessi sono contigui, l'hardware può combinarli in un numero ridotto di transazioni verso posizioni consecutive nella DRAM
    - Nota: In algoritmi specifici, la coalescenza può essere difficile o intrinsecamente impossibile da ottenere.


**CASO MIGLIORE**: Accessi allineati e coalescenti 
- Un warp accede a un blocco contiguo di memoria partendo da un indirizzo allineato.
- Ottimizza il throughput della memoria globale e migliora le prestazioni complessive del kernel


## Caching
blah blah ...

Le GPU non si affidano al principio di località temporale come le CPU, poiché elaborano molti thread che potrebbero accedere a dati differenti. La località spaziale invece continua ad essere (se non più) importante.



### Letture
...

**OSS**: Caricamenti “fine-grained” a 32 byte riducono lo spreco di banda rispetto a caricamenti più grandi da 128 byte su accessi disallineati e/o non coalescenti, dato che non vengono caricati altrettanti byte non richiesti. Stesso discorso per accessi completamente/per nulla sparsi.

...

### Scritture
Le scritture (store) vengono eseguite a livello di segmenti con granularità 32 byte. Le transazioni di memoria possono coinvolgere uno, due o quattro segmenti alla volta.

Quando un warp scrive ad indirizzi contigui il numero di transazioni è minimizzato (diviso per 4 nel caso ottimo). Se un warp scrive allineato ma non in maniera contigua allora si ha un impatto negativo causato dalla mancanza di coalescenza.

**NB**: Chiaramente le scritture non vengono influenzate dalle cache (possono solo eventualmente aggiornarle), mentre le letture vengono notevolmente velocizzate da esse. Per questo motivo a volte è conveniente cercare di spostare un accesso non coalescente sulle letture piuttosto che sulle scritture (vedi problema della matrice trasposta); in questo modo le prime letture lente caricano le cache e velocizzate le letture seguenti.