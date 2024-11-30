## MOLTO IMPORTANTE | Pattern di accesso alla memoria
Questa è una delle cose che impatta maggiormente sulle prestazioni.

La concorrenza è a livello di warp, 32 potenziali accessi alla memoria? In realtà no si usano le **transazioni di memoria**.

Una transazione è un'operazione atomica di lettura/scrittura tra la memoria globale e gli SM della GPU.

...

CASO MIGLIORE: Accessi allineati e coalescenti. 

...

### Loads
...

I caricamenti “fine-grained” a 32 byte riducono lo spreco di banda rispetto ai cached loads da 128 byte su accessi disallineati/non coalescenti (80% vs 50% di utilizzo).

...

### Scritture
...