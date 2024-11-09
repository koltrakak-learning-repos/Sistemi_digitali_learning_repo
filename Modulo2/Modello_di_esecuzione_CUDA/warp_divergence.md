### ALTRA DOMANDA FONDAMENTALE
    In che modo la divergenza influisce sul parallelismo e sulle prestazioni?

#### warp divergence
Cosa è la Warp Divergence?

- In un warp, idealmente tutti i thread eseguono la stessa istruzione contemporaneamente per massimizzare il parallelismo SIMT (condividono un unico Program Counter [ in Architetture Pre-Volta] ).
- Tuttavia, se un'istruzione condizionale (come un if-else o switch ) porta thread diversi a percorrere rami diversi del codice, si verifica la __Warp Divergence__.
- In questo caso, il __warp__ esegue __serialmente__ ogni ramo, utilizzando una maschera di attività (calcolata automaticamente in hardware) per abilitare/disabilitare i thread.
- La divergenza termina quando i thread riconvergono alla fine del costrutto condizionale.
- La Warp Divergence può significativamente degradare le prestazioni perché i thread non vengono eseguiti in parallelo durante la divergenza (le risorse non vengono pienamente utilizzate).
- Notare che il fenomeno della divergenza occorre solo all’interno di un warp.

... bella l'immagine a slide 98 ...

Località
- La divergenza si verifica solo all'interno di un singolo warp.
- Warp diversi operano indipendentemente. I passi condizionali in differenti warp non causano divergenza.
Impatto
- La divergenza può ridurre il parallelismo fino a 32 volte se solo un thread (sui 32 del warp) esegue all'interno di un ramo

## Gestione della Warp Divergence
Per ottenere le migliori prestazioni, è consigliabile evitare percorsi di esecuzione diversi all'interno dello stesso warp. A tal fine, va notato che l'assegnazione dei thread a un warp è __deterministica__, quindi organizzando i dati in modo che i thread nello stesso warp seguano lo stesso percorso di
esecuzione, puoi ridurre o evitare la divergenza del warp.

...esempi...