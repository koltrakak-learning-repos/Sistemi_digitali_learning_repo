## Cos'è una linea di cache?
Una linea di cache è l'unità fondamentale di memorizzazione all'interno della cache. È un piccolo blocco di dati che viene copiato dalla memoria principale alla cache quando il processore richiede un dato specifico. Se il dato è presente nella cache (evento chiamato cache hit), il processore può accedervi molto più velocemente rispetto a dover recuperare il dato dalla memoria principale

In pratica, quando il processore richiede un dato che non è già nella cache (evento chiamato cache miss), la memoria cache carica dalla RAM un __intero blocco di dati, detto "linea di cache", contenente il dato richiesto più un insieme di dati vicini__. Questo è basato sul principio di località spaziale, cioè il fatto che se un dato viene richiesto, è probabile che anche i dati vicini possano essere utilizzati a breve.

### Dimensione della linea di cache
Una linea di cache ha una dimensione fissa, che varia in base all'architettura del sistema, ma può essere, ad esempio, di 32, 64 o 128 __byte__. 

Quindi, quando un dato viene caricato dalla RAM, non viene caricato solo quel dato specifico, ma un'intera porzione di memoria che lo contiene.

    Seguendo questo punto di vista, la memoria principale è pensabile come ad un insieme di cache lines.

Infatti __Tanenbaum dice__:

"Main memory is divided up into cache lines, typically 64 bytes, with addresses 0 to 63 in cache line 0, 64 to 127 in cache line 1, and so on".

NB: quando si cerca un dato all'interno della cache, prima si cerca la cache line-giusta (bit più significativi dell'indirizzo), poi si cerca il dato giusto al suo interno (bit meno significativi) con uno meccanismo simile alla paginazione.

Infine: "The most heavily used cache lines are kept in a high-speed cache located inside or very close to the CPU. When the program needs to read a memory word, the cache hardware checks to see if the line needed is in the cache. If it is, called a cache hit, the request is satisfied from the cache and no memory request is sent over the bus to the main memory. Cache misses have to go to memory, with a substantial time penalty of tens to hundreds of cycles".