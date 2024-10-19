## SOMMA DI MATRICI
Nell'elaborazione di matrici con CUDA, è fondamentale definire come i thread vengono mappati agli elementi della matrice. Questo processo di __mapping incide direttamente sulle prestazioni dell'algoritmo__.

__Impatto della configurazione__ 
La configurazione scelta per la griglia e i blocchi (1D o 2D) influenza come i thread sono associati agli elementi della matrice.
- Una configurazione adeguata permette a ogni thread di gestire porzioni ben definite dei dati.
- Una configurazione non ottimale può portare a inefficienze, come thread che gestiscono intere colonne o righe della matrice, oppure che elaborano dati in modo non bilanciato.

Suddivisione
- La matrice può essere suddivisa in sottoblocchi di dimensioni arbitrarie.
- La scelta delle dimensioni dei blocchi influenza le prestazioni.

Cosa Garantire
- Copertura completa della matrice.
- __Scalabilità__ per diverse dimensioni di matrice.
- Coerenza dei risultati con l'elaborazione sequenziale.
- __Accesso efficiente alla memoria__ (lo vedremo in seguito).

### Osservazioni primo esperimento:
- Tutte le configurazioni GPU offrono un miglioramento rispetto alla CPU.
- Miglioramento drastico passando da (1,1) a dimensioni di blocco maggiori.
- Le configurazioni con più blocchi e thread mostrano miglioramenti drammatici, con speedup superiori a 131x.
- Le differenze tra le configurazioni (16,16) e (32,32) sono relativamente piccole, suggerendo una __saturazione dell'utilizzo delle risorse GPU__.
- Esiste un punto di ottimizzazione oltre il quale ulteriori aumenti nella dimensione o nel numero dei blocchi non producono miglioramenti significativi.

### Perchè è inefficiente utilizzare blocchi con pochi thread?
- Overhead di gestione: Lanciare tanti blocchi singoli crea un enorme overhead di scheduling (dei blocchi) e gestione per la GPU.
- Mancato sfruttamento della località: I thread non sono raggruppati in modo da sfruttare efficientemente la memoria cache e la memoria condivisa dei blocchi.
- Inefficienza nell'utilizzo dei warp: Le GPU operano su gruppi di thread chiamati warp (tipicamente 32 thread). Con un thread per blocco, la maggior parte delle unità di elaborazione in ogni warp rimane inutilizzata (lo vedremo).

Come sono formate le immagini? 
Le fotocamere hanno miglioni di fotodiodi disposti in una configurazione 2D che campionano la luce.

...

baseIndex = (i * width + j)*3 ; * 3 perchè stiamo indirizzando blocchi da 3

### conversione grayscale
media pesata delle 3 componenti con il verde componente con peso maggiore

...

i thread aggiuntivi che fanno delle no-op occupano comunque delle risorse in memoria

    librerie stb per caricare e salvare immagini .jpg/.png in un programma C. 

### image flipping
...

### blurring
quando strabordo metto degli zeri (padding)

### perchè ci interessano le convoluzioni con CUDA
per reti neurali convoluzionali. Esse in sostanza fanno questo, cioè: data una immagine e un sacco di filtri, si fanno solo delle gran convoluzioni (somme di prodotti)