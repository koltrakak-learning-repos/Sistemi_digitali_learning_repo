### gigathread engine = scheduler globale
prende i blocchi generati dal lancio dei kerneò e li schedula ai vari SM dinamicamente ed automaticamente. Applicando anche una politica di load balancing verso  vari SM.

NB: una volta che un blocco viene assegnato ad uno SM il blocco rimane li fino al suo completamento.

### due livelli di parallelismo

ogni blocco schedulato dal giga thread engine viene elaborato in maniera indipendente, ordine di esecuzione non definibile

una volta che i sm si riempono l'esecuzione dei blocchi rimanenti è sequenziale

non c'è comunicazione tra thread di blocchi diversi anche se appartenenti allo stesso SM. La memoria condivisa è condivisa solo intra-block e non inter-block.

## Modello di esecuzione CUDA

### SIMD vs SIMT

simd = un unico thread

simt = vari thread CON PERCORSI DI ESECUZIONE INDIPENDENTE

la caratteristica fondamentale è che permette di scrivere degli if -> divergenza, gestita internamente dall'hardware
tuttavia questo ha un costo, in quanto il ramo if e il ramo else vengono eseguite sequenzialmente in questo ordine

...

### warp

warp sono raggruppamenti di 32 thread ottenuti divendendo un blocco. 

Si distingue tra vista logica che considera a livello di blocco e vista fisica che opera considerando i warp 

I warp sono sempre di 32 thread indipendentemente dalla generazione di GPU. è importante definire dimensione dei blocchi multipla di 32 in questo modo non si hanno warp con dei thread disabilitati che non fanno niente ma occupano risorse all'interno dello SM. 

un blocco di thread, indipendentemente dalla organizzazione 1D, 2D oppure 3D, __fisicamente__ viene organizzato come un insimeme di warp 1D

...

cosa serve riconoscere  i thread di un warp? Ad associarli le risorse come registri all'interno dell SM

### Contesto di esecuzione dei warp

...
warp scheduler effettua una specie di cambio di contesto nell'eseguire i vari warp da lui gestiti. Questo cambio di contesto è poi __senza costo__ in quanto lo stato dei thread viene salvato all'interno del SM. 

## Scheduling dei warp
è presente un warp-scheduler per ogni partizione di uno SM

l'obiettivo dello warp-scheduler è semplicemente di occupare il più possibile le risorse disponibili. Quindi il primo warp che trova disponibile lo schedula