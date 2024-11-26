## FFT (Fast Fourier Transform)
Le formule DFT e IDFT richiedono un __numero di moltiplicazioni__ (e somme) __pari ad N^2__. Infatti sono necessarie N moltiplicazioni per ognuno degli N elementi della n-pla di arrivo. Poiché il tempo di calcolo è praticamente determinato da tali moltiplicazioni, possiamo dire che esso cresce proporzionalmente ad N^2.

Tuttavia, __nel caso di N potenza di 2__, è possibile sfruttare alcune simmetrie nel calcolo ed ottenere lo stesso risultato della DFT con un tempo di calcolo __proporzionale a N__, conseguendo unafondamentale riduzione. 

Gli algoritmi che permettono di conseguire tale risultato sono molteplici: ad essi viene attribuita la denominazione di trasformata di Fourier veloce (FFT = Fast Fourier Transform).

__OCCHIO__: Si faccia attenzione, nel caso si voglia utilizzare una libreria che implementa uno degli algoritmi della FFT, alle convenzioni utilizzate nella specifica implementazione.
- Ad esempio l’ordine degli elementi della n-pla
- oppure la presenza del coefficiente 1/ sqrt(N) sia sulla formula di trasformazione che in quella di antitrasformazione (per maggiore simmetria), anziché 1/N sulla sola formula di antitrasformazione, ecc.

### Come funziona:

bisogna sfruttare la natura periodica delle sinusoidi...

1. dividere la sommatoria della DFT in due, una con indici pari e l'altra con indici dispari

... continua