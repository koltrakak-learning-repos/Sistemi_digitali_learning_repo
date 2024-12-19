### Somiglianze fondamentali

- Suddivisione in problemi più piccoli:
    - Nella versione ricorsiva, i dati vengono suddivisi in campioni pari e dispari a ogni livello della ricorsione.
    - Nella versione iterativa, invece, questa suddivisione viene "preparata" tramite un'operazione di bit-reversal degli indici, per poi procedere ad aggiornare i risultati in-place nei cicli.

- Calcolo del twiddle factor: Entrambe le versioni calcolano i coseni e i seni (fattori "twiddle") che corrispondono alle rotazioni necessarie per combinare i risultati delle FFT parziali.

### Differenze principali
1. Struttura dell'algoritmo
    - Ricorsivo: L'algoritmo ricorsivo lavora a ogni livello scendendo verso sottoproblemi più piccoli, passando da un array di dimensione N a due array di dimensione N/2, fino al caso base N=1. La combinazione avviene "tornando indietro" lungo la pila di ricorsione.
    - Iterativo: L'algoritmo iterativo lavora invece partendo dal caso base, ovvero calcola direttamente la trasformata di singoli elementi (dimensione 1) e poi unisce i risultati in livelli successivi, raddoppiando la dimensione dei sottoproblemi (prima 2, poi 4, ecc.) fino a raggiungere N.

2.  Bit-reversal 
    - Nella versione iterativa, l'array di input viene ordinato in ordine bit-reversed prima di iniziare i calcoli. Questo riordino corrisponde implicitamente alla separazione in pari e dispari che avviene in modo esplicito nella versione ricorsiva.
    - Nella versione ricorsiva, invece, l'array viene suddiviso in campioni pari e dispari direttamente nei passi ricorsivi.

3. Esecuzione dei livelli
    - Ricorsiva: Ogni livello della ricorsione corrisponde a una chiamata a fft() per un array di dimensione dimezzata rispetto al livello precedente.

    - Iterativa: Ogni livello della trasformata (dimensione N_stadio_corrente=2^stadio) viene eseguito con un doppio ciclo for:
        - Il ciclo esterno scorre sui blocchi di lunghezza N_stadio_corrente.
        - Il ciclo interno combina i risultati dei blocchi di dimensione N_stadio_corrente_mezzi.