Come dimensiono blocchi e griglie per i miei kernel se devo lavorare su N dati?
    - usare UN solo blocco con N thread non sfutta bene il parallelismo 
    - usare UN solo thread con N blocchi, uguale
    - serve un indice GLOBALE per identificare i thread all'interno di una stessa griglia.
        -> idx = blockIdx.x * blockDim.x + threadIdx.x -> simile all'indirizzamento che fa il compilatore per matrici n-dimensionali        -> svariati vantaggi:
            -> ...

A dire il vero esistono due metodi per determinare l'indice globale
    - metodo lineare
    - metodo per coordinate
NB: non producono lo stesso indice! -> Usare un metodo rispetto ad un altro produce, accessi in memoria diversi, con relativi svantaggi e vantaggi

COME SI DEFINISCE LA DIMENSIONE DELLA GRIGLIA E DEI BLOCCHI? 
    - la dimensione del blocco lo si dice arbitrariamente "ad occhio"
    - la dimensione della griglia, una volta definita la dimensione del blocco, dipende dalla dimensione dei dati.
        -> bisogna aggiustare il calcolo per tenere conto di una divisione senza resto che conta tutti i dati 