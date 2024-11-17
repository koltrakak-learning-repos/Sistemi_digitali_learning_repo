## Definizione
Il problema generale di applicare un'__operazione associativa__ e (non necessariamente) commutativa __su un array__ è noto come problema di riduzione. La sua esecuzione parallela è chiamata riduzione parallela.

complessità del lavoro = quante somme devo fare
complessità dello step = quanti passi devo fare

#### Riduzione Sequenziale
N-1 somme
N-1 step

#### Riduzione Iterativa a Coppie
N-1 somme
dopo ogni giro di somme la dimensione dell'array si dimezza fino a che non si ottiene il singolo elemento. Di conseguenza ho log_2(N) step da fare.

La cosa più importante è che adesso ad ogni step posso fare delle somme in parallelo dato che non ho più dipendenze.

Due implementazioni:
- Neighbored Pair
    - stride parte da 1 ed arriva ad N/2 raddoppiando ad ogni iterazione
- Interleaved Pair
    - stride parte da N/2 ed arriva ad 1 dimezzandosi ad ogni iterazione

Indipendentemente dall'approccio:
- per sommare N elementi, sono necessarie N-1 operazioni di somma totali.
- Ad ogni step, N/(2^step) operazioni.
- servono log₂N passi per arrivare ad un singolo elemento

La differenza principale tra i due approcci sta nel modo in cui gli elementi vengono accoppiati e nella distribuzione del lavoro tra i thread, non nel numero totale di operazioni o passi.

### Prima soluzione con divergenza
La confizione: if ((tid % (2 * stride)) == 0) causa __warp divergence__!
    -> Solo i thread con indice pari del warp eseguono :(
    -> 16/32 thread in esecuzione all'interno dei warp
Anche se non si considera la warp divergence ad ogni passo di riduzione solo una frazione (sempre minore ad ogni iterazione) dei thread del blocco esegue.

### Soluzioni con divergenza ridotto ma senza sfruttamento di tutti i thread
...
nella versione interleaved non c'è bisogno di calcolare l'index
