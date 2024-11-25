```
Le prestazioni effettive di un kernel non possono essere spiegate unicamente attraverso l'esecuzione dei warp.
```

Spesso, la potenza di calcolo teorica delle GPU moderne supera di gran lunga quella effettivamente sfruttabile nelle applicazioni reali. Il collo di bottiglia diventa la bandwith per l'accesso alla memoria. 

### Limiti Prestazionali
Per ottimizzare un kernel CUDA è cruciale comprendere se il collo di bottiglia risiede negli accessi alla memoria o nella capacità computazionale della GPU. Questa distinzione determina le strategie di ottimizzazione da adottare.

In questo senso i kernel si dividono in:
- **Memory Bound**: La GPU trascorre più tempo in attesa dei dati rispetto a eseguire calcoli (poche operazioni per byte letto/scritto)
- **Compute Bound**: La GPU trascorre più tempo a eseguire calcoli rispetto all’attesa dei dati (molte operazioni per byte letto/scritto)

Queste due tipologie pongono il problema di quale metrica utilizzare per misurare le performance di un kernel: Bandwith o FLOPS?

### Il Modello di Performance Roofline
Il modello Roofline è un metodo grafico utilizzato per rappresentare le prestazioni di un algoritmo (o di un kernel CUDA) in relazione alle capacità di calcolo e memoria di un sistema. Utile per capire se un algoritmo viene limitato da problemi di calcolo o da problemi di accesso alla memoria.

Abbiamo bisogno di definire due grandezze; considerando il contesto dei kernel si hanno:
- **Intensità Aritmetica (AI)**:    
```
AI = FLOPs / Bytes trasferiti
```

- **Soglia di Intensità Aritmetica**:
```
Soglia (AI) = Theoretical Computational Peak Performance (FLOPs/s) / Bandwidth Peak Performance (Bytes/s)
```

**Interpretazione dell’Intensità Aritmetica**:
- Bassa intensità aritmetica (AI < Soglia):
    -  **Memory Bound**, poiché richiede più accesso alla memoria rispetto al calcolo.
- Alta intensità aritmetica (AI > Soglia):
    - **Compute Bound**, poiché esegue molti calcoli rispetto ai dati trasferiti.

**NB**: Un dato algoritmo ha sempre la stessa intensità aritmetica; tuttavia, **la soglia varia in base al sistema considerato**. Di conseguenza, un kernel o un problema può essere classificato come comupute bound/memory bound **dipendentemente dalla GPU che si sta considerando**.


... diagramma ...


**Considerazioni:**
- In un contesto *memory bound*, è di notevole importanza ottimizzare gli accessi alla memoria considerando la gerarchia delle memorie (località dei dati) e i pattern di accesso.

- In un contesto *compute bound*, è cruciale massimizzare l'**occupancy** delle unità di elaborazione, la scelta del tipo di dato e il parallelismo a livello di istruzione.

- Inoltre, le roofline sono multiple se si considerano i vari tipi di dato (che hanno più o meno unità di elaborazione) e la gerarchia delle memorie