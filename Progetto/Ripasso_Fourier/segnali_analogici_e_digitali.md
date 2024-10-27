Le telecomunicazioni (comunicazioni a distanza) si basano sull’__invio e la ricezione di segnali contenenti l’informazione desiderata__. Nell’ambito delle comunicazioni elettriche questi segnali corrispondono a tensioni e correnti variabili nel tempo, spesso ottenute convertendo in segnali elettrici
segnali di altra natura, ad esempio sonori o visivi, tramite degli appositi trasduttori (microfoni, telecamere). Una volta inviati e ricevuti, questi segnali vengono di nuovo convertiti nella forma originaria da altri trasduttori (altoparlanti, schermi televisivi).

### Segnali analogici e segnali digitali
Nel caso dei __segnali analogici__, che variano cioè in maniera “analoga” alla sorgente che li ha generati, x(t) è una funzione definita su tutto l’asse dei tempi (in altre parole il tempo varia con continuità, per cui x(t) è detta __“tempo continua”__) e x può assumere, al variare di t, tutti i valori appartenenti ad un certo intervallo; x(t) è cioè anche __“continua nei valori”__, oltreché nel tempo.

A titolo di esempio sono di tipo analogico i segnali che si presentano all’uscita di un microfono, di un giradischi, di un telefono fisso tradizionale.

Nei segnali __digitali__ ad ogni istante di tempo la funzione x(t) può assumere soltanto un numero finito di valori (per i cosiddetti segnali binari solo zeri ed uni) ed è perciò detta __“discreta nei valori”__. Inoltre in questi segnali l’informazione non varia con continuità nel tempo, ma a precisi istanti posti ad un intervallo fisso T, perciò nel caso in cui si voglia rappresentare solamente l’informazione (cioè il messaggio numerico), conviene adottare un modello “tempo discreto” in cui la funzione x(t) risulta definita soltanto per una successione numerabile di istanti. In altre parole la funzione x(t) oltre ad essere discreta nei valori risulta in questo caso anche __“tempo discreta”__.

NB: Si noti tuttavia che poiché nei dispositivi fisici __un segnale elettrico non può che essere definito su tutto l’asse dei tempi__, sarà
necessario associare al messaggio numerico tempo discreto un segnale numerico tempo continuo, così come sarà mostrato in seguito. Un esempio tipico di segnale numerico è quello prodotto da un lettore di compact disk (uscita digitale del lettore).

#### Classificazione in breve
- __Segnali tempo continui o tempo discreti__: I primi sono definiti su tutto l’asse dei tempi i secondi solo per un’infinità numerabile di istanti t_n con n intero, posti ad un intervallo T. Il segnale in questo caso non è altro che una successione di valori x_n = x(t_n)
- __Segnali continui o discreti nei valori__: Nei primi, fissato un generico istante t_1 , la funzione x=x(t_1) può assumere con continuità tutti i valori di un determinato intervallo, [-M,M], nei secondi solo una quantità numerabile, normalmente finita.