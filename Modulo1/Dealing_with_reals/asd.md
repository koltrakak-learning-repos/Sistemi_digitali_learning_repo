...

### ULP (Unit in the Last Place)
Questa cifra da una indicazione sul grado di precisione possibile nella rapresentazione di un reale.

Questa cifra però ha un perso variabile in base alla mgnitudo del numero da rappresentare; per numeri grandi l'ULP verrà moltiplicato per l'esponente che in questo caso sarà grande. Per numeri piccoli l'ULP verrà moltiplicato per un esponente minore, e quindi si avrà una precisione maggiore.

...

Anche con float, quando si va verso dei numeri molto grandi, si perdono dei numeri interi che non si è più in grado di codificare. Questo è sempre dovuto al fatto che l'ULP viene modificato per l'esponente.

...

La soluzione è aumentare i bit della mantissa; ci sono però dei drawback
- la dimensione della mantissa è proporzionale con il quadrato dell'energia consumata
- proporzionale al tempo di calcolo.