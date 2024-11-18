...
kernel parent e kernel child procedono in modo indipendente. parent lancia i kernel child in maniera asincrona 

#### Visibilità e Sincronizzazione:
- Ogni child grid lanciata da un thread è __visibile__ a tutti i thread dello stesso blocco.
- Se i thread di un blocco terminano prima che tutte le loro griglie child abbiano completato, il sistema attiva automaticamente una sincronizzazione implicita per attendere il completamento di queste griglie.
- Un thread può sincronizzarsi esplicitamente con le proprie griglie child __e con quelle lanciate da altri thread nel suo blocco__ utilizzando primitive di sincronizzazione (cudaDeviceSynchronize).
- Quando un thread parent lancia una child grid, __l'esecuzione della griglia figlio non è garantita immediatamente__, a meno che il blocco di thread genitore non esegua una sincronizzazione esplicita.


### Memoria 
...

puntatori alla memoria globale possono essere passati, puntatori alla memoria locale/condivisa no... 

...

#### Memoria Globale e Costante:
• Le griglie parent e child condividono lo stesso spazio di memoria globale (accesso concorrente) e memoria costante. Tuttavia, la memoria locale e condivisa (shared memory) sono distinte fra parent e child.
• La coerenza della memoria globale non è garantita tra parent e child (be careful), tranne che:
    ○ All'avvio della griglia child.
        - Tutte le operazioni sulla memoria globale eseguite dal thread parent prima di lanciare una griglia child sono garantite essere visibili e accessibili ai thread della griglia child.
    ○ Quando la griglia child completa.
        - Tutte le operazioni di memoria eseguite dalla griglia child sono garantite essere visibili al thread genitore dopo che il genitore si è sincronizzato con il completamento della griglia child

#### Race condition tra kernel padre e figlio
...

__NB__: Se il kernel parent necessita dei risultati del kernel figlio, deve aspettarlo sincronizzandosi esplicitamente con cudaDeviceSynchronize.