1. Nella cartella dove c'è il grafico aprire gnuplot da terminale digitando:
    
    ``` 
    # Apri gnuplot da terminale
    gnuplot
    ```

2. Configurare il grafico con:

    ```
    # Configura il grafico
    set title "Spettro di Ampiezza"
    set xlabel "Frequenza (Hz)"
    set ylabel "Ampiezza"
    set grid
    ```

3. Visualizza i dati con:

    ```
    # Visualizza i dati
    plot "<nome file con i dati.txt>" using 1:2 with lines title "<nome linea graficata>"
    ```

4. Per ridimensionare gli assi

    ```
    # Ridimensina asse x
    set xrange [min:max]
    # Per annullare
    unset xrange
    ```

(backtick ` si digita con altgr + ')