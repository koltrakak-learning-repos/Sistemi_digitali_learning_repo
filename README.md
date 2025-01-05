Repository in cui ho salvato gli appunti ed il progetto svolto per il corso di Sistemi Digitali-M durante l'anno 2024/2025.

## CODICE PROGETTO
Il progetto implementa un **algoritmo di compressione basato su una implementazione parallela con CUDA di una FFT-2D**.

Il codice si trova all'interno della cartella *"Progetto/CUDA_FFT/"*. Al suo interno troverete:
- Vari file '.cu' che contengono il codice (sia sequenziale che parallelo) delle varie versioni, mano a mano ottimizzate, del codice del mio progetto.
- Una cartella chiamata *"nsight-compute_reports/"* in cui ho salvato i file *.ncu-rep* per i miei kernel.
- Una cartella chiamata *"nvvp_reports/"* in cui ho salvato i file *.nvvp* di Nvidia Visual Profiler in cui si può osservare il grado di concorrenza ottenuto nelle varie versioni del codice.
- Una cartella chiamata *"versioni_sbagliate/"* in cui ho salvato approcci NON funzionanti di parallelizzazione del mio codice.
- Una cartella chiamata *"versioni_senza_miglioramenti_o_peggiorative/"* in cui ho salvato approcci che non hanno influenzato o hanno peggiorato le performance del mio codice.

## Cosa guardare? (per i prof. Tosi e Mattoccia)
Sicuramente il codice principale dentro a *"Progetto/CUDA_FFT/"*.

Durante la presentazione del mio progetto, ho intenzione di mostrare anche qualcosa riguardo le versioni peggiorative dentro a *"Progetto/CUDA_FFT/versioni_senza_miglioramenti_o_peggiorative"*. Ad esempio:
- qual'è stata l'idea di miglioramento che non ha funzionato.
- come mai questa idea non ha funzionato.
  
Per voi non è quindi strettamente necessario andare a vedere cosa c'è dentro, ma potrebbe esservi utile siccome è una cosa che mi piacerebbe portare all'esame.

Le versioni_sbagliate invece, in quanto non funzionanti, non le porterò all'esame e quindi possono essere ignorate.

## Resto della repo
- Le cartelle *"Modulo1/"* e *"Modulo2/"* contengono appunti di teoria presi durante il corso delle lezioni.
- La cartella *"Progetto/"* contiene materiale vario riguardante il progetto:
  - *"Progetto/CUDA_FFT/"* contiente quanto scritto sopra.
  - *"Progetto/Ripasso_Fourier/"* contiene vari appunti riguardanti la teoria alla base della trasformata di Fourier e della FFT.
  - *"Progetto/Test/"* contiene vari test riguardanti implementazioni sequenziali in C degli algoritmi di trasformata, e uso di librerie per il parsing di immagini e file audio.
 
