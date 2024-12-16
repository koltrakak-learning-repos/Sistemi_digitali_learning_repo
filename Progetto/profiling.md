### Strumenti principali

1. Nel caso in cui sia necessaria l’analisi delle **interazioni host-device** è possibile utilizzare il profiler integrato di CUDA (nvprof) che fornisce informazioni basilari sui tempi di esecuzione dei kernel e sui trasferimenti di memoria tra host e device.

2. Nsight Compute: Strumento di profilazione e analisi approfondita per **singoli kernel CUDA**.

### Metriche dentro a Nsight-compute

- **Compute (SM) Throughput [%]**
    - Questo valore indica la percentuale di utilizzo delle unità di calcolo Streaming Multiprocessor (SM) rispetto al loro massimo teorico. Un percentuale alta significa che il kernel sfrutta una parte significativa della capacità di calcolo disponibile della GPU.
    
- **Memory Throughput [%]**
    - Rappresenta la percentuale di utilizzo della bandwith della memoria DRAM della GPU rispetto al massimo teorico. Una percentuale relativamente basso indica che l'accesso alla memoria non è un fattore limitante per questo kernel; la GPU ha margine per gestire più trasferimenti di memoria.