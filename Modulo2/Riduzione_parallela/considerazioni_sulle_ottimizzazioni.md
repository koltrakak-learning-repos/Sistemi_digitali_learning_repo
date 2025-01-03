Far processare ad un blocco di thread più blocchi di dati migliora le prestazioni perchè:
- Non ho più metà dei miei thread all'interno del blocco inutilizzati
- Più operazioni di memoria **indipendenti** per thread migliorano le prestazioni nascondendo meglio la latenza.

Unrolling è utili quando:
- Instruction Bottleneck
	- At 17 GB/s, we’re far from bandwidth bound, and we know reduction has low arithmetic intensity. Therefore a likely bottleneck is instruction overhead
- Ancillary instructions that are not loads, stores, or arithmetic for the core computation
	- In other words: address arithmetic and loop overhead
- Strategy: unroll loops!

Qua l'utilizzo della smem è utile perchè
- è abbastanza grande e riesco a starci in un blocco! 
- riduco gli accessi alla memoria globale che, seppur coalescenti, sono lenti
