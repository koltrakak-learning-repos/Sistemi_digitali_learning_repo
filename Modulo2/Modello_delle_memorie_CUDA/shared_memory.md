...

tipicamente la smem mi salva quando mi ritrovo un problema che per sua natura ha degli accessi in memoria globale scattered

...

ogni SM ha una sua shared memory!

### Bank conflict
...
32 persone e 32 archivi -> vorrei che ogni persona avesse il file che desidera in un archivio distinto -> acesso parallelo

se 32 persone hanno il loro file nello stesso ardchivio devono mettersi in file -> accesso sequenziale

...

conviene comunque avere un 32-way bank conflict che un accesso scattered alla memoria globale

**Come decidere se allocare più spazio alla cache L1 o a SMEM?**
... trial&error tm
