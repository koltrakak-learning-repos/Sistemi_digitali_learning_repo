...

Un kernel o un problema può essere classificato come comupute bound/memory bound in base alla GPU che si sta considerando

### Gerarchia di memoria in CUDA
- Registri

- Memoria locale (nice name)

- SMEM e chache L1

- Memoria costante

- Memoria texture (don't care)

- Memoria globale

Bel riassunto in slide 43

**NB**: Molte delle memorie sono programmabili, sta quindi al programmatore cercare di usare le memorie (on chip) più vicine agli SM per avere la maggiore bandwith possibile.