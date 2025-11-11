# Linear Attention Benchmarks

Look at all of these results from different papers, supposedly of the same models on the same benchmark:

<div style="display:flex; flex-wrap:wrap; gap:1rem; align-items:flex-start;">
   <div style="flex:1 1 calc(50% - 0.5rem); box-sizing:border-box;">
      <img src="./images/deltanet.png" alt="deltanet" style="width:100%; height:auto; display:block;" />
   </div>
   <div style="flex:1 1 calc(50% - 0.5rem); box-sizing:border-box;">
      <img src="./images/mesanet.png" alt="mesanet" style="width:100%; height:auto; display:block;" />
   </div>
   <div style="flex:1 1 calc(50% - 0.5rem); box-sizing:border-box;">
      <img src="./images/rwkv7.png" alt="rwkv7" style="width:100%; height:auto; display:block;" />
   </div>
   <div style="flex:1 1 calc(50% - 0.5rem); box-sizing:border-box;">
      <img src="./images/atlas.png" alt="atlas" style="width:100%; height:auto; display:block;" />
   </div>
</div>

What's up with that? This repository contains independent and reproducible benchmarking results for various linear attention mechanisms. This should hopefully help to clarify the discrepancies seen in the literature.

Currently, the repository includes (each on own branch):

- Mechanistic Architecture Design (MAD)
- Pre-training loss comparison
