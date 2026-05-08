



## The Official Implementation of **XQueryer: An Intelligent Crystal Structure Identifier for Powder X-ray Diffraction**


<p align="center">

  <a href="https://doi.org/10.1093/nsr/nwaf421">
    <img src="https://img.shields.io/badge/DOI-Paper-darkred?style=for-the-badge" />
  </a>

  <a href="https://xqueryer.caobin.asia/about">
    <img src="https://img.shields.io/badge/Website-Project-black?style=for-the-badge" />
  </a>

  <a href="https://www.youtube.com/watch?v=OYPoh7K5uM0">
    <img src="https://img.shields.io/badge/YouTube-Demo-FF0000?style=for-the-badge&logo=youtube&logoColor=white" />
  </a>

  <a href="https://github.com/WPEM/XqueryerBench">
    <img src="https://img.shields.io/badge/Benchmarks-Code-blue?style=for-the-badge&logo=github" />
  </a>

  <a href="https://www.mrs.org/meetings-events/annual-meetings/archive/meeting/presentations/view/2025-mrs-spring-meeting/2025-mrs-spring-meeting-4205765">
    <img src="https://img.shields.io/badge/MRS-Talk-6A5ACD?style=for-the-badge" />
  </a>

  <a href="https://ccf.org.cn/chinadata2025/schedule_d_4038">
    <img src="https://img.shields.io/badge/ChinaData-Presentation-008080?style=for-the-badge" />
  </a>

</p>


Our system revolutionizes PXRD-based crystal identification through high-fidelity data synthesis and the cutting-edge **XQueryer** model. Seamlessly integrated with diffractometers, it enables precise, AI-driven material discovery and extends its capabilities to broader chemical applications. **XQueryer** comprises **1.03 B** parameters.



> [!IMPORTANT]
> **XQueryer is among the first AI frameworks to enable cross-system crystal structure identification directly from X-ray diffraction patterns without relying on conventional search-match pipelines.**  
> Unlike traditional XRD analysis workflows that heavily depend on handcrafted databases, phase-by-phase retrieval, and expert-guided matching strategies, XQueryer reformulates crystal identification as a representation learning problem over diffraction space itself. By combining large-scale simulated diffraction data, deep feature alignment, and cross-system structural retrieval, XQueryer demonstrates that neural networks can learn transferable diffraction representations spanning diverse crystal systems and compositions. This work represents a major step toward universal AI-driven diffraction understanding, enabling automated crystal identification beyond fixed candidate libraries and moving XRD analysis closer to foundation-model-style structural reasoning.



## Overview
- **Source Code**: Available in the [./src](./src) directory.
 
- **Dataset**: [OneDrive](https://onedrive.live.com/?redeem=aHR0cHM6Ly8xZHJ2Lm1zL2YvYy81ZDg2MjYyMzg0NzBiNDllL0V1d09VMTNQM2JoSHNiU2lEMTRON3hZQmZCTEdCYTFjX0VhVkhrbGZUajRxZXc%5FZT0xa3liaFg&id=5D8626238470B49E%21s5d530eecddcf47b8b1b4a20f5e0def16&cid=5D8626238470B49E)
- **Benchmarks**: Access the benchmark code at repo [XqueryerBench](https://github.com/WPEM/XqueryerBench).
- **Simulation Code**: Available in the [./sim](./sim) directory.
- **RRUFF–MP ID Matching**: Available in the [./match](./match) directory.

## Tutorials
- **Training/Val/Testing**: [model_tutorial](./src/Tutorial.ipynb)
- **Simulation**: [sim_tutorial](./sim/XRD.ipynb)
- **High-throughput simulation**: [HTsim_tutorial](./sim/tutorial_sim.ipynb)
## About 
Maintained by Bin Cao. Please feel free to open issues in the Github or contact Bin Cao
(bcao686@connect.hkust-gz.edu.cn) in case of any problems/comments/suggestions in using the code. 



