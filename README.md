# Independent Learning in Constrained Markov Potential Games

This repository provides the code for simulations of our independent learning algorithm `iProxCMPG`. We consider two constrained multi-player environments, both of which are inspired by unconstrained variants presented in [Narasimha et al. (2022)](https://ieeexplore.ieee.org/document/9992762):
 - a demand-response marketplace for energy grids, and
 - a pollution tax model.

Parts of our implementation, e.g., estimation of value function gradients and projection onto the policy space, are based on the code provided by [Leonardos et al. (2022)](https://openreview.net/forum?id=gfwON7rAm4) for their paper on "Global convergence of multi-agent policy gradient in Markov potential games". Respective sections are marked within our code.

## Instructions
The code was tested using `python 3.9.2`. To run the simulations, first install the required packages:
```bash
pip install -r requirements.txt
```
Then, execute the run script which will start 10 independent runs with respective seeds for each of the presented experiments:
```bash
./run_simulations.sh
```
Results are stored in the `experiments` directory. Since some runs may take up to a few hours on consumer-grade CPUs, we recommend executing multiple runs in parallel. In our experiments, we used a cluster of 15 4-core CPUs. The script for submitting the respective jobs to a cluster using the [slurm](https://slurm.schedmd.com/sbatch.html) scheduler is provided in `run_simulations_slurm.sh`.

To reproduce the plots (after simulations have terminated and results in `experiments` are complete) shown in the paper (Fig. 1 and 2), run:
```bash
python3 plot.py
```
If `latex` is not available on the system, run `python3 plot.py --no_latex`. The plots will appear as `pdf` files in the `plots` directory.

## References
- [[Narasimha et al. (2022)]](https://ieeexplore.ieee.org/document/9992762) Narasimha, D., Lee, K., Kalathil, D., and Shakkottai, S. (2022). Multi-agent learning via markov potential games in marketplaces for distributed energy resources. In 2022 IEEE 61st Conference on Decision and Control (CDC), pages 6350–6357. IEEE.
- [[Leonardos et al. (2022)]](https://openreview.net/forum?id=gfwON7rAm4) Leonardos, S., Overman, W., Panageas, I., and Piliouras, G. (2022). Global convergence of multiagent policy gradient in markov potential games. In International Conference on Learning Representations.
