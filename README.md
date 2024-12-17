**BayesFRET** is a Python GUI program for estimating the number and behavior of distinct conformational states from intrinsically disordered molecules (ex. enzymes, Holliday junctions). The observed molecules must come from single-molecule Förster Resonance Energy Transfer (smFRET) experiments that use the surface-immobilized method, or from surface-immobilized smFRET experiments.

The program implements an experiment-adjusted Hierarchical Dirichlet Process-Hidden Markov Model (HDP-HMM) to process time-binned photon intensities of the donor and acceptor dyes, or the data source, from the smFRET experiment. Each set of simulations produces a folder of graphs and numerical data. [My thesis paper](https://hdl.handle.net/20.500.11801/3955) provides an in-depth analysis of the math and process behind this program.

**Data Source**: Time-binned photon intensities (TXT) of the donor and acceptor dyes from a surface-immobilized smFRET experiment.

**Method**: Four controlled MCMC simulations using an experiment-adjusted Hierarchical Dirichlet Process-Hidden Markov Model (HDP-HMM).

**Results**: A folder of graphs (PNG) and numerical data (Pickle, TXT files).


# Installation
1. **Compatible Operating Systems**: Tested on Windows 11, macOS. Not yet for Linux, Docker.
1. **Python Interpreter & Environment**
    * Compatible versions: 3.11, 3.12, 3.13
    * This documentation will assume the Python CMD function is `python`, but your local system may use `python3`, `python3.12`, `py`, etc.
1. **Download Files**: Assign a folder specifically for the files of this program (ex. `C:\Users\Cyvu37\Documents\BayesFRET`). 
    * Make sure the parent directory of the program folder (ex. `C:\Users\Cyvu37\Documents`) doesn't require administrator access. BayesFRET must be able to produce subdirectories and files!
1. **Install Packages**: [chime](https://github.com/MaxHalford/chime), [darkdetect](https://github.com/albertosottile/darkdetect), [matplotlib](https://matplotlib.org/), [numpy](https://numpy.org/), [PySide6](https://doc.qt.io/qtforpython-6/), [scipy](https://scipy.org/)
    * If internet is available when BayesFRET opens, the app will auto-run the `python -m pip install -r requirements.txt` command to verify the required packages. Make sure you have [pip](https://pip.pypa.io/en/stable/installation/) and [setuptools](https://pypi.org/project/setuptools/) in your Python environment.
    * If no internet is available, the app will check your Python system for the required packages. If any package is missing, the program will auto-close.


# Handling the Program
The program can either use data from real experiments ("Experimental") or it can run by itself using randomly generated data simulating data from an smFRET experiment ("Synthetic"). Either way, the program will first visualize the data, then run the 4 MCMC simulations one at a time, and finally visualize the results. Graphs and numerical data are saved as soon as they are produced. See Section 3.1 of [the thesis paper] for more details.


## Data Source Requirements
* **Synthetic Mode**: None. The program generates its own data and compares it with the MCMC simulations.
* **Experimental Mode**
    * 2 text/CSV files, each with a single array of numerical values separated by whitespace/commas.
        * 1 for the intensities of the *acceptor* dye throughout the experiment.
        * 1 for the intensities of the *donor* dye throughout the experiment.
    * Parameters of the experiment for the Options section of the GUI.


# Running the Program
When BayesFRET runs the simulations, the program stores the information inputted through the GUI into a directory with the exact timestamp of when the run started, or the active directory. *DO NOT modify the BayesFRET or active directories, including their URIs and internal files while BayesFRET runs the simulations!* Moreover, since BayesFRET v1.0 does not support multithreading yet, *DO NOT interact with the GUI or its windows while it runs the simulations!*

BayesFRET uses a Universal file to store various miscellaneous data, including absolute URIs to their original directories. [...]


# Using Pickle Files

Each pickle file (`.p`) produced by this program contains data that can be instantly used in a Python environment. However, they contain custom Python classes from `code01_classes.py`. Copy `code01_classes.py` to your current directory to access the documentation of the Python classes. An advanced code editor like Visual Studio Code with extensions can provide the user with previews of documentation regarding those custom classes.

The following commands import all of the data produced by the program from a simulation:

```
import os, pickle
from code01_classes import Chain_History, Params, True_Samples, Universal
DIR = "...\\BayesFRET_syn - 2024 04 23, 13 12 01 614687" # Full directory path
function open_pickle(fname: str):
    return pickle.load(open( os.path.join( DIR, fname ), "rb" ))
U: Universal = open_pickle( "BayesFRET_Universal_class.p" )
C: list[Chain_History] = []
for fname in U.filenames:
    C.append( open_pickle(fname) )

# Synthetic data only
P = open_pickle( "BayesFRET_data_params_and_true.p" )
PARAMS: Params = P[0]
TRUE: True_Samples = P[1]

# Experimental data only
PARAMS: Params = open_pickle( "BayesFRET_data_params.p" )
```

* `Universal`: Holds all other miscellaneous variables, including...
    * Filenames (not filepaths) of the Chain_History pickle files (`U.filenames`).\*
    * List of the unique RNG seed values (`U.list_seeds`).
    * List of the RNG classes corresponding to each RNG seed value (`U.RNGs`).
    * Number of Posterior Quantiles (`U.Q`).
    * The full path of the active folder (`U.DIR_ACTIVE`)\* and a function to merge it with a filename (`U.FILEPATH(...)`).\*
    * The full path of the program's original folder (`U.DIR_PROGRAM`).\*
* `Chain_History`: History of samples regarding key variables throughout the MCMC simulation.
    * Experimental + synthetic data: Found in `BayesFRET_data_sim_#_history.p` where `#` is the RNG seed value.
* `Params`: Fixed parameters such as the options from the Algorithm Settings, Photoemission Priors, and Photophysics Priors sections of the GUI. 
    * Experimental data: Found in `BayesFRET_data_params.p`.
    * Synthetic data: Found in `BayesFRET_data_params_and_true.p`.
* `True_Samples`: Variables used in making the synthetic data from `code02_setup.py`.
    * Synthetic data only: Found in `BayesFRET_data_params_and_true.p`.

\*If you manually change a filename(s) or folder name(s) after the simulation finishes, change the appropriate variables in the Universal class.

```
# >> How to Handle Name Changes <<
# Folder name change: DIR --> "...\BayesFRET_syn FINAL"
# Filename change for 3rd seed: 'BayesFRET_data_sim_627_history.p' --> 'BayesFRET_627.p'

DIR = "...\\BayesFRET_syn FINAL"
U: Universal = open_pickle( "BayesFRET_Universal_class.p" )
U.DIR_ACTIVE = DIR                  # This updates the `FILEPATH` function.
U.filenames[2] = 'BayesFRET_627.p'  # This allows you to use the previous for loop to iterate all filenames.
...
with open(U.FILEPATH( "BayesFRET_Universal_class.p" ), "wb") as file:
    pickle.dump( U, file )
```


## Examples

Example 1: Using the default RNG seeds (5, 44, 356, 9918), use the following commands to extract the FRET efficiencies `eff` of each unique state at the last MCMC iteration `n` for the simulation with RNG seed 5 (`C[0]`).

```
n = -1
ks = C[0].K_set[n] # The unique states k at n = 1000.
return C[0].eff[n, ks]*100
```

Example 2: Find $\lambda^A_{s_t}$ for the simulation with RNG seed 9918 (`C[3]`).

```
return C[3].lam_A[ C[3].s_t ]
```