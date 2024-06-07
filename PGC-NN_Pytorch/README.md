# PGC-NN

## Description

This repository presents PGC-NN: a graph neural network able to semantically enriching Points of Interest (POIs) according to spatiotemporal data.

This approach is a piece of a more complete work [1], removed from the original repository [2].

## Data

To generate the matrices specified in [1], see details in [matrix generation documentation](docs/matrix_generation.md).

<span style="color: red;">**OBS: The Data was removed to not overload the repository.**</span>

## Requirements and Execution

### Requirements

* Python 3
    * [link](https://www.python.org/downloads/)
* virtualenv
    * `$ pip install virtualenv`

### Execution

After generate the matrices, you need to run 'processing_data.ipynb' notebook in 'gowalla/processed/' folder to reduce the original dataset and favor tests execution. You will need also to comment line 12 and uncomment line 11 in 'main.py' file.

To execute this project, first you need to create and activate your virtual environment. The step-by-step is:

<span style="color: red;">**OBS: The follow step-by-step was created to use in Linux Operation System.**</span>

1. Create your virtual environment
    * `$ virtualenv myenv`
    * Replace 'myenv' with a name of your choice
    * Execute this command only at the first time

2. Activate the virtual environment
    * `$ source myenv/bin/activate`

3. Update pip
    * `$ pip install --upgrade pip`

4. Install requirements
    * `$ pip install -r requirements.txt`
5. Execute
    * `$ python main.py`
    * see [info](docs/info.md) about pgc-nn execution (pt-br)

***

## References

[1] Cláudio G.S. Capanema, et al. Combining recurrent and Graph Neural Networks to predict the next place’s category. Ad Hoc Netw. 138, C (Jan 2023). https://doi.org/10.1016/j.adhoc.2022.103016

[2] CLAUDIOCAPANEMA. poi_gnn. Disponível em: <https://github.com/claudiocapanema/poi_gnn>.

***