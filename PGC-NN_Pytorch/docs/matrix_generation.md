# Generate Matrices

To generate the matrices needed to execute PGC-NN, data will be required as follows:

1. user id
    - int
2. category name (must be Shopping, Community, Food, Entertainment, Travel, Outdoors or Nightlife)
    - str
3. place id
    - int
4. local_datetime
    - datetime
5. latitude
    - float
6. longitute
    - float
7. country name
    - str
8. state name
    - str

Each line will be a check-in for a given user. This data must be in file named 'checkins.csv' in folder '/gowalla'. After, you need uncomment line 12 and comment line 11 in 'main.py' file.

## Requirements and Execution

### Requirements

The requirements are the same listed in README.md.

### Execution

To execute this project, you need to create and activate your virtual environment. The step-by-step is:

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