# TSPArchipelago

TSPArchipelago is an implementation of an evolutionary algorithm
that tries to approximate the Traveling Salesman Problem (TSP)
using an Island Model Genetic Algorithm (IMGA).
The algorithm optimizes the tour length for a given set of
cities using genetic operations and local search techniques.

## Requirements

Make sure you have the necessary libraries installed.
You can use `pip` to install them from the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

## How to Run

You will need a CSV file containing the distance matrix.
Example file: tour100.csv.
The file should be in the ./tours_data/ directory,
and you need to modify the filename in the code `run.py`
to point to your own file.

Run the `run.py` script to start the optimization process.
The script will automatically read the distance matrix from
the provided file and run the genetic algorithm for optimization.

```bash
python run.py
```
