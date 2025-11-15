# distributed-ai

## General

In the hackathon we tested two model architectures, namely a Transformer modified for tables (FTTransformer) and a XGBoost architecture to predict probabilities whether a tax form (represented as a table row) is a fraud.
We used XGBoost for the demo to get quick and precise results for our test/demo data and generate explanations.
We used FTTransformer, which we the trained for the same task, to apply federated learning, since a transformer architecture would be more accuracte for heterogenous data like it is in a real world scenario.

## Setup

To install/manage the dependencies, we use uv, which can be installed using the instructions on [this](https://docs.astral.sh/uv/getting-started/installation/) website.

After that, you can simply run `uv sync` to install all the current dependencies.

If you want to add dependencies, use `uv add <your python dependency>`.

## Use Provided Infrastructure
### Connect to the ssh:
```bash
ssh team18@129.212.178.168 -p 32605
```
You will be asked to enter the password for our team.

### Executing a single command on the server
```bash
ssh team18@129.212.178.168 -p 32605 'ls -l'
```

### Executing jobs
```
./submit-job.sh "<command>" [--name <job-name>] [--gpu]

# Example for running on CPU
./submit-job.sh "flwr run . cluster-cpu" 

# Example for running on GPU
./submit-job.sh "flwr run . cluster-gpu" --gpu
```

### Executing our Project
First connect to the server using ssh as above.
Then `cd` into the project directory. The project directory had to be renamed to `coldstart` for the server to work.
Pull the newest changes:
(If not already on the main branch)
```bash
git checkout main
```
```bash
git pull
```
It should work without passing any credentials because I added the personal access token.

Then you can run the project using the `submit-job.sh` script as shown above.

### Starting our frontend
In order to start the frontend, you need to execute the following command:
```bash
streamlit run frontend.py
```
