# distributed-ai

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
