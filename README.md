# Forced Temporal Spike-Time Stimulation (FTSTS)

_Open-Loop Deep-Brain Stimulation for Controlling Synchronization of Spiking Neurons._

## Running the program

```sh
# Install Dependencies. (eg...)
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# (opetional) Compile kop.c if using C implementation of kuramoto_syn.
gcc -shared -o libkuramoto.so -fPIC kop.c -lm

# Run the example simulation configuration.
python src/main.py
```
