# <img src="./docs/images/ftsts-logo.png" alt="Logo" width="50" style="vertical-align: middle; margin-right: 10px;"> FTSTS - Forced Temporal Spike-Time Stimulation

_Open-Loop Deep-Brain Stimulation for Controlling Synchronization of Spiking Neurons._

## Example Usage

```sh
python3.10 -m venv venv # create virtual env
source venv/bin/activate
pip install -r requirements.txt # install dependencies

gcc -shared -o libkuramoto.so -fPIC src/kop.c -lm # (opetional)
python src/main.py
```

> _note: computing the Kuramoto Synchrony Parameter is computationally intensive. Using the optional command above to compile the C implementation of kuramoto_syn permits the program to use a faster implementation._

## Resources

- **[ftsts-code](https://github.com/ftsts/ftsts-harnessing-synaptic-weight)** - source code to generate weight profiles for the original open-loop regime.
- **[ftsts-paper](https://doi.org/10.3389/fncom.2019.00061)** - original work on open-loop stimulation regime.
