# Correct-by-construction for waveforms

Simply run
```
python sample_circuit.py
python sample_fsm.py
```


For combinational circuits in `sample_circuit.py` we sample 4 variable boolean logic min-terms. Testbench would also enumerate all valid (16) logical states.

For sequential circuits in `sample_fsm.py` we mainly use fsm1s from VerilogEval as template for generating code and constructing testbench. As we use a extremely naive method for generating test input patterns (random), we can not guarantee that every state is reachable through the test, or if the presented waveform signals in the problem have a single unique solution. Data derived from this might be ill-formed.