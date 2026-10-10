## Running the Web3 Circuit Breaker Example

To see the library in action with real calibrated probabilities, run the provided Web3 demo. You will need to install the optional example dependencies (like `scikit-learn`).

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8475.svg)](https://doi.org/10.5281/zenodo.8475)

```bash
# Clone the repository
git clone https://github.com/bakshiyogesh/decision-policy.git
cd decision-policy

# Install the library and the example dependencies
pip install -e ".[examples]"

# Run the demo
python examples/web3_circuit_breaker_demo.py
