# Enhancing and Evaluating Patient Privacy with HyperLogLog++

This project explores the privacy and accuracy characteristics of HyperLogLog++ (HLL++) in the context of federated hospital data aggregation. HLL++ is a probabilistic data structure designed for efficient cardinality estimation. We investigate how it performs under varying conditions of data duplication, sparsity, and sketch resolution—focusing on its ability to preserve $k$-anonymity, a privacy guarantee requiring that each patient be indistinguishable from at least $k-1$ others.

In federated clinical networks, where hospitals share only aggregated summaries (not raw data), techniques like HLL++ help prevent re-identification while allowing accurate query responses such as “how many patients have diabetes?”. Our distributed implementation simulates multi-hospital settings and evaluates how well HLL++ handles duplicate patient records, privacy risks, and estimation tradeoffs across different configurations.

## Directory
- ``hllpp.py`` contains the abstract data type for the HyperLogLog++ instance.
- ``hllpp_testing.ipynb`` contains testing code for the abstract data type.
- ``hllpp_experiments.ipynb`` contains the visualizations for all experiments run.
