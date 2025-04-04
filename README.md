The code implementing the GS_CS algorithm is located in the `python` folder. It is commented and follows the notations used in the article.

The two experiments described in the article correspond to the notebooks presented here:

- `run_experience_lasso` generates a Lasso problem instance, computes an approximate solution using scikit-learn, and refines it by applying GS_CS.  
- `train_initial_NN` trains a neural network on FashionMNIST and saves the weights in the `poids` folder. The resulting weight vector is used as the starting point for GS_CS in the `run_experience_compression` notebook. In this notebook, the algorithm aims to improve the pruning rate while maintaining a minimal performance threshold.
