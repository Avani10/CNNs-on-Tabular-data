# CNNs-on-Tabular-data

#### Test case of using neural network library fastai in comparison to ensemble methods, motivated from lesson 4 of fastai tutorials

From my past experience I have worked with language models / time series forecasting using neural networks (RNNs & Wavenets). I was always curious to compare neural networks with more generic techniques like ensemble methods (Random Forest, GBM) on tabular datset:

* The notebook shows one test case picked up from fastai tutorials shown as an example, the neural net was trained on the same datset

* I further used ensemble methods to see if these methods could perform better

### Observations:

1. The fastai models seemed to predict the dominant class most of the times, with a lower precision & recall for the less dominant class

2. I could see better results with ensemble methods but these were highly optimized via gridsearch cross validation techniques to find the best parameters

For making the conditions identical, both models were trained on 70% of dataset with the same computation resources (Google Collab). Test dataset was 30%, the ensemble methods were compared to each other with the help of ROC-AUC curves. GBM performed the best on test set.

### Recommendations:

Recommending the best architecture for tabular data would not be ideal through just one test case, but for this one ensemble methods seemed to have performed better than neural network. This could be due to optimization of results through searching best parameters for ensemble methods while for fastai library the benchmark was set by the results seen in lesson 4 of the tutorial. I did play around with neural net parameters but the results were more or less the same. A more thorough training would be best to take this comparison idea further.
