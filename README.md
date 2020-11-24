Neural Unsupervised Style Transfer based on Multiple-Attribute Text Rewriting - ICLR 2019 (https://openreview.net/forum?id=H1g2NhC5KQ). Latent representation pooling is not implemented in this version. Additionally, a new loss from the classifier is added which gives the model more ability to control style.

Tested this model on PASTEL and Yelp dataset which needs to put in the Data folder in the same directory as the Code folder.

Denoising Loss

![Denoising Loss](https://raw.githubusercontent.com/ddhruvkr/Unsupervised-Style-Transfer/master/DenoisingLoss.png)

Cycle Consistency + Classification Loss

![Cycle Consistency + Classification Loss](https://raw.githubusercontent.com/ddhruvkr/Unsupervised-Style-Transfer/master/CycleConsistency_ClassificationLoss.png)

Final Objective

![Final Objective](https://raw.githubusercontent.com/ddhruvkr/Unsupervised-Style-Transfer/master/Loss.png)

To run the code, first call the main.py file in the Classification folder to train the classifier. Then call the main.py file in the Generator folder.

TODO:

1) Add a language model loss.
2) Implement latent representation pooling.
