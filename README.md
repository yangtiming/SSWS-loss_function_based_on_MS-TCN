# SSWS-loss_function_based_on_MS-TCN
Supervised Sliding Window Smoothing Loss Function Based on MS-TCN for Video Segmentation

## [Supervised Sliding Window Smoothing Loss Function Based on MS-TCN for Video Segmentation](https://link.springer.com/chapter/10.1007/978-981-16-8885-0_24 )

## Abstract
Recently, more and more videos have been uploaded to the network, so that video analysis task has been one of the most important applications in various fields. At present, video analysis methods can be divided into two kinds: weakly supervised video action segmentation and supervised video action segmentation. The former uses a sliding window or Markov model, while the latter uses the TCN model. In this paper, we introduce the Supervised Sliding Window Smooth Loss Function (SSWS) into the TCN baseline, which is a complement to MS-TCN smoothing loss function TMSE. In this method, three discriminant frames are selected from the video prediction sequence and combined into an adaptive sliding window to selectively smooth the whole prediction sequence. In particular, it doubles the penalty when it slides to the wrong place in the category. Compared to TMSE, our method effectively increases the receptive field of smoothing loss function. And, the proposed new supervised loss function only penalizes error frames. The experiment shows that compared with the Smoothing loss function TMSE of MS-TCN, SSWS has significantly improved in the three datasets: 50Salads, GTEA and the Breakfast Dataset.

## Citation
    Yang, Timing.
    "Supervised Sliding Window Smoothing Loss Function Based on MS-TCN for Video Segmentation." 
    International Conference on Computing and Data Science. Springer, Singapore, 2021.
