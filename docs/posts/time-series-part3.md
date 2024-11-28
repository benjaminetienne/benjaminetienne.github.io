---
slug: "Time Series in Python — Part 3: Forecasting taxi trips with LSTMs"
date: 2019-02-25
categories:
  - timeseries 
---

Introduction
------------

LSTM (Long Short-Term Memory) is a type a type of recurrent neural network (RNN) architecture, and was proposed in 1997 by [Sepp Hochreiter](https://en.wikipedia.org/wiki/Sepp_Hochreiter) and [Jürgen Schmidhuber](https://en.wikipedia.org/wiki/J%C3%BCrgen_Schmidhuber). RNNs are Deep neural networks specially designed to handle sequential data via recurrence mechanisms. 

<!-- more -->

They behave in an autoregressive manner, as they keep track of the past via internal states (hence the “memory” part). They have been used extensively for speech recognition, machine translation, speech synthesis, etc.  But what are LSTMs worth when used on time-series ? Well, they can prove to be very useful to model non-linear relationships, assuming the size of the data available is large enough..

The Uber use case: Bayesian forecasting
---------------------------------------

When looking for papers implementing time series forecasting with LSTMs, I found a paper written by Uber in 2017, [“](https://arxiv.org/pdf/1709.01907.pdf)[_Deep and Confident Prediction for Time Series at Uber_](https://arxiv.org/pdf/1709.01907.pdf)[”](https://arxiv.org/pdf/1709.01907.pdf). The basic question behind this paper is : **how confident can we be (****_ie how can we quantify uncertainty_****) making predictions with LSTMs ?**

The approach developped by Uber is a mixture of an encoder-decoder (used as an autoencoder) and a fully connected feed-forward network, used to predict the number of trips in a city based on previous data, or to detect anomalies in real time.

![](https://miro.medium.com/v2/resize:fit:1288/1*GE3Ld86Pr0qF85zaG7XQvA.png)

The Uber LSTM forecasting architecture (Zhu & Laptev, 2017)

The Uber paper is one of the first to use a Bayesian approach for time series forecasting. If you want to know more about Bayesian neural networks and Bayesian inference, you can look at the following links:

* [_Making your Neural Network Say I Don’t Know_](https://towardsdatascience.com/making-your-neural-network-say-i-dont-know-bayesian-nns-using-pyro-and-pytorch-b1c24e6ab8cd)
* [_Dropout as a Bayesian Approximation_](https://arxiv.org/pdf/1506.02142.pdf)
* [_Deep Bayesian Neural Networks_](/@joeDiHare/deep-bayesian-neural-networks-952763a9537)
* _Bayesian Methods for Hackers_, Cameron Davidson-Pilon

Bayesian Neural Networks
------------------------

To put Bayesian neural networks in a nutshell, BNNs estimate a probability distribution over each weight, whereas classical Neural networks try to find the optimal value for each weight. When you hear “Bayesian”, think “probability distribution”.

**_— But what’s the_** **_link between forecasting_** **_u_****_ncertainty and Bayesian networks_** **_?_**

Imagine two weather experts, Bob and Alice. You both know they are quite good at predicting weather temperatures, but sometimes they get it wrong too. You ask them what will the temperature be tomorrow at 8am so you can know if you need to put your coat on or not.  
Bob says : “It will be 15.6°C”.  
Alice says : “I’m 95 percent sure that the temperature will be between 16 and 18°C”.  
Although they do not seem to agree, who would you trust ?

Personally, I would trust Alice for two reasons:

* **She gave me a** **confidence interval**. I feel more reassured when someone is able to tell me something and how much he/she is confident with this information.
* **I do not** **really care about the exact temperature**, because a 1°C difference in temperature will not influence my decision to put on my coat.

However, had Alice told me “_I’m 95 percent sure that the temperature will be between 0°C and 18°C_”, I would have said that although she gave me a confidence level, the interval is too large to be informative…

Uncertainty under the BNN framework
-----------------------------------

We usually separate uncertainty in 3 categories : **_model uncertainty_**, **_inherent noise_**, and **_model misspecification_**. The first two are the most famous and are usually referred to as **_epistemic_** and **_aleatoric_** uncertainty.  
Model (or _e__pi__stemic_) uncertainty is the uncertainty about our model parameters. The more data you have, the more you can explain (ie “reduce”) this uncertainty. Noise (or _aleatoric_) uncertainty refers to the noise in the observations. If the noise uncertainty is constant for all samples, we call that **_homoscedastic aleatoric_** uncertainty. Otherwise, if some samples are more incertain than others, we will use the term **_heteroscedastic aleatoric_**. Noise uncertainty cannot be reduced with more data.

In the Uber paper, a third uncertainty is used : model misspecification. This uncertainty aims to “ _capture the uncertainty when predicting unseen samples with very different patterns from the training data set_”.

**— _Now why distinguish all of these uncertainties_** _?_

Well, precisely because some can be combated with more data (model/epistemic), and some cannot. Some researchers therefore argue that it is more relevant to focus on aleatoric uncertainty given it cannot be reduced even with more data ([Kendall & Gal, 2017](https://arxiv.org/pdf/1703.04977.pdf))

_In this article, we will only focus on the model uncertainty, to keep it simple._ _Uber uses different algorithms to evaluate the two other uncertainty types, but investigating them is beyond the scope of this article._

Getting and Preparing the data
==============================

The authors in the paper use 4 years of data over 8 cities in the US to train their model. 3 years are used for training and 1 year for testing. Unfortunately, Uber hasn’t released this data yet, but in order to reproduce results from their paper, we will use data available on the New York Open data [portal](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page). We selected three years of data, spanning from early 2015 to mid 2018. Data was resampled with a daily basis. Here is what the full data looks like

![](https://miro.medium.com/v2/resize:fit:1300/1*fCexDOMzzo6hH3YTFEtIZA.png)

Stationarity
------------

We can notice a couple of things which you should be familiar with if you’re used to analyzing time-series:

1.  We observe a clear upwards trend
2.  Mean and variance increase through time
3.  We also observe spikes which may be caused by external events (holidays and weather ?)

The first two bullet points are a sign that our series is clearly not stationary. The latter shows that we might need to incorporate external data to our series. Stationarity could be checked with an Augmented Dickey-Fuller test or a KPSS test.  So should we carry out these tests just like we did for ARIMA models ?— The answer is : not necessarily.

What we want, when we stationarize a series, is to have no change in mean and variance throughout time. This guarantees that if you take an N-points sample to make a forecast , and repeat this with another N-point different samples, then the relationship between your N-point samples and the N+1 point you are trying to predict are the same (ie they are drawn from the same distribution). If the mean and variance are not equal, then you are taking samples from different distributions to make forecasts, and your model will surely fail to generalize.

Unlike ARIMA, RNNs are able to model nonlinear relationships in the data.  
RNNs, and particularly LSTM and GRU, are able to capture long-term dependencies (provided you have sufficient amounts of data !).  Issues like de-trending and de-seasonalizing are therefore less important, but you should always ask yourself :

> Does the data I use for testing follow the same behavior (ie is from the same distribution) as the data I use for training ?

Holidays & Weather
------------------

Adding holidays indications is quite straightforward. We use the _holidays_ library in Python to get the holidays dates from 2015 to 2018.  
After adding a holiday boolean to our series, we still observe unexplained spikes…  
A quick look on the internet shows that a couple of them were actually days when New York was hit by extreme weather events such as blizzards and snow storms.  
We plot below the data with holidays marked in red and extreme weather events in green:

![](https://miro.medium.com/v2/resize:fit:1304/1*RNPu0k5h7Vt8163Y7llipg.png)

In our dataframe, we therefore have a “_counts_” column, a “_is_holiday_” column, and a “_is\_bad\_weather_” column. However, given we want to make predictions, we need to “anticipate” these dates as we would do for future predictions, we will therefore create two additional columns indicating that the next day is a holiday or that extreme weather is expected the next day :

weather = \[datetime.datetime.strptime(date, "%Y-%m-%d") for date in \['2018-01-04', '2018-03-21','2017-03-14','2017-02-09','2016-01-23'\]\]holidays = \[date for y in range(2015, 2019) for date, _ in sorted(holidays.US(years=y).items())\]df\['is_holiday'\] = np.where(df.index.isin(holidays), 1, 0)  
df\['bad_weather'\] = np.where(df.index.isin(weather), 1, 0)  
df\['next\_is\_holiday'\] = df.is_holiday.shift(-1)  
df\['next\_bad\_weather'\] = df.bad_weather.shift(-1)

Choice of window size & Backtesting
-----------------------------------

When doing time series forecasting you might hear about **_backtesting_**. Backtesting is a procedure used during training which consists in splitting your data into chunks, in an incremental manner. At each iteration, a chunk is used as your training set. You then try to predict 1 or more values ahead of your chunk. Two approaches can be used, _expanding_ and _sliding_ windows:

![](https://miro.medium.com/v2/resize:fit:1350/1*473sUTkrFKDQsN4n2pdQXA.png)

source: [https://www.datapred.com/blog/the-basics-of-backtesting](https://www.datapred.com/blog/the-basics-of-backtesting)

In our case study, the authors use samples consisting of 28-days sliding windows with step size equal to 1, used to predict the next value (1-step ahead forecast).

Logging and scaling
-------------------

In the paper, the authors start by taking the log of the data to “ _alleviate exponential effects_”. They then within each window substract the first value of the window to remove the trend and train the network on fluctuations with regard to the first value of the window(_eq(1)_). Other approaches can also be thought of, such as substracting the first value and dividing by the first value (_eq(2)_):

![](https://miro.medium.com/v2/resize:fit:686/1*1S6f9lsmo8O7Fsh5QEeRjg.png)

Building the Dataset
====================

Our dataset will be a generator yielding batches of sliding windows (each batch is the previous batch shifted of 1 value in the future). To follow the paper’s instructions, we will also substract, within each batch, the first value to all other values. Each batch is then split between 28-days samples and their 1-day targets.

Defining the model
==================

We will use PyTorch to define our model. A simple reason for that is that we will use dropout during inference and that it is simple to implement in PyTorch. We will start by using a simple LSTM network as defined in the paper: 1 LSTM layer with 128 units, 1 LSTM layer with 32 units, and a fully connected layer with 1 output. Dropout is added after each layer.

**_— An aparté on Dropout  
_**Dropout can be seen as a way of doing Bayesian inference (though there is still a debate around this). Technically, dropout is the process used in neural networks consisting in randomly dropping units (along with their connections).  
The fact of randomly turning neurons on and off is roughly equivalent to performing a sampling of a Bernoulli distribution, and therefore “simulates” the mechanics of a Bayesian Neural Network (where weights are distributions, and not single values). Applying dropout is a bit like if we were “sampling” from the network. And if we repeat this process several times duting inference, we will get different predictions with which we can estimate a distribution and eventually, uncertainty ! To sum up:

![](https://miro.medium.com/v2/resize:fit:1400/1*i3qGMn9ZiH6UXP03fKoMjg.png)

Let’s now define the model by adding dropout layers between each LSTM layer (notice how _train_ is set to True so that Dropout is used during training and testing)

Training
--------

We train for 5 epochs, with an Adam optimizer and learning rate set to 0.001, and batch_size of 1.

Results :

![](https://miro.medium.com/v2/resize:fit:1400/1*7sSbsmbJM5UYf-JmndUk0A.png)

Fitted values on the train set

Testing — 1 day forecast horizon
--------------------------------

One of the key interests of this paper is the estimation of uncertainty.

> Specifically, stochastic dropouts are applied after each hidden layer, and the model output can be approximately viewed as a random sample generated from the posterior predictive distribution. **As a result, the model uncertainty can be estimated by the sample variance of the model predictions in a few repetitions.**

The idea behind this paper is therefore to run several times the model with random dropout, which will yield different output values. We can then compute the empirical mean and variance of our outputs to get confidence intervals for each time step !

For each step, we predict 100 values. All values are different given we keep the dropout set. This allows us to simulate sampling from our network (in fact, Dropout is closely linked to Bayesian Neural Networks, given that by randomly disconnecting weights, it simulates a probability distribution). We take the average of these 100 values as our predicted mean, and the standard deviation of our 100 values which will be used for our confidence intervals. Assuming that our predictions are drawn from a normal distribution _N_(_μ_, _σ²_) — with a mean _μ_ equal to our empirical mean and a standard deviation _σ_ equal to our empirical standard error —, we can then estimate confidence intervals. In our case, it is given by :

![](https://miro.medium.com/v2/resize:fit:978/1*QIZGDhphldsz1c6R_774UA.png)

![](https://miro.medium.com/v2/resize:fit:1400/1*-_lbPeWvwKk1EieX3Tf9zQ.png)

Predicted values and uncertainty intervals

The prediction and the test curves seem to be quite close ! However, we need to find a metric to see how our model performs.

If we look at the empirical coverage of 95% predictive intervals (ie the number of true test values included in the predicted 95% confidence intervals), we obtain a value of 28.51%, far from the values obtained on the test set in the paper…When we take the 99% CI, this value goes up to 44% coverage. This is not a great result, but keep in mind our confidence interval is quite small given we only predict model uncertainty…

Testing — 7 days forecast horizon
---------------------------------

Results on a single day forecast are not so good for a prediction model, and still we are not asking a lot to our model as we are asking for a very short prediction horizon. What if we trained our model to predict larger horizon forecasts ?

Conclusion
==========

**LSTMs show interesting perspectives for modelling time series due to their ability to capture long time dependencies, but should be used only for large datasets**. Never expect your RNN to give good results on a 100-sample dataset ! The balance between the number of parameters in a neural network and the size of the data available is important to avoid overfitting. Nikolay Laptev, one of the authors of the Uber paper, concludes by saying that :

> **Classical models** are best for:  
> ○ Short or unrelated time-series  
> ○ Known state of world
> 
> **Neural Network** is best for:  
> ○ A lot of time-series  
> ○ Long time-series  
> ○ Hidden interactions  
> ○ Explanation is not important

**_Learn more about time series in_** [**_Part 1_**](https://towardsdatascience.com/time-series-in-python-exponential-smoothing-and-arima-processes-2c67f2a52788) **_and_** [**_Part 2_**](https://towardsdatascience.com/time-series-in-python-part-2-dealing-with-seasonal-data-397a65b74051) **_!_**