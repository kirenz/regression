(section-lm-lab)=
# Case study

The goal of this case study is to build a model for predicting the sale price of a house based on a particular home’s characteristics. 

We will start with data exploration and afterwards demonstrate how to build regression models with different Python modules:

1. [statsmodels](https://www.statsmodels.org/stable/index.html)
1. [scikit-learn](https://scikit-learn.org/stable/)


```{admonition} Resources
:class: tip
- Reading: [IMS 10](https://openintro-ims.netlify.app/model-application.html)
```

## Data exploration

In November of 2020, information on 98 houses in the Duke Forest neighborhood of Durham, NC were scraped from the real estate marketplace [Zillow](https://www.zillow.com). The homes were all recently sold at the time of data collection.

```{image} ../_static/img/duke-forest.png
:alt: datascience
:class: bg-primary mb-1
:width: 300px
:align: center
```

Let's start with our data exploration:

```{admonition} Jupyter notebook
:class: tip
- [Data exploration](https://kirenz.github.io/regression/docs/case-duke-exploration.html)
```

## Statsmodels

Linear regression with statsmodels:

```{admonition} Jupyter notebook
:class: tip
- [Model: statsmodels](https://kirenz.github.io/regression/docs/case-duke-statsmodel.html)
```

## Scikit-learn

Linear regression with scikit-learn:


```{admonition} Jupyter notebook
:class: tip
- [Model: scikit-learn](https://kirenz.github.io/regression/docs/case-duke-sklearn.html)	
```


<!--
## TensorFlow

To use TensorFlow, you first need to install some modules.

```{admonition} Jupyter notebook
:class: tip

- [TensorFlow installation tutorial](https://kirenz.github.io/codelabs/codelabs/tfx-install/#0)

- [Model: TensorFlow](https://kirenz.github.io/regression/docs/case-duke-tensorflow.html)
```
-->