# MLOps intro

 MLOps is a methodology for [ML engineering](https://cloud.google.com/certification/machine-learning-engineer) that unifies *ML system development* (the ML element) with *ML system operations* (the Ops element). In particular, it provides a set of standardized processes and technology capabilities for building, deploying, and operationalizing ML systems rapidly and reliably {cite:ps}`salama_practitioners_2021`.

## Data-centric AI

To get a first overview about common MLOps related issues, watch this video from AI pioneer Andrew Ng: "A Chat with Andrew on MLOps: From Model-centric to Data-centric AI"

<iframe width="560" height="315" src="https://www.youtube.com/embed/06-AZXmwHjo" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

- [Slides](https://drive.google.com/file/d/1X7qquuEIde1jkNXGhjP4xRCVHCwSWtKt/view?usp=sharing) from the talk.  

After watching this video, you should be able to answer the following questions:  

```{admonition} Questions
:class: tip
- Describe the lifecycle of an ML project.
- What is the difference between a model-centric vs data-centric view?
- Describe MLOps' most important task.
```

## Challenges

Despite the growing recognition of AI/ML as a crucial pillar of digital transformation, successful deployments and effective operations are a bottleneck for getting value from AI {cite:ps}`salama_practitioners_2021`:  

- Only one in two organizations has moved beyond pilots and proofs of concept.  

- 72% of a cohort of organizations that began AI pilots before 2019 have not been able to deploy even a single application in production.  

- Algorithmia’s survey of the state of enterprise machine learning found that 55% of companies surveyed have not deployed an ML model.  

```{note}
Most models don’t make it into production, and if they do, they break because they fail to adapt to changes in the environment.
```

Watch this presentation from Nayur Khan, global head of technical delivery at McKinsey, to get a first understanding of common MLOps related challenges:

<iframe width="560" height="315" src="https://www.youtube-nocookie.com/embed/M1F0FDJGu0Q" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

After watching this video, you should be able to answer the following questions:

```{admonition} Questions
:class: tip
- Describe 4 typical challenges when creating machine learning products.
- Reusability concerns within a codebase: Explain a common way to look at what code is doing in a typical ML project.
- What kind of problems does the open-source framework Kedro solve and where does Kedro fit in the MLOps ecosystem?
```

## Components

Next, you'll get an overview about some of the primary components of MLOps. "An introduction to MLOps on Google Cloud" by Nate Keating:

<iframe width="560" height="315" src="https://www.youtube-nocookie.com/embed/6gdrwFMaEZ0" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

After watching this video, you should be able to answer the following questions:

```{admonition} Questions
:class: tip
- Describe the challenges of current ML systems (where are teams today)? 
- What are the components of the ML solution lifecycle? 
- Explain the steps in an automated E2E pipeline.
```

## Framework

To get a deeper understanding about the complete MLOps framework, read the following resources provided by Google:

- [MLOps: Continuous delivery and automation pipelines in machine learning](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)

- [Practitioners guide to MLOps: A framework for continuous delivery and automation of machine learning](https://drive.google.com/file/d/1ni3QDS4Y40MnWTr1ko83FoZ34JZzRq-U/view?usp=sharing) from {cite:t}`salama_practitioners_2021`.

After reading the reports, you should be able to answer the following questions:

```{admonition} Questions
:class: tip
- Describe the difference betweeen DevOps versus MLOps
- Name and explain the steps for developing ML models
- Describe the three different MLOps maturity levels. In particular, explain the concepts of:
    - Data and model validation
    - Dataset and feature repository
    - Metadata management
    - ML pipeline triggers
    - Continuous training
    - Model deployment 
    - Prediction serving
    - Continuous monitoring
    - Model governance
```

