# Data Lakehouse 

According to [LaPlante (2021)](https://pages.databricks.com/rs/094-YMS-629/images/9781492087922_FinalDeliverable.pdf) a lakehouse combines elements of data lakes and data warehouses to build something new: 

- Lakehouses have similar data structures and data management features to data warehouses, but use the low-cost, flexible storage of data lakes. 

Once you combine a data lake along with analytical infrastructure, the entire infrastructure can be called a data lakehouse (Inmon, Levins & Srivastava, 2021).

## Overview

The article "Evolution to the Data Lakehouse" by Bill Inmon and Mary Levins (2021) provides an overview about the main challenges of current data architectures and how data lakhouse architectures address them:

![Lakehouse](https://databricks.com/wp-content/uploads/2021/05/edl-blog-img-5-832x1024.png)

- [Inmon, B. & Levins, M. (2021). Evolution to the Data Lakehouse. Databricks Blog.](https://databricks.com/de/blog/2021/05/19/evolution-to-the-data-lakehouse.html)

After reading this article, you should be able to answer the following questions:

```{admonition} Questions
:class: tip
- What are the challenges with current data architecture
- How does the data lakehouse architecture solve the key challenges of current data architectures? Describe the key features. 
```

Furthermore, make yourself familiar with the so called lambda-architecture, which many current architecures us:

![](https://databricks.com/wp-content/uploads/2018/12/hadoop-architecture.jpg)

- [Lambda architecture](ttps://databricks.com/de/glossary/lambda-architecture)

```{admonition} Questions
:class: tip
- What are the 3 layers of the lambda architecture?
```


## Architecture

The book "Building the Data Lakehouse" from Inmon, Levins and Srivastava provides a high level overview about important concepts of the lakehouse architecture:

- [Inmon, B., Levins, M. & Srivastava, R. (2021). Building the Data Lakehouse. Technics Publications, NY.](https://drive.google.com/file/d/1bURUyz-zSSCdT_k-MNjuFO0Gbq4vDkvt/view?usp=sharing)

Read the following paragraphs:

- The data lake (p19-22)
- Current data architecture challenges (p22-23)
- Emergence of data lakehouses (p23-29)
- Different Types of Data in the Data Lakehouse (p39-50)
- The Open Environment (p53-70)
- The Analytical Infrastructure for the Data Lakehouse (p87-104)
- Data Lakehouse Housekeeping™ (p135-168)
- Purpose of data governance (p229-243)