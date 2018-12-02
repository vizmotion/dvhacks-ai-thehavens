# dvhacks-ai-thehavens

# TheHavens.ai

AI-driven understanding of databases.
<p align="center">
  <img src="https://github.com/chaitanya1123/dvhacks-ai-thehavens/blob/master/figures/KnowledgeGraph.png" width="400">
</p>

## Summary of the solution

As a data science we often are asked to analyze data that has no documentation, business rules encoded in the system, and table/field relationships not well defined. Hours are spent working through the meaning of the data, asking business people open ended questions and creating our own data model only to find out at the very end of the analysis that we have made erroneous or incorrect assumptions.

Now through the power of AI we are able to take these legacy databases and automatically develop a relationship graph that allows more directed questions and discover to find hidden or lost meaning to the database.

The idea is to take an undocumented database and create a graph describing table and field relationships. The following steps are used to develop this graph:
1. Connect to a database with multiple undocumented tables
2. Using our table connection algorithm (see paper xxx) we create a highly likely table primary and foreign key connection.
3. Using our feature selection model we build a highly correlated mapping between tables and fields
4. Using our graphing system we take that correlation model and build a visual graphical image to allow interactive investigation of these relationships.

## The Problem:

Take an average database for a medium to large corporation: 114 tables with a total
of 2600 fields, with an average of 26 fields per table. If we are lucky, there is a
small set of key fields that can be used to join the tables to create a super-table
with the maximum information content while minimizing redundancy. If we are unlucky -and the planets do not align often- we don't have the key set of fields before hand and we need to discover them.

Joining a set of 10 tables with 5 fields will yield to order 5^10 ~10 million possibilities worst case scenario. So brute force is out of the question. So we need a smart way to learn the relevant fields across all databases that can be used to form the super table with maximal information content.

That's what TheHavens.ai can do for you: a AI-driven, smart way to find the super table with maximal information while optimizing cost and computer power.  



## Structure of the solution
1. Data Connector: Plugging Architecture to connect to arbitrary number of MySQL databases.
2. Table Relationship Identifier: Defining features and meta-features, to compute inclusion dependencies and classifying them as foreign key constrains.   [here](https://www.researchgate.net/publication/221035501)
3. Model Feature Selector: Using Genetic Algorithms we can find the set of features that can be selected to maximize the understanding of the database and enable the data scientist to focus on the most relevant features.
4. Descriptive Analytics: The insights of the solution will be summarized as knowledge graphs, description of the relationships between the features.

<p align="center">
  <img src="https://github.com/chaitanya1123/dvhacks-ai-thehavens/blob/master/figures/UI_screenshot.png" width="400">
</p>

## Feature Roadmap
1. Basic Connection/Map Schema
  - Core
    - Correlation
    - Model Search
    - Column Selection
    - Model Building
  - Document Generation

2. UI
  - Select Data Source
  - View Relationships
  - Evaluate models & connections
  - Specific model exploration
