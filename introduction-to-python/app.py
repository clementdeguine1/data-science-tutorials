"""
Introduction à pandas, visualisation du Titanic
Master 2 TSE — Raphaël Sourty

Adaptation Streamlit : le code d'origine du notebook est conservé,
la fonction Streamlit correspondante est ajoutée AU-DESSUS de chaque affichage.
Lancement : streamlit run app.py
Données    : https://www.kaggle.com/c/titanic/data
"""

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# # Introduction to pandas, visualizing Titanic
# ## Master 2 TSE
# #### Raphaël Sourty
st.markdown("# Introduction to pandas, visualizing Titanic")
st.markdown("## Master 2 TSE")
st.markdown("#### Raphaël Sourty")

# #### You can download the data here: https://www.kaggle.com/c/titanic/data
st.markdown("#### You can download the data here: https://www.kaggle.com/c/titanic/data")

df = pd.read_csv("train.csv")

st.dataframe(df.head())
df.head()

# Dictionnaire des variables
st.markdown(
    """
| Variable | Definition                                 | Key                                            |
|----------|--------------------------------------------|------------------------------------------------|
| survival | Survival                                   | 0 = No, 1 = Yes                                |
| pclass   | Ticket class                               | 1 = 1st, 2 = 2nd, 3 = 3rd                      |
| sex      | Sex                                        |                                                |
| Age      | Age in years                               |                                                |
| sibsp    | # of siblings / spouses aboard the Titanic |                                                |
| parch    | # of parents / children aboard the Titanic |                                                |
| ticket   | Ticket number                              |                                                |
| fare     | Passenger fare                             |                                                |
| cabin    | Cabin number                               |                                                |
| embarked | Port of Embarkation                        | C = Cherbourg, Q = Queenstown, S = Southampton |
"""
)

st.dataframe(df.isnull().sum())
df.isnull().sum()

# # Who are the survivors of the Titanic?
st.markdown("# Who are the survivors of the Titanic?")

# #### Survival rate
st.markdown("#### Survival rate")

st.dataframe(df.groupby("Survived")["PassengerId"].agg(["count"]))
df.groupby("Survived")["PassengerId"].agg(["count"])

fig, ax = plt.subplots(figsize=(10, 10))
df.groupby("Survived")["PassengerId"].agg(["count"]).reset_index().plot(
    x="Survived", y="count", kind="bar", ax=ax
)
st.pyplot(fig)
df.groupby("Survived")["PassengerId"].agg(["count"]).reset_index().plot(
    x="Survived", y="count", kind="bar", figsize=(10, 10)
)

# #### Survival rate depending on genre
st.markdown("#### Survival rate depending on genre")

st.dataframe(df.groupby(["Survived", "Sex"])["PassengerId"].agg(["count"]))
df.groupby(["Survived", "Sex"])["PassengerId"].agg(["count"])

st.dataframe(df.groupby(["Survived", "Sex"])["PassengerId"].agg(["count"]).unstack())
df.groupby(["Survived", "Sex"])["PassengerId"].agg(["count"]).unstack()

fig, ax = plt.subplots(figsize=(10, 10))
df.groupby(["Survived", "Sex"])["PassengerId"].count().unstack().plot(kind="bar", ax=ax)
st.pyplot(fig)
df.groupby(["Survived", "Sex"])["PassengerId"].count().unstack().plot(
    kind="bar", figsize=(10, 10)
)

# #### Survival rate depending on Ticket class
st.markdown("#### Survival rate depending on Ticket class")

st.dataframe(df.groupby(["Survived", "Pclass"])["PassengerId"].agg(["count"]))
df.groupby(["Survived", "Pclass"])["PassengerId"].agg(["count"])

st.dataframe(df.groupby(["Survived", "Pclass"])["PassengerId"].count().unstack())
df.groupby(["Survived", "Pclass"])["PassengerId"].count().unstack()

fig, ax = plt.subplots(figsize=(10, 10))
df.groupby(["Survived", "Pclass"])["PassengerId"].count().unstack().plot(kind="bar", ax=ax)
st.pyplot(fig)
df.groupby(["Survived", "Pclass"])["PassengerId"].count().unstack().plot(
    kind="bar", figsize=(10, 10)
)

# #### Survival rate depending on Age
st.markdown("#### Survival rate depending on Age")

df["generation"] = pd.cut(df["Age"], 8)

st.dataframe(pd.cut(df["Age"], 8))


st.dataframe(df.head())


fig, ax = plt.subplots(figsize=(10, 10))
df.groupby(["Survived", "generation"])["PassengerId"].count().unstack().plot(
    kind="bar", ax=ax
)
st.pyplot(fig)
df.groupby(["Survived", "generation"])["PassengerId"].count().unstack().plot(
    kind="bar", figsize=(10, 10)
)

# #### Survival rate depending on Fare
st.markdown("#### Survival rate depending on Fare")

df["fare_category"] = pd.cut(df["Fare"], 12)

st.dataframe(pd.cut(df["Fare"], 10))
pd.cut(df["Fare"], 10)

fig, ax = plt.subplots(figsize=(10, 10))
df.groupby(["Survived", "fare_category"])["PassengerId"].count().unstack().plot(
    kind="bar", ax=ax
)
st.pyplot(fig)
df.groupby(["Survived", "fare_category"])["PassengerId"].count().unstack().plot(
    kind="bar", figsize=(10, 10)
)

# #### What about correlations?
st.markdown("#### What about correlations?")

st.dataframe(df[["Survived", "Pclass", "Age", "Fare", "SibSp", "Parch"]].corr())
df[["Survived", "Pclass", "Age", "Fare", "SibSp", "Parch"]].corr()

st.dataframe(
    df[["Survived", "Pclass", "Age", "Fare", "SibSp", "Parch"]]
    .corr()
    .style.background_gradient(cmap="coolwarm")
)
df[["Survived", "Pclass", "Age", "Fare", "SibSp", "Parch"]].corr().style.background_gradient(
    cmap="coolwarm"
)

# ## Questions:
st.markdown("## Questions:")

st.markdown(
    """
##### Cabins on the port side have an even number and cabins on the starboard side have an odd number.

#### For example, cabin B57 is located on the starboard side.

#### Which side of the boat is better to be on?

#### The deck number of the boat is indicated on the ticket. Cabin B57 is located on deck B. Which deck is best to be on?

#### Where is the best place to be on the boat in general?

#### Is there a link between the number of parents/family on the boat and chances of survival?
"""
)

st.markdown(
    """
#### What is the typical profile of the person who will survive the shipwreck?

#### What is the typical profile of the person who will not survive the shipwreck?
"""
)



