# Personality-prediction
> From the given database Find out the personality using this personality traits. 

## Let's get started
[![](https://img.shields.io/badge/author-@SagarDarji-blue.svg?style=flat)](www.linkedin.com/in/sagar-darji-7b7011165?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base_contact_details%3BMxOAgkrVTVyw0F4O977G1w%3D%3D)

### Applications in psychology:
Factor analysis has been used in the study of human intelligence and human personality as a method for comparing the outcomes of (hopefully) objective tests and to construct matrices to define correlations between these outcomes, as well as finding the factors for these results. The field of psychology that measures human intelligence using quantitative testing in this way is known as psychometrics (psycho=mental, metrics=measurement). 

### Advantages:
- Offers a much more objective method of testing traits such as intelligence in humans
- Allows for a satisfactory comparison between the results of intelligence tests
- Provides support for theories that would be difficult to prove otherwise

## Now understand and implement the code 
> Import all library which we needed to perform this `python code`
```#Librerias
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
```

```
#Data
df = pd.read_csv("responses.csv")
df.shape
```
MUSIC PREFERENCES (19) 0:19

MOVIE PREFERENCES (12) 19:31

HOBBIES & INTERESTS (32) 31:63

PHOBIAS (10) 63:73

HEALTH HABITS (3) 73:76

PERSONALITY TRAITS, VIEWS ON LIFE & OPINIONS (57) 76:133

SPENDING HABITS (7) 133:140

DEMOGRAPHICS (10 ) 140:150

We will take only: PERSONALITY TRAITS, VIEWS ON LIFE & OPINIONS (57) 76:133


```
df = df.iloc[:, 76:133]
df.head(5)
```

#dataset
<h3>1. Prepare the Data</h3>

```#Drop NAs
df = df.dropna()#...............................................................................................#Encode categorical data
from sklearn.preprocessing import LabelEncoder

df = df.apply(LabelEncoder().fit_transform)
df
```

#data sheet
<h3>2. Choose the Factors</h3>

```pip install factor_analyzer 
```
dd

```#Try the model with all the variables 
from factor_analyzer import FactorAnalyzer         # pip install factor_analyzer 
fa = FactorAnalyzer(rotation="varimax")
fa.fit(df) 

# Check Eigenvalues
ev, v = fa.get_eigenvalues()
ev

# Create scree plot using matplotlib
plt.scatter(range(1,df.shape[1]+1),ev)
plt.plot(range(1,df.shape[1]+1),ev)
plt.title('Scree Plot')
plt.xlabel('Factors')
plt.ylabel('Eigenvalue')
plt.grid()
plt.show()
```
#scree graph
As you can see the most usefull factors for explain the data are between 5-6 until falling significantly.

We will fit the model with 5 Factors:

```#Factor analysis with 5 fators
fa = FactorAnalyzer(5, rotation="varimax")
fa.fit(df)
AF = fa.loadings_
AF = pd.DataFrame(AF)
AF.index = df.columns
AF
```
#data sheet
```#Get Top variables for each Factor 
F = AF.unstack()
F = pd.DataFrame(F).reset_index()
F = F.sort_values(['level_0',0], ascending=False).groupby('level_0').head(5)    # Top 5 
F = F.sort_values(by="level_0")
F.columns=["FACTOR","Variable","Varianza_Explica"]
F = F.reset_index().drop(["index"],axis=1)
F
```

#data sheet
```#Show the Top for each Factor 
F = F.pivot(columns='FACTOR')["Variable"]
F.apply(lambda x: pd.Series(x.dropna().to_numpy()))
```

#data sheet
FACTOR 1: Energy levels, Number of friends, Socializing...

Could be: Extraversion

FACTOR 2: Self-ciricism, Fake, Loneliness...

Looks very similar to "Neuroticism"

Factor 3: Thinking ahead, Prioritising workload...

very similar to "Conscientiousness"

Factor 4: Children, God, Finding lost valuables

This factor could be something like "religious" or "conservative", maybe have lowest scores of a "Openness" in Big Five model.

Factor 5: Appearence and gestures, Mood swings

Mmmm it could be "Agreeableness". What do you think it could be represent?
<h3>Conclusion</h3>
The first three Factors are very clear: Extraversion, Neuroticism and Conscientiousness. The other two not to much. Anyway is a very interesting approximation

Maybe doing first a PCA for remove hight correlate variables like "God" and "Final judgement"could help.

What do you think?

Thanks you!
