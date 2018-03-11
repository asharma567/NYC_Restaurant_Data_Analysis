I have a barber/friend that's looking to open up a Kazakhstan based resturant and I mentioned I'd dome some analysis a few years back to see whether there was a relationship between health grades and cuisine type. 

Kazakhstan based cuisine isn't actually on here. I'd imagine he'd have to find sometihng similar and use that as his measuring stick.

At any rate, I think anyone trying to open up a business in the food industry in NYC should have a quick look at this first just to see what they're getting into.

In this [dataset](https://nycopendata.socrata.com/Health/DOHMH-New-York-City-Restaurant-Inspection-Results/xx67-kt59) about the restaurant evaluations in NYC. There's about 15 variables. 


[repo](https://github.com/asharma567/NYC_Restaurant_Data_Analysis) 




```python
import pandas as pd
import matplotlib as plt
import seaborn as sns
from scipy import stats
import numpy as np
from IPython.display import display, HTML
plt.style.use('ggplot')
%pylab inline
;
```

    Populating the interactive namespace from numpy and matplotlib


    WARNING: pylab import has clobbered these variables: ['plt']
    `%matplotlib` prevents importing * from pylab and numpy





    ''



## Exploring the data set "EDA"


```python
df = pd.read_csv('../data/DOHMH_New_York_City_Restaurant_Inspection_Results.csv')
```


```python
df[:2]
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CAMIS</th>
      <th>DBA</th>
      <th>BORO</th>
      <th>BUILDING</th>
      <th>STREET</th>
      <th>ZIPCODE</th>
      <th>PHONE</th>
      <th>CUISINE DESCRIPTION</th>
      <th>INSPECTION DATE</th>
      <th>ACTION</th>
      <th>VIOLATION CODE</th>
      <th>VIOLATION DESCRIPTION</th>
      <th>CRITICAL FLAG</th>
      <th>SCORE</th>
      <th>GRADE</th>
      <th>GRADE DATE</th>
      <th>RECORD DATE</th>
      <th>INSPECTION TYPE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td> 30075445</td>
      <td> MORRIS PARK BAKE SHOP</td>
      <td> BRONX</td>
      <td> 1007      </td>
      <td> MORRIS PARK AVE                               ...</td>
      <td> 10462</td>
      <td> 7188924968</td>
      <td> Bakery</td>
      <td> 03/03/2014</td>
      <td>   Violations were cited in the following area(s).</td>
      <td> 10F</td>
      <td> Non-food contact surface improperly constructe...</td>
      <td>   Not Critical</td>
      <td>  2</td>
      <td>   A</td>
      <td> 03/03/2014</td>
      <td> 11/15/2014</td>
      <td>    Cycle Inspection / Initial Inspection</td>
    </tr>
    <tr>
      <th>1</th>
      <td> 30075445</td>
      <td> MORRIS PARK BAKE SHOP</td>
      <td> BRONX</td>
      <td> 1007      </td>
      <td> MORRIS PARK AVE                               ...</td>
      <td> 10462</td>
      <td> 7188924968</td>
      <td> Bakery</td>
      <td> 10/10/2013</td>
      <td> No violations were recorded at the time of thi...</td>
      <td> NaN</td>
      <td>                                               NaN</td>
      <td> Not Applicable</td>
      <td>NaN</td>
      <td> NaN</td>
      <td>        NaN</td>
      <td> 11/15/2014</td>
      <td> Trans Fat / Second Compliance Inspection</td>
    </tr>
  </tbody>
</table>
</div>



### What are some other stats we can take into consideration? How is the data distributed?


```python
display(df.describe())
```


<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CAMIS</th>
      <th>ZIPCODE</th>
      <th>SCORE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>   523967.000000</td>
      <td> 523965.000000</td>
      <td> 490100.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td> 41969457.460018</td>
      <td>  10661.922953</td>
      <td>     21.852934</td>
    </tr>
    <tr>
      <th>std</th>
      <td>  2471428.753225</td>
      <td>    599.848347</td>
      <td>     14.892681</td>
    </tr>
    <tr>
      <th>min</th>
      <td> 30075445.000000</td>
      <td>   7005.000000</td>
      <td>     -1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td> 41024211.000000</td>
      <td>  10019.000000</td>
      <td>     12.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td> 41393597.000000</td>
      <td>  10465.000000</td>
      <td>     19.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td> 41611217.000000</td>
      <td>  11229.000000</td>
      <td>     27.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td> 50017476.000000</td>
      <td>  11697.000000</td>
      <td>    156.000000</td>
    </tr>
  </tbody>
</table>
</div>


### What do the rows look like?


```python
df[['GRADE','CUISINE DESCRIPTION' ,'DBA']][:10]
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>GRADE</th>
      <th>CUISINE DESCRIPTION</th>
      <th>DBA</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>   A</td>
      <td> Bakery</td>
      <td> MORRIS PARK BAKE SHOP</td>
    </tr>
    <tr>
      <th>1</th>
      <td> NaN</td>
      <td> Bakery</td>
      <td> MORRIS PARK BAKE SHOP</td>
    </tr>
    <tr>
      <th>2</th>
      <td>   A</td>
      <td> Bakery</td>
      <td> MORRIS PARK BAKE SHOP</td>
    </tr>
    <tr>
      <th>3</th>
      <td>   A</td>
      <td> Bakery</td>
      <td> MORRIS PARK BAKE SHOP</td>
    </tr>
    <tr>
      <th>4</th>
      <td> NaN</td>
      <td> Bakery</td>
      <td> MORRIS PARK BAKE SHOP</td>
    </tr>
    <tr>
      <th>5</th>
      <td> NaN</td>
      <td> Bakery</td>
      <td> MORRIS PARK BAKE SHOP</td>
    </tr>
    <tr>
      <th>6</th>
      <td> NaN</td>
      <td> Bakery</td>
      <td> MORRIS PARK BAKE SHOP</td>
    </tr>
    <tr>
      <th>7</th>
      <td> NaN</td>
      <td> Bakery</td>
      <td> MORRIS PARK BAKE SHOP</td>
    </tr>
    <tr>
      <th>8</th>
      <td> NaN</td>
      <td> Bakery</td>
      <td> MORRIS PARK BAKE SHOP</td>
    </tr>
    <tr>
      <th>9</th>
      <td> NaN</td>
      <td> Bakery</td>
      <td> MORRIS PARK BAKE SHOP</td>
    </tr>
  </tbody>
</table>
</div>




```python
def df_to_perc_breakouts_per_cuisine(input_df):
    '''
    INPUT: DF with each restuarant and grade listed (Raw Data)
    OUTPUT: DF with percentage composition by cuisine type
    '''
    output_df = input_df[['CUISINE DESCRIPTION','GRADE']].groupby('CUISINE DESCRIPTION').count()
    
    #function used to normalize grades by percentage
    perc_computation_function = \
    lambda x: x[x['GRADE'] == str(letter)].count() / x['GRADE'].count()

    for letter in input_df['GRADE'].unique():
        temp_df = input_df.groupby('CUISINE DESCRIPTION')
        temp_df = temp_df.apply(perc_computation_function)
        output_df[str(letter)] = temp_df['GRADE']    
    return output_df
```

## Preprocess the data


```python
# transforming the raw data the percentage versus counts
df_cuisine_grades_by_composition = df_to_perc_breakouts_per_cuisine(df)

del df_cuisine_grades_by_composition['GRADE']
del df_cuisine_grades_by_composition['nan']

# now we're ready to plot it
df_cuisine_grades_by_composition[:2] 
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>Z</th>
      <th>C</th>
      <th>P</th>
      <th>Not Yet Graded</th>
    </tr>
    <tr>
      <th>CUISINE DESCRIPTION</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Afghan</th>
      <td> 0.689655</td>
      <td> 0.155172</td>
      <td> 0.000000</td>
      <td> 0.112069</td>
      <td> 0.043103</td>
      <td> 0.000000</td>
    </tr>
    <tr>
      <th>African</th>
      <td> 0.515957</td>
      <td> 0.268617</td>
      <td> 0.027926</td>
      <td> 0.156915</td>
      <td> 0.023936</td>
      <td> 0.006649</td>
    </tr>
  </tbody>
</table>
</div>




```python
def plot_stacked_bar(df_to_plot, label, color_map = 'YlOrBr'):
    '''
    INPUT: DF, label(string) for the x-axis to be displayed at the top
    OUTPUT: Stacked Bar Chart
    '''
    # create a figure of given size
    fig = plt.figure(figsize=(25,25))

    # add a subplot
    ax = fig.add_subplot(111)

    # set color transparency (0: transparent; 1: solid)
    a = 0.8

    # set x axis label on top of plot, set label text
    xlab = label
    ax.set_xlabel(xlab, fontsize=20, alpha=a, ha='left')
    ax.xaxis.set_label_coords(0, 1.04)

    # position x tick labels on top
    ax.xaxis.tick_top()

    # remove tick lines in x and y axes
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticks_position('none')

    # remove grid lines (dotted lines inside plot)
    ax.grid(False)

    # Remove plot frame
    ax.set_frame_on(False)

    # using the actual data to plot
    df_to_plot[::-1].plot(
    ax=ax, 
    kind='barh', 
    alpha=a, 
    edgecolor='w',
    fontsize=12, 
    grid=True, 
    width=.8, 
    stacked=True,
    cmap=get_cmap(color_map)
    )

    # remove weird dotted line on axis
    ax.lines[0].set_visible(False)

    # multiply xticks by format into pct
    fmt = '%.0f%%' # Format you want the ticks, e.g. '40%'
    xticks = matplotlib.ticker.FormatStrFormatter(fmt)
    ax.xaxis.set_major_formatter(xticks)
    ax.xaxis.set_ticks(ax.xaxis.get_majorticklocs()[:-1])

    plt.legend(prop={'size':20}, frameon=False, fancybox=None)
    plt.tight_layout()
    plt.show();
```

## Current grade by type of cuisine
* I found a stacked bar chart to be the best solution to displaying each cuisine type and the distribution grades recieved by each restaurant per cuisine. e.g.
    * Afghan: 69% recieved A's, 15.5% B's, ...

    
#### Additional modifications for the future:
- Adding counts of resutaurants per cuisine type as a 3rd dimension
    * Scaling each individual bar appropriately or
    * Including a count on the right or left side



```python
plot_stacked_bar(df_cuisine_grades_by_composition * 100, 'Percentage by grade')
```


![png](output_16_0.png)



```python
def plot_dist(df_to_plot, series_name, lab):
    '''
    INPUT: DF, name of the column or series, label for the x-axis
    OUTPUT: distribution of violation codes sorted by popularity (Bar Chart)
    '''

    # create a figure of given size
    fig = plt.figure(figsize=(10,20))

    # add a subplot
    ax = fig.add_subplot(111)
    
    # set color transparency (0: transparent; 1: solid)
    a = 0.8

    # set x axis label on top of plot, set label text
    ax.xaxis.set_label_position('bottom')
    xlab = lab
    ax.set_xlabel(xlab, fontsize=20, alpha=a, ha='left')
    ax.xaxis.set_label_coords(0, 1.04)

    # position x tick labels on top
    ax.xaxis.tick_top()

    # remove tick lines in x and y axes
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticks_position('none')

    # remove grid lines (dotted lines inside plot)
    ax.grid(False)

    # remove plot frame
    ax.set_frame_on(False)

    # labeling 
    labels = df_to_plot[series_name].value_counts().index
    labels_cnv = [str(labels[i]) for i, acct in enumerate(labels)]

    # using the actual data to plot
    df_to_plot[series_name].value_counts()[::-1].plot(
    ax=ax, 
    kind='barh', 
    color=(0.9698, 0.6378, 0.3373), 
    alpha=a, 
    edgecolor='w',
    label=labels_cnv, 
    fontsize=12, 
    grid=True, 
    width=.8
    )

    # remove weird dotted line on axis
    ax.lines[0].set_visible(False)
    
    plt.tight_layout()
    plt.show();
```

## A distribution of violation code, to see which violations are most popular

* Sorted by popularity (or highest count)


```python
plot_dist(df, 'VIOLATION CODE', 'Count of violation codes')
```


![png](output_19_0.png)



```python
def code_to_description_lookup(raw_df, violation_code_str):
    '''
    INPUT: raw df, violation code eg '10F' 
    OUTPUT: description (text), count of the code (int)
    
    Purpose: extra tool to look-up descriptions given the violation code
    '''
    df_dictionary_violation_code = raw_df[['VIOLATION CODE', 'VIOLATION DESCRIPTION']].groupby(['VIOLATION CODE', 'VIOLATION DESCRIPTION']).size()
    return df_dictionary_violation_code.ix[violation_code_str]

code_to_description_lookup(df, '10F')
```




    VIOLATION DESCRIPTION
    Non-food contact surface improperly constructed. Unacceptable material used. Non-food contact surface or equipment improperly maintained and/or not properly sealed, raised, spaced or movable to allow accessibility for cleaning on all sides, above and underneath the unit.    66006
    dtype: int64




```python
def print_top_k_violation_codes(raw_df, k):
    '''
    INPUT: DF, top k eg top 5 violations codes; k = 5
    OUTPUT: None, prints to the screen (stdout)
    
    Purpose: Offer a ranked list of the violation codes with descriptions
    '''
    dfn = raw_df[['VIOLATION CODE', 'VIOLATION DESCRIPTION']].groupby(['VIOLATION CODE', 'VIOLATION DESCRIPTION']).agg(np.size)

    dfn.name = 'code'
    dfn.sort('code', ascending=False)    
    for rank, code_desc in enumerate(zip(dfn[:k].index ,dfn[:k])):
        
        # unpacking nested tuples
        code, desc = code_desc[0]
        
        print str(rank + 1) + ')'
        print code, code_desc[1]
        
        # formatting to match length of line
        print '-' * (len(code + str(code_desc[1])) + 1) 
        print desc
        print '=' * 65

print_top_k_violation_codes(df, 4)        
```

    1)
    10F 132012
    ----------
    Non-food contact surface improperly constructed. Unacceptable material used. Non-food contact surface or equipment improperly maintained and/or not properly sealed, raised, spaced or movable to allow accessibility for cleaning on all sides, above and underneath the unit.
    =================================================================
    2)
    08A 98962
    ---------
    Facility not vermin proof. Harborage or conditions conducive to attracting vermin to the premises and/or allowing vermin to exist.
    =================================================================
    3)
    02G 89148
    ---------
    Cold food item held above 41º F (smoked fish and reduced oxygen packaged foods above 38 ºF) except during necessary preparation.
    =================================================================
    4)
    04L 74908
    ---------
    Evidence of mice or live mice present in facility's food and/or non-food areas.
    =================================================================


#### Setting up Hypothesis Test: Chi-square goodness of fit

* The null states H<sub>0</sub>: there is no relationship between cuisine type and restaurant grade

_Assuming no relationship between cuisine and restaurant grade the following proportions should make sense regardless of cuisine_



```python
df_proportions_with_ct = df_to_perc_breakouts_per_cuisine(df)[:]

# no need for nans
del df_proportions_with_ct['nan']

# Proportions -> observed counts per cuisine
for col in df_proportions_with_ct.columns.tolist():
    if col == 'GRADE': continue
    df_proportions_with_ct[col] *= df_proportions_with_ct['GRADE'] 

# Setting up H_0
total_distribution_proportion = {}
for col in df_proportions_with_ct.columns.tolist():
    if col == 'GRADE': continue
    total_distribution_proportion[col] = df_proportions_with_ct[col].sum() / df_proportions_with_ct['GRADE'].sum()

# Display proportions in table from of aggregate data
pd.DataFrame(total_distribution_proportion, index=[0])
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>Not Yet Graded</th>
      <th>P</th>
      <th>Z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td> 0.680488</td>
      <td> 0.212704</td>
      <td> 0.070549</td>
      <td> 0.007381</td>
      <td> 0.00944</td>
      <td> 0.019438</td>
    </tr>
  </tbody>
</table>
</div>




```python
#used to randomly sample from our set of cuisines
random_observation_seed = np.random.randint(1, df_proportions_with_ct.shape[0] + 1)

# Generate expected frequencies
expected = {}
rand_obs = df_proportions_with_ct.ix[random_observation_seed,:].to_dict()
rand_obs_cuisine = df_proportions_with_ct.ix[random_observation_seed,:].name

for key, val in total_distribution_proportion.iteritems():
    expected[key] = rand_obs['GRADE'] * val

rand_obs.pop('GRADE')

df_expected = pd.DataFrame().from_dict(expected, orient='index')
df_expected.columns = ['Expected frequency']
display(df_expected)

print 

df_rand_obs = pd.DataFrame().from_dict(rand_obs, orient='index')
df_rand_obs.columns = ['Random observation frequency']
display(df_rand_obs)


# We can safely reject the H_0, this p-value is too extreme 
# to happen by random chance.
chi, p_val = stats.chisquare(rand_obs.values(), expected.values())

df_random_obs_stats = pd.DataFrame().from_dict({
    'p-value: ': p_val, 
    'random cuisine type:': rand_obs_cuisine, 
    'chi-squared test statistic:' : chi
}, orient='index')

df_random_obs_stats.columns = ['H Test Stats']
display(df_random_obs_stats)

```


<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Expected frequency</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A</th>
      <td> 39.468284</td>
    </tr>
    <tr>
      <th>C</th>
      <td>  4.091831</td>
    </tr>
    <tr>
      <th>B</th>
      <td> 12.336860</td>
    </tr>
    <tr>
      <th>Not Yet Graded</th>
      <td>  0.428069</td>
    </tr>
    <tr>
      <th>P</th>
      <td>  0.547548</td>
    </tr>
    <tr>
      <th>Z</th>
      <td>  1.127408</td>
    </tr>
  </tbody>
</table>
</div>


    



<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Random observation frequency</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A</th>
      <td> 50</td>
    </tr>
    <tr>
      <th>C</th>
      <td>  0</td>
    </tr>
    <tr>
      <th>B</th>
      <td>  8</td>
    </tr>
    <tr>
      <th>Not Yet Graded</th>
      <td>  0</td>
    </tr>
    <tr>
      <th>P</th>
      <td>  0</td>
    </tr>
    <tr>
      <th>Z</th>
      <td>  0</td>
    </tr>
  </tbody>
</table>
</div>



<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>H Test Stats</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>p-value: </th>
      <td> 0.0615444</td>
    </tr>
    <tr>
      <th>random cuisine type:</th>
      <td>     Czech</td>
    </tr>
    <tr>
      <th>chi-squared test statistic:</th>
      <td>   10.5297</td>
    </tr>
  </tbody>
</table>
</div>


### Is there a statistically significant relationship between type of cuisine and restaurant grade?
* With a critical region defined as **$\alpha $ = 0.01** i.e. 99% confidence interval, we can safely reject the H<sub>0</sub>: There is no relationship between cuisine type and restaurant grade. The alternative being **H<sub>A</sub>: There is a relationship between cuisine type and restaurant grade.**

### What test did we perform?
* $\chi^2$ Chi-Squared test: Goodness of fit. It's used to find the significance of association in two categorical variables. In our case these variables are cuisine type (Indian, American, ..) and restaurant grade (A, B, ..).

### What does this finding mean in this context?
* There is a statistically significant relationship between the type of cuisine and the grade the restaurant received. Given the p-value, **it is highly unlikely that a relationship between grade and cuisine type doesn't exist.**

### Based on findings, what recommendations would you give the DOHMH to prioritize inspections?
* At a glance, these findings might suggest some cuisine types are given an unfair advantage versus other types, hence recommending that a fair inspection be given to all restaurants regardless of the type of cuisine. Though, that may very well not be the case -- cleanliness of a restaurant has more to do with the management. Specifically, the management of particular cuisine type is likely the culprit. As a hypothetical example, maybe it is true that typically the restaurant management of Indian cuisines tend to be a bit more unkept, say versus American. **In which case DOHMH should focus efforts on improving restaurant environments of particular cuisine types that typically seem to be falling short.**


* On the other end, maybe an inspection favors certain cuisines merely by design. It's entirely possible that it was designed at a time where there wasn't such a diverse selection of cuisine types, hence our findings might be suggesting that **inspection methods need to be revised such that it's custom tailored (or tweaked) to the subject cuisine type.**


* Violation codes regarding vermin control seem to be among the more popular. This may be a potential low hanging solution for the DOHMH and it speaks more to the city as opposed to restaurant owners.


### Next step

_It may be prudent to look for inferences in the corpus of popular violation code descriptions. As an initial technique we could use a simple bag-of-words model to look at popular n-grams being used. We can also use a latent topic modeler to look potential latent topics within the descriptions._

