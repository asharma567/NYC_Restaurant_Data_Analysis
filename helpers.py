import pandas as pd
import googlemaps
from scipy import stats
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

def multithread_map(fn, work_list, num_workers=50):
    
    '''
    spawns a threadpool and assigns num_workers to some 
    list, array, or any other container. Motivation behind 
    this was for functions that involve scraping.
    '''
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        return list(executor.map(fn, work_list))


def make_expected_frequency_H0(df):
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

    return total_distribution_proportion, df_proportions_with_ct


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

def missing_values_finder(df):
    '''
    finds missing values in a data frame returns to you the value counts
    '''
    import pandas as pd
    missing_vals_dict= {col : df[col].dropna().shape[0] / float(df[col].shape[0]) for col in df.columns}
    output_df = pd.DataFrame().from_dict(missing_vals_dict, orient='index').sort_index()
    return output_df


def get_lat_lon(str_, stop_words=None):
    geocode_result = []
    gmaps = googlemaps.Client(key='AIzaSyCol8kK-GVXAIukXhICNXuaBIgqzENNp7I')
    
    try:

        if stop_words:
            str_ = ' '.join([word for word in str_.split() if word not in set(stop_words)])

        geocode_result = gmaps.geocode(str_)

        return geocode_result[0]['geometry']['location']
    except:
        print (geocode_result)
        return None

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
#     ax.lines[0].set_visible(False)

    # multiply xticks by format into pct
    fmt = '%.0f%%' # Format you want the ticks, e.g. '40%'
    xticks = matplotlib.ticker.FormatStrFormatter(fmt)
    ax.xaxis.set_major_formatter(xticks)
    ax.xaxis.set_ticks(ax.xaxis.get_majorticklocs()[:-1])

    plt.legend(prop={'size':20}, frameon=False, fancybox=None)
    plt.tight_layout()
    plt.show();

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

def code_to_description_lookup(raw_df, violation_code_str):
    '''
    INPUT: raw df, violation code eg '10F' 
    OUTPUT: description (text), count of the code (int)
    
    Purpose: extra tool to look-up descriptions given the violation code
    '''
    df_dictionary_violation_code = raw_df[['VIOLATION CODE', 'VIOLATION DESCRIPTION']].groupby(['VIOLATION CODE', 'VIOLATION DESCRIPTION']).size()
    return df_dictionary_violation_code.ix[violation_code_str]


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
        
        print (str(rank + 1) + ')')
        print (code, code_desc[1])
        
        # formatting to match length of line
        print ('-' * (len(code + str(code_desc[1])) + 1) )
        print (desc)
        print ('=' * 65)


def show_chi_squared_test_results(total_distribution_proportion, df_proportions_with_ct, cuisine_type_str=None):
    if cuisine_type_str:
        random_observation_seed = [(idx, item) for idx, item in enumerate(df_proportions_with_ct.index) \
              if item == cuisine_type_str ][0][0]
    else:
        #used to randomly sample from our set of cuisines
        random_observation_seed = np.random.randint(1, df_proportions_with_ct.shape[0] + 1)

    # Generate expected frequencies
    expected = {}
    rand_obs = df_proportions_with_ct.ix[random_observation_seed,:].to_dict()
    rand_obs_cuisine = df_proportions_with_ct.ix[random_observation_seed,:].name

    for key, val in total_distribution_proportion.items():
        expected[key] = rand_obs['GRADE'] * val

    rand_obs.pop('GRADE')

    df_expected = pd.DataFrame().from_dict(expected, orient='index')
    df_expected.columns = ['Expected frequency']

    df_rand_obs = pd.DataFrame().from_dict(rand_obs, orient='index')
    df_rand_obs.columns = ['Target observation frequency' ]


    # We can safely reject the H_0, this p-value is too extreme 
    # to happen by random chance.
    chi, p_val = stats.chisquare(list(rand_obs.values()), list(expected.values()))

    df_random_obs_stats = pd.DataFrame().from_dict({
        'p-value: ': p_val, 
        'random cuisine type:': rand_obs_cuisine, 
        'chi-squared test statistic:' : chi
    }, orient='index')

    df_random_obs_stats.columns = ['H Test Stats']
    pd.concat([df_rand_obs, df_expected], axis=1).sort_index(ascending=False).plot(
        kind='barh', 
        figsize=(8,5)
    );
    return df_expected, df_rand_obs, df_random_obs_stats
    