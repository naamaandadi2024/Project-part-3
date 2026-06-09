import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt




def prepare_data(df_input):
    df = df_input.copy()
    import ast
    def count_actors(x):
         if pd.isna(x) or str(x).strip() == '' or str(x) == 'NaN':
            return 0
         try:
            return len(ast.literal_eval(x))
         except:
            return 0
   

   
    df['num_lead_actors'] = df['lead_actors_ids'].map(count_actors)
    
    df['duration_category'] = np.select(
        [
            (df['runtimeMinutes'] > 0) & (df['runtimeMinutes'] < 90),
            (df['runtimeMinutes'] >= 90) & (df['runtimeMinutes'] <= 130),
            (df['runtimeMinutes'] > 130)
        ],
        [0, 1, 2],
        default=np.nan
    )
    df['plot_word_count'] = df['plot'].apply(lambda x: len(str(x).split()) if pd.notna(x) else np.nan)

     
    clean_language = df['Language'].str.strip().str.title()
    df['is_english'] = np.where(clean_language == 'English', 1, 
                   np.where(clean_language.isna(), np.nan, 0))
    
    df['is_drama'] = np.select(
        [df['genres'].str.contains('Drama', na=False,case=False), df['genres'].notna() & ~df['genres'].str.contains('Drama', na=False)],
        [1, 0], default=np.nan
    )
    
    
    df['is_us'] = np.select(
    [
        (df['Country'] == 'United States') | 
        (df['Country'] == 'United States of America') | 
        (df['Country'] == 'USA') | 
        (df['Country'] == 'US')], [1], default=0)

    df['decade'] = (df['startYear'] // 10) * 10
  
   
    df = df.drop(columns=['numVotes', 'BoxOffice', 'tconst', 'primaryTitle', 'plot', 'Country', 'Language', 'genres','budget','averageRating'])

    return df  