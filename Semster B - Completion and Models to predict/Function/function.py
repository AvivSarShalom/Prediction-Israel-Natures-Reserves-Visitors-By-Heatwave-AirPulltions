import calendar
import pandas as pd

def remove_high_corr(df,target,threshold=0.5):
  '''
  return dataframe without corrlation that can be drop.
  
  args:
  df = dataframe
  target = string of the target
  threshold = default 0.5
  '''
  import pandas as pd
  import numpy as np

  target_col = df.pop(target)
  df.insert(len(df.columns), target, target_col)
  cor_matrix = df.corr().abs()
  corr_df = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(np.bool))
  #מתודה שאומרת בי בקורלורציה עם מי
  cols = corr_df.columns.to_list()
  list_corr_not_empty=[]

  for i in range(len(cols)-1):
      tmp = []
      for j in range(len(cols)-1):
        if abs(corr_df.iloc[i,j]) >= threshold and cols[i] is not cols[j] :
          tmp.append(cols[j])
      if len(tmp)>0:
          tmp.append(cols[i])
          list_corr_not_empty.append(tmp)
  def Key(p):
   return  corr_df[target][p]
  stay = [max(sub,key=Key) for sub in list_corr_not_empty]
  drops = [ c for sub in list_corr_not_empty for c in sub if c not in stay ]
  return df.drop(list(set(drops)),axis=1)
  pass

def remove_outliers(df,target_name):
  '''
  return df without outliers.
  
  args:
  df = dataframe
  target = string of the target
  '''
  import matplotlib.pyplot as plt
  plt.cla()
  bp = plt.boxplot(df[target_name])
  minimums = [round(item.get_ydata()[0], 4) for item in bp['caps']][::2]
  maximums = [round(item.get_ydata()[0], 4) for item in bp['caps']][1::2]
  return df.drop(df [ (df[target_name]>maximums[0])  | (df[target_name]<minimums[0])].index)



def plot_line(prediction,actual,title='',path_save=None,file_name=None,fig_size_tuple=(18,8),xlim=None,ylim=None
,alpha_prediction=1,alpha_actual=1):
  '''plot the line graph of the resualts 
  title can be added with title
  if want to save add path_name and file_name

  arugments: prediction, actual, title='' ,path_save=None ,file_name=None , fig_size_tuple=(18,8) xlim=None,ylim=None

  example for saving:
  path_name= 'folder1/save_here_folder/'
  path_file = file_name.png
  '''
  import os
  from pylab import rcParams
  rcParams['figure.figsize'] = fig_size_tuple[0],fig_size_tuple[1]
  import matplotlib.pyplot as plt
  import pandas as pd
  res = pd.DataFrame(data={
    'Predictions':prediction,
    'Actual':actual
  })
  plt.plot(res.index, res['Predictions'], color='r', label='Predicted Visitors',alpha=alpha_prediction)
  plt.plot(res.index, res['Actual'], color='b', label='Actual Visitors',alpha=alpha_actual)
  plt.grid(which='major', color='#cccccc', alpha=0.5)
  plt.legend(shadow=True)
  plt.title(title, family='Arial', fontsize=26)
  plt.ylabel('Visitors', family='Arial', fontsize=22)
  plt.xticks(rotation=45, fontsize=16)
  plt.yticks(rotation=45, fontsize=16)
  plt.xlim(xlim)
  plt.ylim(ylim)
  
  if path_save is not None:
    isExist = os.path.exists(path_save)
    if not isExist:
      os.makedirs(path_save)
    plt.savefig(path_save+file_name)
  plt.show()



def plot_residuals(prediction,actual,title='',path_save=None,file_name=None,fig_size_tuple=(18,8),xlim=None,ylim=None):
  '''plot the residuales of the resualts 
  if want to save add path_name and file_name
  
  arugments: prediction, actual, title='' ,path_save=None ,file_name=None, fig_size_tuple=(18,8) xlim=None,ylim=None
  example:
  path_name= 'folder1/save_here_folder/'
  path_file = file_name.png
  '''
  import os
  from pylab import rcParams
  rcParams['figure.figsize'] = fig_size_tuple[0],fig_size_tuple[1]
  import matplotlib.pyplot as plt
  import pandas as pd
  res = pd.DataFrame(data={
    'Predictions':prediction,
    'Actual':actual
  })
  res['residuals'] = res['Predictions'] - res['Actual']
  plt.plot(res.Predictions,res.residuals,color='r',marker='.',linestyle='None')
  plt.xlabel('Visitors', family='Arial', fontsize=22)
  plt.ylabel('Residuals', family='Arial', fontsize=22)
  plt.plot(res.Predictions,res.residuals*0,color='b')
  plt.title(title, family='Arial', fontsize=26)
  plt.grid(which='major', color='#cccccc', alpha=0.5)
  plt.legend(shadow=True)
  plt.yticks(rotation=45, fontsize=16) 
  plt.xlim(xlim)
  plt.ylim(ylim)
  
  
  plt.xticks(rotation=45, fontsize=16)
  if path_save is not None:
    isExist = os.path.exists(path_save)
    if not isExist:
      os.makedirs(path_save)
    plt.savefig(path_save+file_name)
  plt.show()



def split_date(dataframe):
  '''
  split the date in the df to columns years
  month and days 
  
  return df with column year,month,day
  '''

  import pandas as pd
  dataframe = dataframe.set_index("Date")
  dataframe['day'] = dataframe.index.day
  dataframe['month'] = dataframe.index.month
  dataframe['year'] = dataframe.index.year
  dataframe.reset_index(drop=False,inplace=True)

  return dataframe

def get_rmse(x,y):
  from sklearn.metrics import mean_squared_error
  from math import sqrt
  return sqrt(mean_squared_error(x,y))

def remove_unique_one(df):
  '''
  remove columns with 1 feature only
  
  return df without columns with 1 feature only 
  '''
  drop_one_unique = [x for x in df.columns if len(df[x].value_counts())==1]
  return df.drop(drop_one_unique,axis=1)


def last_year_entries_info(dataframe,target):
  '''
  this function must run after split date function

  return dataframe with column last year visitors 
  '''
  import datetime
  def make_groups(dataframe,target):
      #getting the mean value for every day and month. it returns a dataframe of the results.
      return dataframe.groupby(['day','month'])[target].mean().reset_index()

  def fill_last_year_nulls(groups,day,month,target):
      #fill the nulls from the groups we made
      return int(groups.loc[(groups['day'] == day) & (groups['month'] == month)][target])


  def last_year_visitors(dataframe,day,month,year,target):
      #Extract the visitors 
      visitors_last_year = dataframe.loc[(dataframe['day'] == day) &(dataframe['month'] == month) & (dataframe['year'] == year-1)][target]    
      #Return the value
      if visitors_last_year.empty:
          return None
      else : return int(visitors_last_year)

    # dataset['Last_year_visitors'] = dataset.apply(lambda row : last_year_visitors(dataset,row['day'],row['month'],row['year'],'Israelis_Count') , axis = 1)

  #get last year info
  if 'Day_before_Total' in dataframe.columns:
    dataframe.drop('Day_before_Total',axis=1,inplace=True)
  dataframe['Last_year_visitors'] = dataframe.apply(lambda row : last_year_visitors(dataframe,row['day'],row['month'],row['year'],target) , axis = 1)
  
  #make use of group day to fill null values based on avg 
  groups = make_groups(dataframe , target)
  
  #getting the indexes where its null
  mask = dataframe['Last_year_visitors'].isna()
  #fill the nulls
  dataframe.loc[mask, 'Last_year_visitors'] = dataframe.loc[dataframe['Last_year_visitors'].isna()].apply(lambda row :fill_last_year_nulls(groups,row['day'],row['month'],target),axis=1)

  #make target last columns 
  t=dataframe[target]
  dataframe.drop(target,axis=1,inplace=True)
  dataframe[target] = t

  print('Add Last year visitors Successfully')
  return dataframe
  
def remove_pollution_site(dataset):
  '''
  remove the feature
   'nox','pm10','pm2.5','so2','is_Site_exceeded_pm10','is_Site_exceeded_pm2.5', 'is_Site_exceeded_nox','is_Site_exceeded_so2'
  '''
  print('remove pollution site Successfully')
  return dataset.drop(['nox','pm10','pm2.5','so2','is_Site_exceeded_pm10','is_Site_exceeded_pm2.5', 'is_Site_exceeded_nox','is_Site_exceeded_so2'],axis=1)



def move_target_to_last(dataset,target):
  t = dataset[target]
  dataset.drop(target,axis=1,inplace=True)
  dataset[target] = t
  return dataset

def remove(df , to_remove):
  cols = df.columns
  if to_remove in cols:
    df.drop(to_remove , inplace=True , axis = 1)

  return df


def get_weekday(dataset):

  dataset['week_Day'] = dataset.Date.apply(lambda date : calendar.day_name[date.weekday()])
  days = pd.get_dummies(dataset['week_Day'])
  dataset = pd.concat([dataset , days] , axis=1)
  dataset = remove(dataset , 'week_Day')
  return dataset

  def add_last_visitors_for_all_sites_in_df(df,target):
    '''
    use the method last_year_entries_info for each site in the dataframe

    return df
    '''
    dataset = df.copy()
    dataset['Last_year_visitors_IL'] = 0
    sites = dataset.Site_Name.unique()
    dataset = function.split_date(dataset)
    dataset = function.move_target_to_last(dataset, target)
    dataset = dataset.sort_values(['year','month','day'])
    for site in sites:
      print(site)
      site_dataset = dataset.loc[dataset.Site_Name==site]
      site_dataset = function.last_year_entries_info(site_dataset,target)
      # print(site_dataset.Last_year_visitors_IL  )
      dataset.loc[dataset.Site_Name==site,'Last_year_visitors_IL'] = site_dataset.Last_year_visitors
      pass

    print('**********************************************')
    print('Add All Sites Last year visitors Successfully')
    print('**********************************************')
    return dataset