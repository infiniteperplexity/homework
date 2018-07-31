# go with Python 2.7 for now
import numpy as np, pandas as pd
path = 'C:/Users/M543015/Desktop/GitHub/homework/'
base = pd.read_csv(path+'meps_base_data.csv', sep=',')
meds = pd.read_csv(path+'meps_meds.csv', sep=',')

base.dtypes
base.head()
base.describe()

base.dtypes
meds.head()
meds.describe()


# get disease in long format
diseases = base.columns[8:].values.tolist()
dis_long = base.melt('id',diseases)
dis_long = dis_long.drop_duplicates()
dis_long = dis_long.loc[dis_long['value'].isin(['Yes','No'])]
dis_long = dis_long.replace(['Yes','No'],[1,0])

# indicator variable for each medication by id
# maybe do this for each med separately, to avoid the super-wide data?
subs = meds[['id', 'rxNDC']]
subs = subs.drop_duplicates()
subs['ones'] = 1
subs = subs.pivot('id','rxNDC')
meds_wide = subs.fillna(0)


medsdd = meds[['id', 'rxNDC']]
medsdd = medsdd.drop_duplicates()
dismeds = pd.merge(dis_long, medsdd,'inner','id')
dismeds.groupby(['variable','rxNDC'], as_index=False).count()


yes = dismeds.loc[dismeds['value']==1][['id','variable','rxNDC']]
sums = yes.groupby(['variable','rxNDC'], as_index=False).count()
sums = sums.sort_values(['variable','id'], ascending=[True, False])
#dismeds.groupby(['variable','rxNDC'])['']

#https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.fisher_exact.html
#need to make a 2x2 table
import scipy.stats as stats
oddsratio, pvalue = stats.fisher_exact([[29467, 12933], [58, 804]])

#need to look up what Indexes are all about




import sklearn




# from sklearn import datasets
# from sklearn import metrics
# from sklearn.tree import DecisionTreeClassifier
# # load the iris datasets
# dataset = datasets.load_iris()
# # fit a CART model to the data
# model = DecisionTreeClassifier()
# model.fit(dataset.data, dataset.target)
# print(model)
# # make predictions
# expected = dataset.target
# predicted = model.predict(dataset.data)
# # summarize the fit of the model
# print(metrics.classification_report(expected, predicted))
# print(metrics.confusion_matrix(expected, predicted))


#sklearn.ensemble.RandomForestClassifier
#sklearn.linear_model.LogisticRegression
#sklearn.model_selection

#http://scikit-learn.org/stable/modules/cross_validation.html