"""
1) What are the most common medications for each disease in the base file?
2) What medications are most indicative of each disease?
3) Choose ONE of the diseases and build a model to infer whether that disease is present from the medications.
4) Demonstrate that the end user should be confident in teh result.
5) Can you find any evidence that for the disease you've modeled, a certain drug is preferred by a certain demographic subgroup?
"""





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


ndc = meds[['rxNDC','rxName','id']]
ndc = ndc.groupby(['rxNDC','rxName'], as_index=False).count()
ndc = ndc.sort_values(['rxNDC','id'], ascending=[True, False])
ndc = ndc.drop_duplicates()
ndc = dict(ndc[['rxNDC','rxName']].values.tolist())

ndc = ndc.drop_duplicates()
ndc1 = ndc.drop_duplicates('rxNDC')
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

#1)
results1 = {}
for disease in diseases:
	vals = sums.loc[sums['variable']==disease][:5].values
	results1[disease] = [(val[1], ndc[val[1]]) for val in vals]

#2)


import scipy.stats as stats
dis = dis_long.loc[dis_long['variable']=='anginaDiagnosed']
medyn = meds[['id','rxNDC']].loc[meds['rxNDC']==63653117101]
medyn = medyn.drop_duplicates()
medyn['value2'] = 1
dm = pd.merge(dis_long,medyn,'left','id')
dm = dm.fillna(0)[['value','value2','id']]
dm = dm.groupby(['value','value2'], as_index=False).count().values.tolist()
table = [[dm[0][2], dm[1][2]],[dm[2][2], dm[3][2]]]
oddsratio, pvalue = stats.fisher_exact(table)

results2 = {}
for disease in diseases:
	result = []
	dis = dis_long.loc[dis_long['variable']==disease]

	for med in ndc.keys():
		medyn = meds[['id','rxNDC']].loc[meds['rxNDC']==med]
		medyn = medyn.drop_duplicates()
		medyn['value2'] = 1
		
		dm = pd.merge(dis_long,medyn,'left','id')
		dm = dm.fillna(0)[['value','value2','id']]
		dm = dm.groupby(['value','value2'], as_index=False).count().values.tolist()
		if (len(dm)==4):
			table = [[dm[0][2], dm[1][2]],[dm[2][2], dm[3][2]]]
			oddsratio, pvalue = stats.fisher_exact(table)
			if pvalue < 0.01:
				result.push((oddsratio, med, ndc[med]))

	result = sorted(result)[::-1][:5]
	results[disease] = result







import sklearn

#https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.fisher_exact.html
#need to make a 2x2 table
import scipy.stats as stats
oddsratio, pvalue = stats.fisher_exact([[29467, 12933], [58, 804]])


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