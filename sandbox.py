"""
1) What are the most common medications for each disease in the base file?
2) What medications are most indicative of each disease?
3) Choose ONE of the diseases and build a model to infer whether that disease is present from the medications.
4) Demonstrate that the end user should be confident in the result.
5) Can you find any evidence that for the disease you've modeled, a certain drug is preferred by a certain demographic subgroup?
"""





# go with Python 2.7 for now
import numpy as np, pandas as pd
path = 'C:/Users/M543015/Desktop/GitHub/homework/'
# path = 'C:/Users/Glenn Wright/Contacts/Documents/GitHub/homework/'
base = pd.read_csv(path+'meps_base_data.csv', sep=',')
meds = pd.read_csv(path+'meps_meds.csv', sep=',')

base.dtypes
base.head()
base.describe()

meds.dtypes
meds.head()
meds.describe()


ndc = meds[['rxNDC','rxName','id']]
ndc = ndc.groupby(['rxNDC','rxName'], as_index=False).count()
ndc = ndc.sort_values(['rxNDC','id'], ascending=[True, False])
ndc = ndc.drop_duplicates()
ndc = dict(ndc[['rxNDC','rxName']].values.tolist())
diseases = base.columns[8:].values.tolist()
# get disease in long format

dis_long = base.melt('id',diseases)
dis_long = dis_long.drop_duplicates()
dis_long = dis_long.loc[dis_long['value'].isin(['Yes','No'])]
dis_long = dis_long.replace(['Yes','No'],[1,0])
medsdd = meds[['id', 'rxNDC']]
medsdd = medsdd.drop_duplicates()
dismeds = pd.merge(dis_long, medsdd,'inner','id')
dismeds.groupby(['variable','rxNDC'], as_index=False).count()
yes = dismeds.loc[dismeds['value']==1][['id','variable','rxNDC']]
sums = yes.groupby(['variable','rxNDC'], as_index=False).count()
sums = sums.sort_values(['variable','id'], ascending=[True, False])
# indicator variable for each medication by id
# maybe do this for each med separately, to avoid the super-wide data?

#1)
results1 = {}
ten = {}
fifty = {}
for disease in diseases:
	vals = sums.loc[sums['variable']==disease]
	ten[disease] = [val[1] for val in vals.values if val[2]>=10]
	fifty[disease] = [val[1] for val in vals.values if val[2]<=50]
	vals = vals[:10].values
	results1[disease] = [(val[1], ndc[val[1]]) for val in vals]


#2)
import scipy.stats as stats
results2 = {}
for disease in diseases:
	result = []
	dis = dis_long.loc[dis_long['variable']==disease]
	for med in ndc.keys():
		if (med in fifty[disease]):
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
					print disease
					print ndc[med]
					print oddsratio
					result.append((oddsratio, med, ndc[med]))

	result = sorted(result)[::-1][:5]
	results2[disease] = result

#3)
diabetes = dis_long.loc[dis_long['variable']=='diabetesDiagnosed'][['id','value']]
dmeds = meds[['id','rxNDC']].loc[meds['rxNDC'].isin(ten['diabetesDiagnosed'])]
dmeds = dmeds.drop_duplicates()
dmeds['ones'] = 1
dmeds = dmeds.pivot('id','rxNDC')
dmeds = dmeds.fillna(0)
dmeds['id'] = dmeds.index

medids = set([id for id in dmeds['id'].values.tolist()])
diabids = set([id for id in diabetes['id'].values.tolist()])
shared = set([id for id in diabids if id in medids])

data = dmeds.loc[dmeds['id'].isin(shared)]
data = data.sort_values('id').drop('id', axis=1)
data = data.values

target = diabetes.loc[diabetes['id'].isin(shared)]
target = target.sort_values('id')
target = target['value'].values

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.25, random_state=0)
lrmodel = LogisticRegression()
lrmodel.fit(x_train, y_train)
lrpredict = lrmodel.predict(x_test)

from sklearn.ensemble import RandomForestClassifier
rfmodel = RandomForestClassifier()
rfmodel.fit(x_train, y_train)
rfpredict = rfmodel.predict(x_test)


#4)
lrscores = lrmodel.score(x_test, y_test)
rfscores = rfmodel.score(x_test, y_test)

from sklearn.metrics import roc_curve, auc, confusion_matrix

lrconfuse = confusion_matrix(lrpredict, y_test)
rfconfuse = confusion_matrix(rfpredict, y_test)

tn, fn, fp, tp = lrconfuse.ravel()
lrsens = float(tp) / (tp+fn)
lrprec = float(tp) / (tp+fp)

tn, fn, fp, tp = rfconfuse.ravel()
rfsens = float(tp) / (tp+fn)
rfprec = float(tp) / (tp+fp)

lrprobs = lrmodel.predict_proba(x_test)[:,1:]
rfprobs = rfmodel.predict_proba(x_test)[:,1:]
import matplotlib.pyplot as plt
fpr, tpr, _ = roc_curve(y_test, lrprobs)
fpr, tpr, _ = roc_curve(y_test, rfprobs)
roc_auc = auc(fpr, tpr)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


ptable = []
for i in np.arange(0.0, 1.05, 0.05):
	predicts = np.where(lrprobs>=i, 1, 0)
	confuse = confusion_matrix(predicts, y_test)
	tn, fn, fp, tp = confuse.ravel()
	sens = float(tp) / (tp+fn)
	prec = float(tp) / (tp+fp)
	row = (i, sens, prec)
	ptable.append(row) 

#5)
demog = base.loc[base['diabetesDiagnosed']=='Yes']
demogs = {}
demog['marital'] = demog.married.apply(lambda x: x.replace(' IN ROUND',''))
demogs['race'] = demog.race.unique().tolist()
demogs['marital'] = demog.marital.unique().tolist()
demogs['sex'] = demog.sex.unique().tolist()
demog['agecat'] = pd.cut(demog['age'], [0,18,30,40,50,60,70,120])
demogs['agecat'] = demog.agecat.unique().tolist()

dprefer = {}
top = [row[0] for row in results1['diabetesDiagnosed']]
for demo in demogs:
	for cat in demogs[demo]:
		dm = demog[['id',demo]]
		dm[cat] = np.where(dm[demo]==cat, 1, 0)
		dm = dm.drop_duplicates()[['id',cat]]
		dprefer[cat] = []
		for med in top:
		# for med in fifty['diabetesDiagnosed']:
			md = meds.loc[meds['rxNDC']==med]
			md['value'] = 1
			mg = pd.merge(dm, md, 'left', 'id')
			mg = mg.fillna(0)
			mg = mg[['id', cat, 'value']]
			tb = mg.groupby([cat,'value'], as_index=False).count().values.tolist()
			if (len(tb)==4):
				table = [[tb[0][2], tb[1][2]],[tb[2][2], tb[3][2]]]
				oddsratio, pvalue = stats.fisher_exact(table)
				if pvalue < 0.01:
					dprefer[cat].append((oddsratio, med, ndc[med]))

		dprefer[cat] = sorted(dprefer[cat])

#most notable results are for children