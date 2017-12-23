
import sklearn 
from sklearn import metrics
import json
from rdkit import Chem
from rdkit import DataStructs
import collections
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier as RFC



batch_size = 5
input_size = 4
hidden_size = 4
num_classes = 1
learning_rate = 0.005
test_split = 1100
cutoff = 50
x_val_arr = []
c = 0
while c < 300:
	x_val_arr.append(c)
	c += 1

effects = []
smiles_og = []
max_len = 0
#use count to decrease training set data if you want
count = 0
cutoff_length = 150

#pass in two numpy arrays
def get_auc(predictions, real_values):
	fpr, tpr, thresholds = metrics.roc_curve(real_values, predictions)
	print metrics.auc(fpr, tpr)

def mol2imageT(x, N = 4096):
    try:
        m = Chem.MolFromSmiles(x)
        fp = Chem.RDKFingerprint(m, maxPath=4, fpSize=N)
        res = np.zeros(len(fp))
        DataStructs.ConvertToNumpyArray(fp,res)
        return res
    except:
        pass
        return np.nan


# GET THE SMILES STRINGS AND THE FIRST EFFECT

print "side effect, data split, accuracy, f1_score, AUC"

with open("./json/side_effects_all_binary_top_100.json") as f:
	d = json.load(f)

current_effect = 0
while current_effect < len(d):
	effects = []
	effect_to_print = "n"
	for cid in d:
		tmp = []
		#just get the first side effect
		ordered_effects = collections.OrderedDict(sorted(d[cid]['side effects'].items()))

		c = 0
		for effect in ordered_effects:

			if c == 1:
				effect_to_print = effect
				#print d[cid]['side effects'][effect]
				#tmp.append(d[cid]['side effects'][effect])
				tmp = d[cid]['side effects'][effect]
				break
			c += 1

		#remove long smiles strings 
		if len(d[cid]['SMILES']) > cutoff_length:
			continue

		effects.append(tmp)
		#get the max length 
		smiles_og.append(d[cid]['SMILES'])
		if len(d[cid]['SMILES']) > max_len:
			max_len = len(d[cid]['SMILES'])
		count += 1

	#print the name of the side effect
	print effect_to_print,

	print "amount of 1 " + str(effects.count(1))
	print "amount of 0 " + str(effects.count(0))
	print effects 

	#print the amount of 1s vs 0s in the side effect
	print ","+str(float(effects.count(1))/float(len(effects))),

	test_split = int(0.8*len(smiles_og))
	#plt.hist(lengths_arr, bins=len(lengths_arr)/5)
	#plt.savefig('charts/length_200_hist_bins_full_len_div_5.png')

	#train_sm is the traiing set of the smiles strings
	#train target is the training set for if the SMILES string has the effect we are focusing on now
	train_sm = smiles_og[:test_split]
	train_target = np.asarray(effects[:test_split])
	#these two are empty right now
	test_sm = smiles_og[test_split:]
	test_target = effects[test_split:]
	i = 0
	
	test_target = np.asarray(test_target)

	fp_train = []
	for mol in train_sm:
	    fp_train.append(mol2imageT(mol, N=2048))
	fp_train = np.asarray(fp_train)

	fp_test = []
	for mol in test_sm:
	    fp_test.append(mol2imageT(mol, N=2048))
	fp_test = np.asarray(fp_test)

	classifier = RFC(n_estimators=100, oob_score=True)
	'''
	print fp_train 
	print "------printing special fp-train-------"
	print fp_train[:,None]
	'''

	classifier.fit(fp_train, train_target.ravel())

	#using oob_for for data set
	pred_ans = classifier.predict(fp_test)
	#pred_ans = classifier.oob_decision_function_

	#get rounded answers
	binary_pred_ans = []
	for p in pred_ans:
		#binary_pred_ans.append(int(round(p[1])))
		#non oob version below
		binary_pred_ans.append(int(round(p)))	

	#print "accuracy " + str(sklearn.metrics.accuracy_score(train_target, binary_pred_ans))
	#print "," + str(sklearn.metrics.accuracy_score(train_target, binary_pred_ans)),
	print "," + str(sklearn.metrics.accuracy_score(test_target, binary_pred_ans)),

	#print binary_pred_ans
	#print "," + f1_score(train_target, np.asarray(binary_pred_ans)),
	print ",",
	
	#I think that pred_ans[:,1] gives you the positive answers
	#get_auc(pred_ans[:,1], train_target)
	#non oop version below
	get_auc(pred_ans, test_target)

	#print(classification_report(train_target, binary_pred_ans, target_names=['class 0', 'class 1']))


	# for doc get name of side effect, data split, accuracy, and AUC for each row in csv

	'''
	print "predicted ansers:"
	print pred_ans
	print "target answers:"
	print test_target
	print "training set: " + str(test_split) + ", testing set: " + str(len(test_target))
	i = 0
	count = 0
	while i < len(test_target):
		if int(test_target[i]) == int(pred_ans[i]): 
			count += 1
		i += 1
	print 'accuracy: ' + str(float(count)/float(len(test_target))) 

	get_auc(pred_ans, np.asarray(test_target))

	'''

	current_effect+=1