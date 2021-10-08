import numpy as np
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score,precision_score,roc_curve,auc, accuracy_score,confusion_matrix,cohen_kappa_score
import matplotlib
import matplotlib.pyplot as plt
import random


def shuffle(X,Y,Z=[]):
    perm=np.random.permutation(X.shape[0])
    X=X[perm]
    Y=Y[perm]
    if(Z==[]):
        return X,Y
    else:
        Z=Z[perm]
        return X,Y,Z


def num_to_cat(y,sz):
    temp=np.zeros([len(y),sz])
    for i in range(len(y)):
        for j in range(sz):
            if(j==y[i]):
                temp[i,j]=1

    return temp


def plot_auc(test_L,y_scores):
	fpr, tpr, _ = roc_curve(test_L, y_scores)
	roc_auc = auc(fpr, tpr)
	plt.figure()
	lw = 2
	plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic example')
	plt.legend(loc="lower right")
	plt.show()


def kappa(t_L,preds):
	a=sum(t_L)
	ca=sum(preds)

	na=len(t_L)-a
	cna=len(preds)-ca

	aca=a*ca
	nacna=na*cna


	acaDnl = aca/ len(t_L)
	nacnaDnl = nacna/len(t_L)

	pexp=(acaDnl+nacnaDnl)/float(len(t_L))


	kappa=(accuracy_score(preds,t_L) -pexp)/(1-pexp)

	return kappa


def print_scores(preds,labels,prnt='on'):
        if(np.amax(labels)<=1):
            con=confusion_matrix(labels,preds)
            spec=con[0,0]/(con[0,1]+con[0,0])
            sens=con[1,1]/(con[1,0]+con[1,1])
            acc=accuracy_score(preds,labels)
            prec=precision_score( labels,preds)
            kappa1=kappa(labels,preds)
            if(prnt=='on'):print("Binary: Spec:",spec,"Sens:",sens,"Prec:", prec,"Acc:",acc,"Kappa:",kappa1)
            return spec,sens, prec,acc,kappa1

        else:
            k=cohen_kappa_score(labels,preds)
            acc=accuracy_score(labels,preds)
    
            #kappa1=kappa(labels,preds)
            if(prnt=='on'):print("Multi: Acc:",acc,"k:",k)
            return  acc,0,0,0,k
        





def test_accuracy(train_D,train_L,val_D,val_L,test_D,test_L,prnt='on'):

	print("Validation:")
	####MLP#####################################

	nb= MLPClassifier(solver='adam', alpha=1e-5,activation='relu',max_iter=1000,random_state=1234)
	nb.fit(train_D,train_L)

	pred_NN=nb.predict(val_D)
	acc_NN=accuracy_score(pred_NN,val_L)
	if(prnt=='on'):print("Acc Neural:",acc_NN)

	con=confusion_matrix(val_L,pred_NN)



	y_scoreNN=nb.predict_proba(val_D)[:,1]


	aurocNN=roc_auc_score( val_L,y_scoreNN)
	kappaNN_val=kappa(val_L,pred_NN)

	spec_NN=con[0,0]/(con[0,1]+con[0,0])
	sens_NN=con[1,1]/(con[1,0]+con[1,1])

	if(prnt=='on'):print("spec,sens, precision NN ROC, kappa ANN:",spec_NN,sens_NN,precision_score( val_L,pred_NN),aurocNN,kappaNN_val)



	#########SVM################################


	clf=SVC(probability=True)
	clf.fit(train_D,train_L,)


	y_scoreSVM=clf.predict_proba(val_D)[:,1]





	pred_SVM=clf.predict(val_D)
	acc_SVM=accuracy_score(pred_SVM,val_L)



	if(prnt=='on'):print("acc SVM is:",acc_SVM)


	con=confusion_matrix(val_L,pred_SVM)


	aurocSVM=roc_auc_score(val_L,y_scoreSVM)
	kappaSVM_val=kappa(val_L,pred_SVM)


	spec_SVM=con[0,0]/(con[0,1]+con[0,0])
	sens_SVM=con[1,1]/(con[1,0]+con[1,1])
	if(prnt=='on'): print("spec,sens,prec ROC SVM,kappa SVM:",spec_SVM,sens_SVM,precision_score( val_L,pred_SVM),aurocSVM,kappaSVM_val)




	##########KNN##################################

	neigh = KNeighborsClassifier(n_neighbors=12,weights='distance')
	neigh.fit(train_D,train_L)
	pred_KNN=neigh.predict(val_D)
	acc_KNN=accuracy_score(pred_KNN,val_L)

	if(prnt=='on'):print(" acc KNN is:",acc_KNN)


	con=confusion_matrix(val_L,pred_KNN)



	aurocKNN=roc_auc_score( val_L,neigh.predict_proba(val_D)[:,1])
	kappaKNN_val=kappa(val_L,pred_KNN)



	spec_KNN=con[0,0]/(con[0,1]+con[0,0])
	sens_KNN=con[1,1]/(con[1,0]+con[1,1])
	if(prnt=='on'):print("spec,sens,prec,ROC KNN, kappaKNN:",spec_KNN,sens_KNN,precision_score( val_L,pred_KNN),aurocKNN,kappaKNN_val)




	####random_forest#####################################

	rndm= RandomForestClassifier(50,random_state=1234)
	rndm.fit(train_D,train_L)



	pred_RndmF=rndm.predict(val_D)
	acc_Rndm=accuracy_score(pred_RndmF,val_L)




	if(prnt=='on'):print("acc RandomForest is:",acc_Rndm)


	con=confusion_matrix(val_L,pred_RndmF)


	aurocRndm=roc_auc_score(val_L,rndm.predict_proba(val_D)[:,1])
	kappaRndm_val=kappa(val_L,pred_RndmF)


	spec_Rndm=con[0,0]/(con[0,1]+con[0,0])
	sens_Rndm=con[1,1]/(con[1,0]+con[1,1])
	if(prnt=='on'):print("spec,sens,prec,ROC Rndm,kappaRndm:",spec_Rndm,sens_Rndm,precision_score( val_L,pred_RndmF),aurocRndm,kappaRndm_val)



	'''

	total_acc_val=(1/4)*(acc_NN+acc_SVM+acc_KNN+acc_Rndm)
	total_spec_val=(1/4)*(spec_NN+spec_SVM+spec_KNN+spec_Rndm)
	total_sens_val=(1/4)*(sens_NN+sens_SVM+sens_KNN+sens_Rndm)

	total_auroc_val=(aurocNN+aurocRndm+aurocSVM+aurocKNN)/4
	total_kappa_val=(kappaNN+kappaRndm+kappaSVM+kappaKNN)/4
	print("total  acc mean, total Ensemble ROC,total spec, total sens,kappa",total_acc_val,total_auroc_val,total_spec_val,total_sens_val,total_kappa_val)


	'''




	##########################################################################################################################
	###############################################################################################################################

	print("TEST:")


	#NN

	pred_NN=nb.predict(test_D)
	acc_NN=accuracy_score(pred_NN,test_L)
	if(prnt=='on'):print("Acc Neural:",acc_NN)

	con=confusion_matrix(test_L,pred_NN)


	y_scoreNN=nb.predict_proba(test_D)[:,1]


	aurocNN=roc_auc_score( test_L,y_scoreNN)
	kappaNN=kappa(test_L,pred_NN)

	spec_NN=con[0,0]/(con[0,1]+con[0,0])
	sens_NN=con[1,1]/(con[1,0]+con[1,1])

	if(prnt=='on'):print("spec,sens, precision NN ROC, kappa ANN:",spec_NN,sens_NN,precision_score( test_L,pred_NN),aurocNN,kappaNN)


	y_scoreSVM=clf.predict_proba(test_D)[:,1]


	#####################SVM




	pred_SVM=clf.predict(test_D)
	acc_SVM=accuracy_score(pred_SVM,test_L)





	if(prnt=='on'):print("acc SVM is:",acc_SVM)


	con=confusion_matrix(test_L,pred_SVM)
	#print("confusion Matrix:\n",con)


	aurocSVM=roc_auc_score(test_L,y_scoreSVM)
	kappaSVM=kappa(test_L,pred_SVM)


	spec_SVM=con[0,0]/(con[0,1]+con[0,0])
	sens_SVM=con[1,1]/(con[1,0]+con[1,1])
	if(prnt=='on'): print("spec,sens,prec ROC SVM,kappa SVM:",spec_SVM,sens_SVM,precision_score( test_L,pred_SVM),aurocSVM,kappaSVM)

	##########KNN

	pred_KNN=neigh.predict(test_D)
	acc_KNN=accuracy_score(pred_KNN,test_L)


	if(prnt=='on'):print(" acc KNN is:",acc_KNN)


	con=confusion_matrix(test_L,pred_KNN)
	#print("confusion Matrix:\n",con,con[0,1])


	aurocKNN=roc_auc_score( test_L,neigh.predict_proba(test_D)[:,1])
	kappaKNN=kappa(test_L,pred_KNN)



	spec_KNN=con[0,0]/(con[0,1]+con[0,0])
	sens_KNN=con[1,1]/(con[1,0]+con[1,1])
	if(prnt=='on'):print("spec,sens,prec,ROC KNN, kappaKNN:",spec_KNN,sens_KNN,precision_score( test_L,pred_KNN),aurocKNN,kappaKNN)


	#RF
	pred_RndmF=rndm.predict(test_D)
	acc_Rndm=accuracy_score(pred_RndmF,test_L)


	if(prnt=='on'):print("acc RandomForest is:",acc_Rndm)


	con=confusion_matrix(test_L,pred_RndmF)

	aurocRndm=roc_auc_score(test_L,rndm.predict_proba(test_D)[:,1])
	kappaRndm=kappa(test_L,pred_RndmF)


	spec_Rndm=con[0,0]/(con[0,1]+con[0,0])
	sens_Rndm=con[1,1]/(con[1,0]+con[1,1])
	if(prnt=='on'):print("spec,sens,prec,ROC Rndm,kappaRndm:",spec_Rndm,sens_Rndm,precision_score( test_L,pred_RndmF),aurocRndm,kappaRndm)

	'''
	total_acc=(1/4)*(acc_NN+acc_SVM+acc_KNN+acc_Rndm)
	total_spec=(1/4)*(spec_NN+spec_SVM+spec_KNN+spec_Rndm)
	total_sens=(1/4)*(sens_NN+sens_SVM+sens_KNN+sens_Rndm)

	total_auroc=(aurocNN+aurocRndm+aurocSVM+aurocKNN)/4
	total_kappa=(kappaNN+kappaRndm+kappaSVM+kappaKNN)/4
	print("total  acc mean, total Ensemble ROC,total spec, total sens,kappa",total_acc,total_auroc,total_spec,total_sens,total_kappa)
	'''

	return kappaRndm_val,kappaKNN_val,kappaNN_val,kappaSVM_val



def test_accuracy2(train_D,train_L,val_D,val_L,test_D,test_L,s="MLP" ,prnt='on'):
        print("Validation"+s+":")
        if(s=="MLP"):

                nb= MLPClassifier(solver='adam', alpha=1e-2,activation='relu',max_iter=1000,random_state=1234)
                nb.fit(train_D,train_L)

                pred_NN=nb.predict(val_D)

        elif(s=="RF"):
                rndm= RandomForestClassifier(50,random_state=1234)
                rndm.fit(train_D,train_L)

                pred_NN=rndm.predict(val_D)

        elif(s=="KNN"):
                neigh = KNeighborsClassifier(n_neighbors=5,weights='distance',metric='euclidean')
                neigh.fit(train_D,train_L)
                pred_NN=neigh.predict(val_D)



        acc_NN=accuracy_score(pred_NN,val_L)
        if(prnt=='on'):print("acc "+s+" is:",acc_NN)

        con=confusion_matrix(val_L,pred_NN)

        kappaNN=kappa(val_L,pred_NN)

        spec_NN=con[0,0]/(con[0,1]+con[0,0])
        sens_NN=con[1,1]/(con[1,0]+con[1,1])

        if(prnt=='on'):print("spec,sens, precision ROC, kappa:",spec_NN,sens_NN,precision_score( val_L,pred_NN),kappaNN)





        total_acc_val=(acc_NN)
        total_spec_val=(spec_NN)
        total_sens_val=(sens_NN)

        total_kappa_val=kappaNN
        print("total  acc mean, total Ensemble ROC,total spec, total sens,kappa",total_acc_val,total_spec_val,total_sens_val,total_kappa_val)





	##########################################################################################################################
	###############################################################################################################################

        print("TEST"+s+":")




        #NN
        if(s=="MLP"):
        	pred_NN=nb.predict(test_D)
        elif(s=="RF"):
                pred_NN=rndm.predict(test_D)
        elif(s=="KNN"):
                pred_NN=neigh.predict(test_D)


        acc_NN=accuracy_score(pred_NN,test_L)
        if(prnt=='on'):print("Acc Neural:",acc_NN)

        con=confusion_matrix(test_L,pred_NN)

        kappaNN=kappa(test_L,pred_NN)

        spec_NN=con[0,0]/(con[0,1]+con[0,0])
        sens_NN=con[1,1]/(con[1,0]+con[1,1])

        if(prnt=='on'):print("spec,sens, precision  kappa :",spec_NN,sens_NN,precision_score( test_L,pred_NN),kappaNN)



        total_acc=(acc_NN)
        total_spec=(spec_NN)
        total_sens=(sens_NN)


        total_kappa=(kappaNN)
        print("total  acc mean,total spec, total sens,kappa",total_acc,total_spec,total_sens,total_kappa)


        return total_acc,total_spec,total_sens,total_kappa,total_acc_val,total_spec_val,total_sens_val,total_kappa_val




def test_accuracy3(train_D,train_L,val_D,val_L,test_D,test_L,s="MLP",prnt='off'):


	total_acc_val=0
	total_spec_val=0
	total_sens_val=0

	total_kappa_val=0

	total_acc=0
	total_spec=0
	total_sens=0

	total_kappa=0

	N=10
	print("Classifier: "+s)
	for i in range(N):
		if(s=="MLP"):
			nb= MLPClassifier(solver='adam', alpha=1e-5,activation='relu',max_iter=1000,random_state=i*500)
			nb.fit(train_D,train_L)

			pred_NN=nb.predict(val_D)

		elif(s=="RF"):

			rndm= RandomForestClassifier(50,random_state=i*500)
			rndm.fit(train_D,train_L)
			pred_NN=rndm.predict(val_D)


		acc_NN=accuracy_score(pred_NN,val_L)


		con=confusion_matrix(val_L,pred_NN)

		kappaNN=kappa(val_L,pred_NN)

		spec_NN=con[0,0]/(con[0,1]+con[0,0])
		sens_NN=con[1,1]/(con[1,0]+con[1,1])

		if(prnt=='on'):print("Val acc,spec,sens, precision, kappa:",acc_NN,spec_NN,sens_NN,precision_score( val_L,pred_NN),kappaNN)


		total_acc_val=total_acc_val+acc_NN/N
		total_spec_val=total_spec_val+spec_NN/N
		total_sens_val=total_sens_val+sens_NN/N

		total_kappa_val=total_kappa_val+kappaNN/N

		########################################################################

		if(s=="MLP"):
			pred_NN=nb.predict(test_D)
		elif(s=="RF"):
			pred_NN=rndm.predict(test_D)

		acc_NN=accuracy_score(pred_NN,test_L)

		con=confusion_matrix(test_L,pred_NN)

		kappaNN=kappa(test_L,pred_NN)

		spec_NN=con[0,0]/(con[0,1]+con[0,0])
		sens_NN=con[1,1]/(con[1,0]+con[1,1])

		if(prnt=='on'):print("Test acc, spec,sens, precision NN, kappa ANN:",acc_NN,spec_NN,sens_NN,precision_score( test_L,pred_NN),kappaNN)


		total_acc=total_acc+acc_NN/N
		total_spec=total_spec+spec_NN/N
		total_sens=total_sens+sens_NN/N

		total_kappa=total_kappa+kappaNN/N

	print("\n\n")
	print("total Val  acc mean, total Ensemble ROC,total spec, total sens,kappa",total_acc_val,total_spec_val,total_sens_val,total_kappa_val)
	print("total Test  acc mean, total Ensemble ROC,total spec, total sens,kappa",total_acc,total_spec,total_sens,total_kappa)



	return total_acc,total_spec,total_sens,total_kappa,total_acc_val,total_spec_val,total_sens_val,total_kappa_val




def transf(x):
    try:
        return float(x)
    except ValueError:
        return 0  # np.nan


def clearNaN(x):
	 # <TODO>: change  the NaN values with the average of the previous and the next of the array instead of putting zeros on transf
    return 0


def standardize(X):
    x_mean = np.mean(X)
    x_std = np.std(X)
    return (X - x_mean) / x_std


def scale(x):
    xmin = x.min()
    xmax = x.max()

    x_tmp = (x ) / (xmax - xmin)
    return x_tmp


def rescale(X):
    X1, X2, X3, X4 = X[:, 0:60], X[:, 60:120], X[:, 120:180], X[:, 180:240]
    X1_n, X2_n, X3_n, X4_n = scale(X1), scale(X2), scale(X3), scale(X4)

    return np.concatenate((X1_n, X2_n, X3_n, X4_n), axis=1)



def HarmonicMean(tr1_D,tr1_L,tr2_D,tr2_L):
	acc1,sens1,spec1,_=test_accuracy(tr1_D,tr1_L,tr2_D,tr2_L,prnt='off')
	acc2,sens2,spec2,_=test_accuracy(tr2_D,tr2_L,tr1_D,tr1_L,prnt='off')
	hrm=(2*acc1*acc2)/(acc1+acc2)
	print()
	print("Harmonic mean of TSTR:",hrm)
	print()
	return hrm


#for 2 classes
def BalanceClasses(X,Y,size1):
    X1,Y1=RandomSubsampleCond(X,Y,size1,1)
    X0,Y0=RandomSubsampleCond(X,Y,size1,0)
    X_new=np.concatenate((X0,X1),0)
    Y_new=np.concatenate((Y0,Y1))

    _,_,X_new,Y_new=RandomSubsample(X_new,Y_new,X_new.shape[0])
    return X_new,Y_new




def downsample(inp_col, f, N):
    inp_col_float = [transf(i) for i in inp_col]
    inp_matr = np.reshape(inp_col_float, (-1, f))

    t = np.mean(inp_matr, axis=1)

    t1 = np.reshape(t, (N, -1))

    # print np.size(t1,0),np.size(t1,1),t1[0,:]
    return t1


def nextBatch(Data, start, finish):
    return Data[start:finish, ]




#generate a random subsample based only on apneic events (this is why we need the labels Y)
def RandomSubsampleCond(X,Y,size1,c):
    Y_Appos=[i for i in range(len(Y)) if(Y[i]==c)]
    #print("yapos:",Y_Appos)
    X_new,Y_new=X[Y_Appos,],Y[Y_Appos,]
    perm=np.random.choice(len(Y_Appos),size=size1)
    return X_new[perm,],Y_new[perm,]
 
def RandomSubsample(X,Y,size1,Z=np.zeros(1),sd=-1):
    
    if(sd==-1):
        perm=np.random.permutation(len(Y))
    elif(sd>0):
        perm=np.random.RandomState(seed=sd).permutation(len(Y))
 

    #print(*perm)
    perm_test=perm[:size1]
    perm_train=perm[size1:]

    #print((perm_train),len(perm_test))

    tr_Data=X[perm_train]
    tr_Data_L=Y[perm_train]

    tst_Data=X[perm_test]
    tst_Data_L=Y[perm_test]

    if(Z.shape[0]>1):
        print("RandomSubsample: Using Z...")
        tr_Z=Z[perm_train]
        tst_Z=Z[perm_test]
        return  tr_Data,tr_Data_L,tst_Data,tst_Data_L,tr_Z,tst_Z
    else:
        return tr_Data,tr_Data_L,tst_Data,tst_Data_L



def get_Batch(X,Y,sz):
    perm=[(random.randint(0,X.shape[0])-1) for j in range(sz)]
    batch_x=X[perm,:]
    batch_y=Y[perm,:]

    return batch_x,batch_y





#Corrupt labels based on the noise transition matrix
def CorruptLabels(Y,Qn,qMx=100):
    Y_new=[]

    for i in range(len(Y)):
        q=random.randint(0,qMx-1)


        c,j=0,0
        while(1):

            if(c<=q<c+Qn[Y[i],j]):break
            c=c+Qn[Y[i],j]
            j+=1
            if(j==9):break

        #print("Label Old",Y[i],"Number:",q,"Matrix value",c,"Label New:",j,"\n")

        Y_new.append(j)

    cnt_changed=sum([1 for i in range(len(Y)) if(Y[i]!=Y_new[i])])
    print("Count Changed:",cnt_changed)
    indx_changed=np.array([i for i in range(len(Y)) if(Y[i]!=Y_new[i])])

    return np.array(Y_new),indx_changed


