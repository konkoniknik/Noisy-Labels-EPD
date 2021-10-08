#%%writefile pipi2.py

import sys,os
import tensorflow as tf
import numpy as np
import time
import help_functions as h
import argparse

    
n_classes=10


def create_Classifier(soft=0):
  model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=8, kernel_size=(5, 5), activation=tf.nn.elu,padding="same"),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), activation=tf.nn.elu,padding="same"),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(filters=32, kernel_size=(5, 5), activation=tf.nn.elu,padding="same"),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='elu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(256, activation='elu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(n_classes)])
  
  if(soft==1): model.add(tf.keras.layers.Softmax())
  return model


def Train(model,trD,trL,tstD,tstL,N=20,show_acc=False,detailed=True):
    if(detailed):
      s_a,s_spec,s_sens,s_k=[],[],[],[]
      for i in range(N):
        print("\n",i)
        model.fit(trD, trL, epochs=1,batch_size=128,verbose=1,validation_data=(tstD,tstL))
        preds=model.predict(tstD)
        a,_,_,_,k=h.print_scores(np.argmax(preds,1),np.argmax(tstL,1))

        s_a.append(a)
        s_k.append(k)

      if(show_acc):print("Accuracy Performance for",N," epochs is:",s_a)

      #Calculate the average ofthe 10 last performances
      return sum(s_a[-10:])/10,sum(s_k[-10:])/10
    
    else:
      model.fit(trD, trL, epochs=N,batch_size=128,verbose=1,validation_data=(tstD,tstL))


def init_weights(model, init,weights=None):
    if weights is None:
        weights = model.get_weights()
    weights = [init(w.shape).numpy() for w in weights]
    model.set_weights(weights)


def labelSusp(Indx,t_l,susp_l,cnt_l):
  for i in range(len(t_l)):
    cnt_l[Indx[i]]+=1
    susp_l[Indx[i]]=susp_l[Indx[i]]+t_l[i]

  return susp_l,cnt_l



######################################Preprocess###########################
def run_test(base_epoch=4,K=50,train_size=40000,epd_flag=1 ):

  #Extract MNIST
  (X_train, y_train), (X_test, y_test)=tf.keras.datasets.mnist.load_data(path='mnist.npz')

  #Rescale to 0-1 and reshape
  X_train=X_train/255
  X_test=X_test/255

  X_train=np.reshape(X_train,[-1,28,28,1])
  X_test=np.reshape(X_test,[-1,28,28,1])


  print("Data shape",X_train.shape,np.amax(X_train),np.amin(X_train))

  #########################################################################################

  #Old contains the uncorrupted labels
  y_trainOld=y_train


  #Noise transition matrix example symmetrical case
  bv,tv,qBig=4960,560,10000#2494,834,10000 #7975,225,10000#3700,700,10000 #50:4960,560,10000#2494,834,10000 #Uniform label noise (symmetry)
  Q=[[bv,tv,tv,tv,tv,tv,tv,tv,tv,tv],
    [tv,bv,tv,tv,tv,tv,tv,tv,tv,tv],     
    [tv,tv,bv,tv,tv,tv,tv,tv,tv,tv],     
    [tv,tv,tv,bv,tv,tv,tv,tv,tv,tv],     
    [tv,tv,tv,tv,bv,tv,tv,tv,tv,tv],     
    [tv,tv,tv,tv,tv,bv,tv,tv,tv,tv],         
    [tv,tv,tv,tv,tv,tv,bv,tv,tv,tv],     
    [tv,tv,tv,tv,tv,tv,tv,bv,tv,tv],     
    [tv,tv,tv,tv,tv,tv,tv,tv,bv,tv],     
    [tv,tv,tv,tv,tv,tv,tv,tv,tv,bv]]
     

  print("Corrupting Labels...")
  y_train,index_ch=h.CorruptLabels(y_trainOld,np.array(Q),qMx=qBig)
  flag_corrupted=np.array([1 if(i in index_ch) else 0 for i in range(len(y_train))])
 
  print("X_ train shape", X_train.shape)
    
  ##########################################################################################

  #Create validation set after corruption: Assumes corrupted val set
  X_val,y_val=X_train[55000:,],y_train[55000:]
  X_train,y_train=X_train[:55000,],y_train[:55000]

  print("Some more shapes:",X_train.shape,y_train.shape,X_val.shape,y_val.shape)
 
  #Transform to one hot vectos
  train_CategoriesOld=h.num_to_cat(y_trainOld,n_classes)

  train_Categories=h.num_to_cat(y_train,n_classes)
  test_Categories=h.num_to_cat(y_test,n_classes)
  val_Categories=h.num_to_cat(y_val,n_classes)

  print("Train Classifier....")
  Classifier=create_Classifier()
  Classifier.compile(optimizer='adam',     
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

  Classifier.build(input_shape=(None,28,28,1))
  #Set up the initializer. We ll need it for restarting our model
  initializer = tf.keras.initializers.GlorotUniform()
  l_softmx=tf.keras.layers.Softmax()

  ####Running EPD:
  #EPD or vanilla run? 
  print("EPD?",epd_flag) 
  if(epd_flag==1):
    
    print("Running EPD:")

    cnt_sumPreds_Tr=np.zeros(X_train.shape[0])

    sumPreds_Tr=np.zeros((X_train.shape[0],n_classes))

    train_Indx=np.reshape(np.arange(X_train.shape[0]),[-1,1])#guides us which data are chosen by each net


    for k in range(K):


      init_weights(Classifier,initializer)


      trEval_Data,trEval_Labels,tr_D,tr_L,trEval_Indx,_=h.RandomSubsample(X_train,y_train,train_size,train_Indx)

      tr_L=h.num_to_cat(tr_L,n_classes)
      print("Model ",k,", train shape:",tr_D.shape,tr_L.shape,"trEval.shape",trEval_Data.shape)

      Train(Classifier,tr_D,tr_L,X_val,val_Categories,N=base_epoch,detailed=False)
                 
      ls_Tr=l_softmx(Classifier(trEval_Data)).numpy()
          
      preds_Tr=np.argmax(ls_Tr,1)

      preds_TrOH=h.num_to_cat(preds_Tr,n_classes)

      ###adding predictions for the evaldata
      sumPreds_Tr,cnt_sumPreds_Tr=labelSusp(trEval_Indx,preds_TrOH,sumPreds_Tr,cnt_sumPreds_Tr)

      print("Some preds:",[sumPreds_Tr[i,]/cnt_sumPreds_Tr[i] for i in range(10)])
      print("Sum of 0s",sum([1 for i in range(cnt_sumPreds_Tr.shape[0]) if(cnt_sumPreds_Tr[i]==0)]))


    #For the small runs of the K models graph
    X_train=np.array([X_train[i,] for i in range(len(sumPreds_Tr)) if (cnt_sumPreds_Tr[i]>0)])
    sumPreds_logits_Tr=np.array([sumPreds_Tr[i]/cnt_sumPreds_Tr[i] for i in range(len(sumPreds_Tr))if (cnt_sumPreds_Tr[i]>0)])
    print("\n Train Data Size", X_train.shape,sumPreds_logits_Tr.shape,sumPreds_logits_Tr[0:10])

        
    
    
    
  #################################Final evaluation() EPD or Baseline##############
  print("Train Classifier....")
  h_Ev=create_Classifier(soft=1)
  h_Ev.compile(optimizer='adam',
      loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
      metrics=['accuracy'])
  

  if(epd_flag==1):
   
    print("EPD...")
    print("some EPs",sumPreds_logits_Tr[:50,:])
    
    print("EP-Entropy (or one-hot 'self-'cross-entropy??):")
    cce = tf.keras.losses.CategoricalCrossentropy()
    print(cce(sumPreds_logits_Tr,sumPreds_logits_Tr).numpy())
 
    acc,k=Train(h_Ev,X_train,sumPreds_logits_Tr,X_test,test_Categories,N=200)
 

  else:
    #Perform normal training: just train a classifier with the corrupted labels with the same evaluation protocol
    print("Normal...")
    acc,k=Train(h_Ev,X_train,train_Categories,X_test,test_Categories,N=200)
    

  print("Final Res:Acc,kappa",acc,k)
  h_Ev.summary()
    


#Runs a test. Corrupts labels based on symmetric label noise and then uses EPD to produce 
#probabilistic outputs or uses the corrupted labels to train a classifier for 200 epochs, taking the average performance over the last 10 epochs.
#based on (https://arxiv.org/pdf/1804.06872.pdf  (Co-Teaching))

parser = argparse.ArgumentParser(description='Configs an EPD ensemble')
parser.add_argument('--base_epoch', metavar='path', required=False,
                        help='how many epochs to train the base models of the ensemble')
parser.add_argument('--K', metavar='path', required=False,
                        help='how many models included in the ensemble')
parser.add_argument('--train_size', metavar='path', required=False,
                        help='train size for the members of the ensemble')
parser.add_argument('--epd_flag', metavar='path', required=False,
                        help='will we run EPD or simply train on noisy labels? (0 or 1)')

parser.add_argument('--GPU', metavar='path', required=False,
                        help='which GPU shall we run?(starting from 0)')




args = parser.parse_args()

if args.GPU!=None:
    os.environ["CUDA_VISIBLE_DEVICES"]=args.GPU


base=4 if args.base_epoch==None else int(args.base_epoch)
member_num=30 if args.K==None else int(args.K)
tr_sz=40000 if args.train_size==None else int(args.train_size)
epd_fl=1 if args.epd_flag==None else int(args.epd_flag)


run_test(base_epoch=base,K=member_num,train_size=tr_sz,epd_flag=epd_fl)


