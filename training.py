import dataBuilder as dtb
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D

K=2     #Number of Transmitters
P=10    #Max transmit power constraint
load= True #Set to True in pre-initialized experts has to be used
InitCSI=[[0,0],[0.9,1]]   #CSI Levels for the initiliaziations
dir_path = os.path.dirname(os.path.realpath(__file__))  #Current path
drop_out_prob=1
dedug = False




def gatingNet(sigma):
    """
    :param sigma: CSI noise level estimate
    :return: Expert weighting
    """
    with tf.variable_scope('gatingNet'):
        dense_11 = tf.layers.dense(sigma,5,activation=tf.nn.relu,name="FC1",reuse=tf.AUTO_REUSE)
        dense_2l = tf.layers.dense(dense_11, 5, activation=tf.nn.relu, name="FC2", reuse=tf.AUTO_REUSE)
        w_vec = tf.layers.dense(dense_2l, 2, activation=None, name="FC3",reuse=tf.AUTO_REUSE)
    return w_vec

def expert(x,keep_prob,name):
    """
    :param x: Expert input [CSI,CSI noise level estimate]
    :param keep_prob: DropOut probability
    :return: Transmit power level
    """
    with tf.name_scope(str(name)):
        output = []
        dense_11 = tf.layers.dense(tf.concat([x[0][:,0:4]/2.5,x[0][:,4:6]],1),10,activation=tf.nn.relu,name=str(name)+"FC1.1")
        drop_out_11 = tf.nn.dropout(dense_11, keep_prob)
        dense_12 = tf.layers.dense(drop_out_11, 10, activation=tf.nn.relu, name=str(name)+"FC2.1")
        drop_out_12 = tf.nn.dropout(dense_12, keep_prob)
        dense_13 = tf.layers.dense(drop_out_12, 10, activation=tf.nn.relu, name=str(name)+"FC3.1")
        out_1 = tf.layers.dense(dense_13, 1, activation=tf.nn.sigmoid, name=str(name)+"FC5.1")*P
        output.append(out_1)
        dense_21 = tf.layers.dense(tf.concat([x[1][:, 0:4] / 2.5, x[1][:, 4:6]], 1), 10, activation=tf.nn.leaky_relu, name=str(name) + "FC1.2")
        drop_out_21 = tf.nn.dropout(dense_21, keep_prob)
        dense_22 = tf.layers.dense(drop_out_21, 10, activation=tf.nn.relu, name=str(name) + "FC2.2")
        drop_out_22 = tf.nn.dropout(dense_22, keep_prob)
        dense_23 = tf.layers.dense(drop_out_22, 10, activation=tf.nn.relu, name=str(name) + "FC3.2")
        out_2 = tf.layers.dense(dense_23, 1, activation=tf.nn.sigmoid, name=str(name) + "FC5.2") * P
        output.append(out_2)
    return output

def formatData(G_hat,sigma):
    trainDataset = []
    for n in range(K):
        enlargedG_hat = np.concatenate(
            [np.concatenate([G_hat[n], G_hat[n]], -1), np.concatenate([G_hat[n], G_hat[n]], -1)], 1)
        enlargedsigma = np.concatenate([sigma, sigma], -1)
        trainDataset.append(
            np.column_stack((np.reshape(enlargedG_hat[:, n:n + K, n:n + K], (-1, K * K)), enlargedsigma[:, n:n + K])))
    return trainDataset

def supLoss(p_vec,labels):
    """
    :param p_vec: output power levels
    :param labels: desired power levels
    :return: avg L2 distance between powers and labels
    """
    l=tf.reshape(tf.stack(p_vec,axis=1),(-1,K))
    loss =tf.reduce_mean( (labels-l)*(labels-l))
    return loss

def labelingFunc(p_vec_1,p_vec_2,G):
    """
        :param p_vec_1: output power levels of user 1
        :param p_vec_2: output power levels of user 2
        :param G: true channel gain matrix
        :return: performance comparison between the two experts
        """
    p_vec = tf.reshape(tf.stack(p_vec_1, axis=1), (-1, K))
    P_vect_epoch_TF = tf.reshape(p_vec, [-1, K])
    Desired_channels_epoch_TF = tf.matrix_diag_part(G)
    signal_power_epoch_TF = tf.multiply(P_vect_epoch_TF, Desired_channels_epoch_TF)
    P_vect_prime_epoch_TF = tf.reshape(P_vect_epoch_TF, [-1, K, 1])
    interference_power_epoch_TF = tf.reshape(tf.matmul(G, P_vect_prime_epoch_TF), [-1, K]) - signal_power_epoch_TF
    SINR_epoch_TF = tf.div(signal_power_epoch_TF, interference_power_epoch_TF + 1.0)
    sum_rate_epoch_TF = tf.reduce_sum(tf.math.log(1.0 + SINR_epoch_TF), 1)/tf.math.log(2.0)

    p_vec = tf.reshape(tf.stack(p_vec_2, axis=1), (-1, K))
    P_vect_epoch_TF = tf.reshape(p_vec, [-1, K])
    Desired_channels_epoch_TF = tf.matrix_diag_part(G)
    signal_power_epoch_TF = tf.multiply(P_vect_epoch_TF, Desired_channels_epoch_TF)
    P_vect_prime_epoch_TF = tf.reshape(P_vect_epoch_TF, [-1, K, 1])
    interference_power_epoch_TF = tf.reshape(tf.matmul(G, P_vect_prime_epoch_TF), [-1, K]) - signal_power_epoch_TF
    SINR_epoch_TF = tf.div(signal_power_epoch_TF, interference_power_epoch_TF + 1.0)
    sum_rate_epoch_TF_2= tf.reduce_sum(tf.math.log(1.0 + SINR_epoch_TF), 1)/tf.math.log(2.0)

    res=tf.stack((tf.cast(sum_rate_epoch_TF, tf.float32),tf.cast(sum_rate_epoch_TF_2, tf.float32)),axis=1)
    labels=tf.cast(tf.stack((res[:,0]>res[:,1],res[:,1]>res[:,0]),axis=1),tf.float32)
    weights=tf.abs(res[:,0]-res[:,1])
    norm_res=tf.stack((res[:,0]-tf.math.reduce_min(res,1),res[:,1]-tf.math.reduce_min(res,1)),axis=1)
    return labels,weights

def unsupLoss(p_vec,G):
    """
    :param p_vec: output power levels
    :param G: true channel gain matrix
    :return: sum-rate corresponding to the (p_vec,G) pair
    """
    p_vec=tf.reshape(tf.stack(p_vec, axis=1), (-1, K))
    P_vect_epoch_TF = tf.reshape(p_vec, [-1, K])
    Desired_channels_epoch_TF = tf.matrix_diag_part(G)
    signal_power_epoch_TF = tf.multiply(P_vect_epoch_TF, Desired_channels_epoch_TF)
    P_vect_prime_epoch_TF = tf.reshape(P_vect_epoch_TF, [-1, K, 1])
    interference_power_epoch_TF = tf.reshape(tf.matmul(G, P_vect_prime_epoch_TF), [-1, K])- signal_power_epoch_TF
    SINR_epoch_TF = tf.div(signal_power_epoch_TF, interference_power_epoch_TF + 1.0)
    sum_rate_epoch_TF =-tf.reduce_mean(tf.reduce_sum(tf.math.log(1.0 + SINR_epoch_TF), 1))/tf.math.log(2.0)
    return sum_rate_epoch_TF



expIn1=[]
expIn2=[]
for n in range(K):
    expIn1.append(tf.placeholder(tf.float32, shape=(None, K * K + K)))  #Inputs placeholders for the pair of experts 1
    expIn2.append(tf.placeholder(tf.float32, shape=(None, K * K + K)))  #Inputs placeholders for the pair of experts 2
keep_prob = tf.placeholder(tf.float32, shape=(None)) #Drop-out probability
exp1=expert(expIn1,keep_prob,'Pair1')  #output of the first pair of experts
exp2=expert(expIn2,keep_prob,'Pair2')  #output of the second pair of experts
labels=tf.placeholder(tf.float32, shape=(None, K))  #Label for Supervised learning phase
G_1=tf.placeholder(tf.float32, shape=(None,K,K))
unLoss1=unsupLoss(exp1,G_1)
supLoss1=supLoss(exp1,labels)
unOpt1 = tf.train.AdagradOptimizer(learning_rate=0.01).minimize(loss=unLoss1, global_step=tf.train.get_global_step())
supOpt1 = tf.train.AdagradOptimizer(learning_rate=0.01).minimize(loss=supLoss1, global_step=tf.train.get_global_step())
G_2=tf.placeholder(tf.float32, shape=(None,K,K))
unLoss2=unsupLoss(exp2,G_2)
supLoss2=supLoss(exp2,labels)
unOpt2 = tf.train.AdagradOptimizer(learning_rate=0.01).minimize(loss=unLoss2, global_step=tf.train.get_global_step())
supOpt2 = tf.train.AdagradOptimizer(learning_rate=0.01).minimize(loss=supLoss2, global_step=tf.train.get_global_step())

#Building the gatingNet
sigmaHat=tf.placeholder(tf.float32, shape=(None,K))
gatingOut=gatingNet(sigmaHat)
#Building the MoE output
MoEOut=[]
MoEOut.append(tf.add(tf.multiply(exp1[0][:,-1],gatingOut[:,0]),tf.multiply(exp2[0][:,-1],gatingOut[:,1])))
MoEOut.append(tf.add(tf.multiply(exp1[1][:,-1],gatingOut[:,0]),tf.multiply(exp2[1][:,-1],gatingOut[:,1])))
gatingLabels=tf.placeholder(tf.float32, shape=(None,K))
weights=tf.placeholder(tf.float32, shape=(None))
G_Moe=tf.placeholder(tf.float32, shape=(None,K,K))
y= labelingFunc(exp1,exp2,G_Moe)

'''OLD LOSS FUNCTION'''
#reg2= tf.sqrt((tf.reduce_mean(gatingOut[:,0])-tf.reduce_mean(gatingOut[:,1]))*(tf.reduce_mean(gatingOut[:,0])-tf.reduce_mean(gatingOut[:,1])))
#reg=-tf.reduce_mean(tf.math.sqrt(tf.math.square((gatingOut[:,0]-tf.reduce_mean(gatingOut[:,0])))))
#gatLoss = -tf.reduce_mean(tf.reduce_sum(tf.math.multiply(gatingOut,gatingLabels),1))+reg

gatLoss=tf.reduce_mean(weights*tf.nn.softmax_cross_entropy_with_logits(labels=gatingLabels,logits=gatingOut))
gating_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"gatingNet")
gatOpt = tf.train.AdagradOptimizer(learning_rate=0.1).minimize(loss=gatLoss, global_step=tf.train.get_global_step())




#Initilialize the two pair of experts to different CSI levels regions. One close to (0,0) the other close to (1,1).
saver = tf.train.Saver()
init = tf.global_variables_initializer()
if not load:
    initSample1=30000   #Samples for perfect CSI initialziation
    initIteration1=1000    #Iterations for perfect CSI initialziation
    initSample2=30000   #Samples for imperfect CSI initialziation
    initIteration2=1000    #Iterations for imperfect CSI initialziation
    '''
    Initialization loop:
     -Supervised training using WMMSE solution (initIteration1 epochs, initSample1 samples)
     -Unsupervised training maximing sum-rate  (initIteration1 epochs, initSample1 samples)
     -For the second pair, unsupervised training on the noisy CSI regime    (initIteration2 epochs, initSample2 samples)
     '''
    with tf.Session() as sess:
        sess.run(init)
        G, G_hat, sigma = dtb.buildDataSigma(initSample1, InitCSI[0])

        initializationData=formatData(G_hat,sigma)
        #SUPERVISED LEARNING PHASE
        WMSEE = dtb.dataLabeling(G)     #Labeling according to WMSEE Solution
        #Feeding data into placeholders
        feed = {labels: WMSEE,keep_prob:drop_out_prob}
        for k in range(K):
            feed[expIn1[k]] = initializationData[k]
            feed[expIn2[k]] = initializationData[k]
        print('SUPERVISED INITIALIZATION')
        for i in range(initIteration1):
            _,_, lossVal1, lossVal2 = sess.run([supOpt1,supOpt2, supLoss1, supLoss2], feed_dict=feed)
            if(i%200==0):
                print('Iteration: '+str(i)+'    |  loss Value EXP1: '+str(lossVal1)+'   |  loss Value EXP2: '+str(lossVal2))
        #UNSUPERVISED LEARNING PHASE
        feed[G_1] = G
        feed[G_2] = G
        print('UNSUPERVISED INITIALIZATION')
        for i in range(initIteration1):
            _,_, lossVal1, lossVal2= sess.run([unOpt1,unOpt2, unLoss1, unLoss2], feed_dict=feed)
            if (i % 200 == 0):
                print( 'Iteration: ' + str(i) + '   |  loss Value EXP1: ' + str(lossVal1) + '   |  loss Value EXP2: ' + str(lossVal2))
        #UNSUPERVISED LEARNING PHASE LOW CSI QUALITY REGIME
        G, G_hat, sigma = dtb.buildDataSigma(int(initSample1),  InitCSI[1])
        '''
        G_p, G_hat_p, sigma_p = dtb.buildDataSigma(int(initSample1), [1, 1])
        G = np.concatenate([G, G_p], axis=0)
        G_hat = np.concatenate([G_hat, G_hat_p], axis=1)
        sigma = np.concatenate([sigma, sigma_p], axis=0)
        '''
        initializationData = formatData(G_hat, sigma)
        feed = {G_2: G, keep_prob: drop_out_prob}
        print('LOW QUALITY CSI UNSUPERVISED INITIALIZATION')
        for k in range(K):
            feed[expIn2[k]] = initializationData[k]
        for i in range(initIteration2*2):
            _, lossVal2 = sess.run([unOpt2, unLoss2], feed_dict=feed)
            if (i % 200 == 0):
                print('Iteration: ' + str(i)+ ' |  loss Value EXP2: ' + str(lossVal2))
        if(dedug):
            G, G_hat, sigma = dtb.buildDataSigma(10000,  InitCSI[0])
            testingData = formatData(G_hat, sigma)
            feed ={G_1: G, G_2: G,keep_prob:1}
            for k in range(K):
                feed[expIn1[k]] = testingData[k]
                feed[expIn2[k]] = testingData[k]
            lossVal1,lossVal2 = sess.run([unLoss1,unLoss2], feed_dict=feed)
            print('FINAL '+str(InitCSI[0])+' :')
            print(lossVal1)
            print(lossVal2)

            G, G_hat, sigma = dtb.buildDataSigma(10000, InitCSI[1])
            testingData = formatData(G_hat, sigma)
            feed ={G_1: G, G_2: G,keep_prob:1}
            for k in range(K):
                feed[expIn1[k]] = testingData[k]
                feed[expIn2[k]] = testingData[k]
            lossVal1,lossVal2 = sess.run([unLoss1,unLoss2], feed_dict=feed)
            print('FINAL '+str(InitCSI[1])+' :')
            print(lossVal1)
            print(lossVal2)
        saver.save(sess, dir_path + '/initializedExperts/model.ckpt')




saver = tf.train.Saver()
init = tf.global_variables_initializer()
#Training parameters for the MoE model
nSamples=100000
batchSize=1000
gatEpochs=200
expBatchSize=1000
expEpochs=200
EMIter=20
delta=20
yc = np.linspace(0, 1, delta)
xc = np.linspace(0, 1, delta)
xvc, yvc = np.meshgrid(xc, yc)
xvc=np.reshape(xvc,(-1,1))
yvc=np.reshape(yvc,(-1,1))
tasks_temp=np.concatenate((xvc,yvc), axis=-1)
sigmaHats=tasks_temp[tasks_temp[:,0]<=tasks_temp[:,1]]

with tf.Session() as sess:
        sess.run(init)
        saver.restore(sess, dir_path + '/initializedExperts/model.ckpt')
        G_d, G_hat, sigma = dtb.buildDataCont(nSamples, 1, 0) #Channel estimates and CSI estimates
        '''
        trainDataset=[[],[]]
        G_d, G_hat, sigma = dtb.buildDataSigma(30000, sigmaHats[0, :], np.random.randint(1, 1000, 1))
        for t in range(1,53):
            G_d_t, G_hat_t, sigma_t = dtb.buildDataSigma(30000, sigmaHats[t, :], np.random.randint(1, 1000, 1))
            G_d = np.concatenate((G_d,G_d_t))
            G_hat = np.concatenate((G_hat,G_hat_t),axis=1)
            sigma = np.concatenate((sigma,sigma_t))
        '''
        trainDataset = formatData(G_hat, sigma) # Formatting the TrainingSet into a K element list, one for each DM
        '''
        plt.scatter(sigma[:,0], sigma[:,1])
        plt.show()
        '''
        #

        for ei in range(EMIter):
            print('E-M STEP: ' + str(ei))
            indices = np.random.randint(1, nSamples, batchSize)
            feed = {G_Moe: G_d[indices, :, :], keep_prob: 1, sigmaHat: sigma[indices, :]}
            for k in range(K):
                feed[expIn1[k]] = trainDataset[k][indices, :]
                feed[expIn2[k]] = trainDataset[k][indices, :]
            test = sess.run([y], feed_dict=feed);
            t = sigma[indices, :]
            feed[gatingLabels] = test[0][0]
            feed[weights] = test[0][1]
            for i in range(gatEpochs):
                _, gatLossVal,gOut,t,e1,e2 = sess.run([gatOpt, gatLoss,gatingOut,y,exp1,exp2], feed_dict=feed)
                if (i  == gatEpochs-1 and dedug):
                    test = np.argmax(sess.run(gatingOut, feed_dict={sigmaHat: sigmaHats}),1)
                    fig = plt.figure()
                    plt.scatter(sigmaHats[:,0],sigmaHats[:,1], c=test)
                    plt.show()


            for i in range(expEpochs):
                indices = np.random.randint(1, nSamples, expBatchSize)
                feed = {sigmaHat: sigma[indices, :]}
                gOutval = sess.run([gatingOut], feed_dict=feed)
                gOut = np.argmax(gOutval[0], 1)
                indices_1 = indices * (-gOut + 1)
                indices_1 = indices_1[indices_1 != 0]
                indices_2 = indices * (gOut)
                indices_2 = indices_2[indices_2 != 0]
                feed = {keep_prob: drop_out_prob}
                for k in range(K):
                    t = trainDataset[k][indices_1, :]
                    feed[expIn1[k]] = trainDataset[k][indices_1, :]
                    t2 = trainDataset[k][indices_2, :]
                    feed[expIn2[k]] = trainDataset[k][indices_2, :]
                t2 = G_d[indices_1, :, :]
                feed[G_1] = G_d[indices_1, :, :]
                feed[G_2] = G_d[indices_2, :, :]
                _, _, lossVal1, lossVal2 = sess.run([unOpt1, unOpt2, unLoss1, unLoss2], feed_dict=feed)
                if (i % 10 == 0):
                    print('Iteration: ' + str(i) + '    |  loss Value EXP1: ' + str(lossVal1) + '   |  loss Value EXP2: ' + str(lossVal2))

        if (True):
            val = []
            for i in range(len(tasks_temp[:, 0])):
                G_test, G_hat_test, sigma_test = dtb.buildDataSigma(10000, tasks_temp[i, :])
                testSet = formatData(G_hat_test, sigma_test)
                feed = {sigmaHat: sigma_test}
                gOut = np.argmax(sess.run([gatingOut], feed_dict=feed), -1)
                if (gOut[0, 0] == 0):
                    feed = {keep_prob: 1}
                    for k in range(K):
                        feed[expIn1[k]] = testSet[k]
                    feed[G_1] = G_test
                    lossVal1 = sess.run([unLoss1], feed_dict=feed)
                    val.append(lossVal1)
                else:
                    feed = {keep_prob: 1}
                    for k in range(K):
                        feed[expIn2[k]] = testSet[k]
                    feed[G_2] = G_test
                    lossVal2 = sess.run([unLoss2], feed_dict=feed)
                    val.append(lossVal2)
            val = np.array(val)
            fig = plt.figure()
            ax = Axes3D(fig)
            xdata = np.linspace(0, 1, delta)
            ydata = np.linspace(0, 1, delta)
            xdata, ydata = np.meshgrid(xdata, ydata)
            ax.scatter(xdata, ydata, -val, label='TDNN', c='black')
            ax.scatter(xdata, ydata, -val * 0 + 3.7876, label='TDNN', c='red')
            ax.legend()
            plt.show()
        saver.save(sess, dir_path + '/finalModel/model0.2.ckpt')
