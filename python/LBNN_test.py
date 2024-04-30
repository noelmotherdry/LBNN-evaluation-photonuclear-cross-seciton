import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

import numpy as np
import math
#from scipy.stats import norm
import matplotlib.pyplot as plt


cuda_is_available = 0#torch.cuda.is_available()


class Linear_BBB(nn.Module):
    """
        Layer of our BNN.
    """
    def __init__(self, input_features, output_features, prior_var=1.):
        """
            Initialization of our layer : our prior is a normal distribution
            centered in 0 and of variance 20.
        """
        # initialize layers
        super().__init__()
        # set input and output dimensions
        self.input_features = input_features
        self.output_features = output_features

        # initialize mu and rho parameters for the weights of the layer
        # 定义一个矩阵，大小为output_features * input_features
        # 并且把这个矩阵转换为Parameter类型。转换后，成为了模型中根据训练可以改动的参数。
        # 使用“nn.Parameter”这个函数的目的也是想让某些变量在学习的过程中不断的修改其值以达到最优化。
        self.w_mu = nn.Parameter(torch.zeros(output_features, input_features))
        self.w_rho = nn.Parameter(torch.zeros(output_features, input_features))

        #initialize mu and rho parameters for the layer's bias
        self.b_mu =  nn.Parameter(torch.zeros(output_features))
        self.b_rho = nn.Parameter(torch.zeros(output_features))

        #initialize weight samples (these will be calculated whenever the layer makes a prediction)
        self.w = None
        self.b = None

        # initialize prior distribution for all of the weights and biases
        self.prior = torch.distributions.Normal(0.5, prior_var)

        if cuda_is_available:
            self = self.cuda()

    def forward(self, input):
        ####这个函数并未被调用，如何理解？
        #对于torch类而言，forward函数即为将该类自己的名称调用做函数
        """
          Optimization process
        """
        # sample weights
        w_epsilon = Normal(0.5, 1).sample(self.w_mu.shape)#!
        # sample bias
        b_epsilon = Normal(0.5, 1).sample(self.b_mu.shape)##!
        if cuda_is_available:
            w_epsilon=w_epsilon.cuda()
            b_epsilon=b_epsilon.cuda()




        self.w = self.w_mu + torch.log(1+torch.exp(self.w_rho)) * w_epsilon#初始分布为u+ε*log(1+exp(ρ))
        self.b = self.b_mu + torch.log(1+torch.exp(self.b_rho)) * b_epsilon

        # record log prior by evaluating log pdf of prior at sampled weight and bias

        w_log_prior = self.prior.log_prob(self.w)
        b_log_prior = self.prior.log_prob(self.b)
        self.log_prior = torch.sum(w_log_prior) + torch.sum(b_log_prior)

        # record log variational posterior by evaluating log pdf of normal distribution defined by parameters with respect at the sampled values
        self.w_post = Normal(self.w_mu.data, torch.log(1+torch.exp(self.w_rho)))
        self.b_post = Normal(self.b_mu.data, torch.log(1+torch.exp(self.b_rho)))
        self.log_post = self.w_post.log_prob(self.w).sum() + self.b_post.log_prob(self.b).sum()

        return F.linear(input, self.w, self.b)


class MLP_BBB(nn.Module):
    def __init__(self, input_units, hidden_units, Lorentz_units, output_units, noise_tol=0.1,  prior_var=1.):

        # initialize the network like you would with a standard multilayer perceptron, but using the BBB layer
        super().__init__()
        #产生一个可以优化的参数矩阵，大小为input_units * hidden_units
        #定义了这个参数矩阵，就相当于定义了一个映射，由input_units维度的输入数据，映射到hidden_units个节点参数
        self.hidden = Linear_BBB(input_units, hidden_units, prior_var=prior_var)#i*h矩阵，i节点h隐藏参数
        #self.hidden2 = Linear_BBB(hidden_units, hidden_units, prior_var=prior_var)
        # 由hidden_units个节点参数，映射到洛伦兹参数
        self.out = Linear_BBB(hidden_units,Lorentz_units, prior_var=prior_var)#h*L矩阵，h隐藏参数L洛伦兹参数
        # 由洛伦兹参数，映射到output_units节点的输出数据
        #self.Lorentz = Linear_BBB(hidden_units, Lorentz_units, prior_var=prior_var)
        # 应该是在这里插入物理公式定义
        self.noise_tol = noise_tol # we will use the noise tolerance to calculate our likelihood
        if cuda_is_available:
            self = self.cuda()

    def forward(self, x, t):
        # again, this is equivalent to a standard multilayer perceptron

        x = torch.sigmoid(self.hidden(x)) #hidden(x)：用x计算出节点参数；torch.sigmoid(t)则是做1/(1+exp(-t))计算
        #x = torch.sigmoid(self.hidden2(x))

        x = torch.sigmoid(self.out(x)) #再用做了做1/(1+exp(-t))计算的节点，计算出输出数据y



        x = Lorentz_function_with_R(x, t)
        #应该是在这里插入物理公式
        return x

    def log_prior(self):
        # calculate the log prior over all the layers
        #self.out.
        return self.hidden.log_prior + self.out.log_prior #+self.hidden2.log_prior

    def log_post(self):
        # calculate the log posterior over all the layers
        return self.hidden.log_post + self.out.log_post #+self.hidden2.log_post

    def sample_elbo(self, input, input_t, target, samples):
        # we calculate the negative elbo, which will be our loss function
        #initialize tensors
        # outputs 有三个维度，第一个维度是抽样数samples；第二维度是训练数据的个数；第三个维度是LBNN的输出维度，其中的0:1是能谱，1:7是贝叶斯网络输出的洛伦兹参数
        outputs = torch.zeros(samples, target.shape[0], 6)
        log_priors = torch.zeros(samples)
        log_posts = torch.zeros(samples)
        log_likes = torch.zeros(samples)
        if cuda_is_available:
            outputs=outputs.cuda()
            log_likes=log_likes.cuda()
            log_posts=log_posts.cuda()
            log_priors=log_priors.cuda()
        # make predictions and calculate prior, posterior, and likelihood for a given number of samples
        for i in range(samples):
            #print(self(input,input_t).shape,outputs.shape)
            #print(input.shape,input_t.shape)
            outputs[i] = self(input, input_t) # make predictions

            log_priors[i] = self.log_prior() # get log prior
            log_posts[i] = self.log_post() # get log variational posterior
            #outputs[i,:,0:1]是能谱，outputs[i,:,1:7]是洛伦兹参数；抽样只抽能谱的，因为洛伦兹参数是没有训练数据的
            #print("0:====",outputs[i,:,0:1])
            log_likes[i] = Normal(outputs[i,:,0:1], self.noise_tol).log_prob(target).sum() # calculate the log likelihood
        # calculate monte carlo estimate of prior posterior and likelihood

        log_prior = log_priors.mean()
        log_post = log_posts.mean()
        log_like = log_likes.mean()
        # calculate the negative elbo (which is our loss function)
        loss = log_post - log_prior - log_like
        return loss



def Lorentz_function_with_R(bnn_out, nuclei_info):
    Z = nuclei_info[:,0:1]
    A = nuclei_info[:,1:2]
    N=A-Z
    e = nuclei_info[:,2:3]
    reaction_channel = nuclei_info[:,3:4]
    f=torch.zeros(reaction_channel.shape)
    if cuda_is_available:
        f=f.cuda()
    #(f.shape)
    S2n = nuclei_info[:,4:5]
    sig_QD = nuclei_info[:,5:6]
    s = bnn_out[:,0:1]
    E = bnn_out[:,1:2]
    GAMMA = bnn_out[:,2:3]
    deE = bnn_out[:,3:4]
    #print(torch.div(1, (1 + torch.exp(torch.div(1.0986123 * (e - S2n - torch.div(deE, 2)), deE)))))
    R = torch.div(1, (1 + torch.exp(torch.div(1.0986123 * (e - S2n - torch.div(deE, 2)), deE))))
    for  i in range(reaction_channel.shape[0]):

        if  reaction_channel[i] == 1:#represent reaction chanenl (g,n)
            f[i]=R[i]
        elif reaction_channel[i] == 2:#represent reaction chanenl (g,2n)
            f[i]=(1-R[i])
        elif reaction_channel[i] == 3:
            f[i]=1
        elif reaction_channel[i] == 4:#represent reaction chanenl (g,xn)
            f[i]=(2-R[i])
    sig_TRK = 60*torch.div(torch.mul(N,Z),A)
    F=0.636619*torch.div(torch.mul(torch.pow(e,2),GAMMA),torch.pow(torch.pow(e,2)-torch.pow(E,2),2)+torch.pow(torch.mul(e,GAMMA),2))
    sig_abs = torch.mul(sig_TRK,torch.mul(s,F))
    sig = torch.mul(sig_abs,f)
    #print(torch.cat((sig,E,GAMMA,s,deE,sig_abs)
    return torch.cat((sig,E,GAMMA,s,deE,sig_abs),dim = 1)





def read_train_data_file(filename1,filename2):
    #文件1为反应截面数据，文件2为核素的各项信息
    #输出结果每列分别是Z，A，e，e_err，sig，sig_err，reaction_channel，
    cs_file=open(filename1,'r',encoding="utf-8")


    A=0
    Z=0
    info = []
    res=[]
    for line in cs_file.readlines():
        curLine = line.strip().split()
        if line[0]=='#' or len(curLine) == 0:
            continue
        Z = int(np.asfarray(curLine[0]))
        A =  int(np.asfarray(curLine[1]))

        if (not len(info) == 0) and (Z == info[0] and A == info[1]) :
            ax = np.array(np.concatenate((np.asfarray(curLine[0:7]), info[2:6])))

            res = np.c_[res, ax]
            #print(np.array(res))
            #exit(0)
        else:
            search_success=0
            info_file = open(filename2, 'r')

            for line2 in info_file.readlines():
                if line2 == '#' :
                    continue
                curLine2 = line2.strip().split()
                #print(int(np.asfarray(curLine[0]))+2,"≠",int(np.asfarray(curLine2[0])),int(np.asfarray(curLine[1])),"≠",int(np.asfarray(curLine2[1])))
                if Z == int(np.asfarray(curLine2[0])) and A == int(np.asfarray(curLine2[1])):
                    info = np.asfarray(curLine2[0:6])
                    search_success=1
                    break
            if not search_success :

                print("原子序数",Z,"原子质量",A,"，警告，未找到相应的核数据！")
                exit("Data error")
            ax=np.array(np.concatenate((np.asfarray(curLine[0:7]),info[2:6])))
            if len(res) == 0:
                res = ax

            else:
                res=np.c_[res,ax]

            info_file.close()
    #res = np.array([list(data) for data in res])



    return np.transpose(np.array(res))


def read_predict_data_file(filename1,filename2):
    #文件1为反应截面数据，文件2为核素的各项信息
    #输出结果每列分别是Z，A，e，reaction_channel，delta，B，Sn，S2n
    cs_file=open(filename1,'r',encoding="utf-8")


    A=0
    Z=0
    info = []
    res=[]
    for line in cs_file.readlines():
        curLine = line.strip().split()
        if line[0] == '#' or len(curLine) == 0:
            continue
        Z = int(np.asfarray(curLine[0]))
        A =  int(np.asfarray(curLine[1]))
        if (not len(info) == 0) and (Z == info[0] and A == info[1]) :
            a = np.array(np.concatenate((np.asfarray(curLine[0:4]), info[2:6])))
            res = np.c_[res, a]
            #print(np.array(res))
            #exit(0)
        else:
            search_success=0
            info_file = open(filename2, 'r')
            for line2 in info_file.readlines():
                if line2 == '#' :
                    continue
                curLine2 = line2.strip().split()
                #print(int(np.asfarray(curLine[0]))+2,"≠",int(np.asfarray(curLine2[0])),int(np.asfarray(curLine[1])),"≠",int(np.asfarray(curLine2[1])))
                if Z == int(np.asfarray(curLine2[0])) and A == int(np.asfarray(curLine2[1])):
                    info = np.asfarray(curLine2[0:6])
                    search_success=1

                    break
            if not search_success :
                print("原子序数",Z,"原子质量",A,"，警告，未找到相应的核数据！")
                exit("Data error")
            a=np.array(np.concatenate((np.asfarray(curLine[0:4]),info[2:6])))
            if len(res) == 0:
                res = a

            else:
                res=np.c_[res,a]
            info_file.close()
    #res = np.array([list(data) for data in res])
    return np.transpose(np.array(res))






#在此处添加文件路径
path = ""

train_data = read_train_data_file(path+"sign2n_EXFOR_with_mass.dat",path+"nulide_info.dat")
#输出结果每列分别是Z，A，e，e_err，sig，sig_err，reaction_channel，delta，B，Sn，S2n


#dealing with y and yerr
y=np.array(train_data[:,4:5])*1000
yerr = np.array(train_data[:,5:6])*1000.
imax = yerr.shape[0]
N_intensify = np.zeros((imax, 1))
for i in range(imax):
    if(yerr[i]==0):
        N_int = 1
    else:
        N_int = math.floor(0.1 * y[i] / yerr[i]  +0.5) + 1
    N_intensify[i] = N_int



#dealing with t[Z,A,e,reaction_channel,S2n,sigma_QD]
t_for_train = np.array(train_data[:,0:3])
t_for_train=np.c_[t_for_train,train_data[:,6]]
t_for_train=np.c_[t_for_train,train_data[:,10]]
t_for_train = np.array(t_for_train)
t = t_for_train

#计算洛伦兹公式中的sigma QD
imax = t.shape[0]
sig_QD = np.zeros((imax, 1))
for i in range(imax):
    if t[i, 0:1] > 2.224:
      sig_QD[i,0:1] = 397.8 *t[i, 0:1]*((t[i, 1:2] - t[i, 0:1]) / t[i, 1:2])*(t[i, 2:3] - 2.224)**1.5 /t[i, 2:3]**3
    if t[i, 0:1] <= 20:
        sig_QD[i, 0:1] = sig_QD[i, 0:1]*math.exp(-73.3 / t[i, 2:3])
    if t[i, 0:1] > 20:
        sig_QD[i, 0:1] = sig_QD[i, 0:1] * (
                    0.083714 - 0.0098343 * t[i, 2:3] + 4.1222 * (t[i, 2:3] * 0.01) ** 2 - 3.4762 * (
                        t[i, 2:3] * 0.01) ** 3 + 0.93537 * (t[i, 2:3] * 0.01) ** 4)

t_for_train = np.c_[t_for_train, sig_QD[:, 0]]#e+Z+A+?+σQD
t_for_train = np.array(t_for_train)




#dealing with x[] train input
#输出数准备，包括弹核Z与A，靶核Z与A，入射能量，产物的Z和A
x_out = train_data[:, 0] #Z
x_out = np.c_[x_out, train_data[:, 1]] #A
x_out = np.c_[x_out, train_data[:, 2]] #能量

# 筛选input自由度   1, 18, 8,5,4
x_for_train = train_data[:, 0:2]#Z\A

x_for_train = np.c_[x_for_train, train_data[:, 7]]#delta
x_for_train = np.c_[x_for_train, train_data[:, 8]]#B
#x_for_train = np.c_[x_for_train, x0to23[:, 24]]
#x_for_train = np.c_[x_for_train, x0to23[:, 44]]
#x_for_train = np.c_[x_for_train, x0to23[:, 45]]
x_for_train = np.c_[x_for_train, train_data[:,9]]#Sn
x_for_train = np.c_[x_for_train, train_data[:,10]]#S2n
x_for_train = x_for_train.tolist()  # 将x从数组转成list




#t_for_train suppose to be the nuclei info that use in the empirical formula in LBNN
predict_data = read_predict_data_file(path+"sign2n_EXFOR_with_mass_to_BNN.dat",path+"nulide_info.dat")


#dealing with t[]
t_for_pre = np.array(predict_data[:,0:4])
t_for_pre=np.c_[t_for_pre,predict_data[:,7]]
t_for_pre = np.array(t_for_pre)
t = t_for_pre
#计算洛伦兹公式中的sigma QD
imax = t.shape[0]
sig_QD = np.zeros((imax, 1))
for i in range(imax):
    if t[i, 0:1] > 2.224:
      sig_QD[i,0:1] = 397.8 *t[i, 0:1]*((t[i, 1:2] - t[i, 0:1]) / t[i, 1:2])*(t[i, 2:3] - 2.224)**1.5 /t[i, 2:3]**3
    if t[i, 0:1] <= 20:
        sig_QD[i, 0:1] = sig_QD[i, 0:1]*math.exp(-73.3 / t[i, 2:3])
    if t[i, 0:1] > 20:
        sig_QD[i, 0:1] = sig_QD[i, 0:1] * (
                    0.083714 - 0.0098343 * t[i, 2:3] + 4.1222 * (t[i, 2:3] * 0.01) ** 2 - 3.4762 * (
                        t[i, 2:3] * 0.01) ** 3 + 0.93537 * (t[i, 2:3] * 0.01) ** 4)

t_for_pre = np.c_[t_for_pre, sig_QD[:, 0]]#e+Z+A+?+σQD
t_for_pre = np.array(t_for_pre)







"""
#dealing with x[] train input
#输出数准备，包括弹核Z与A，靶核Z与A，入射能量，产物的Z和A
x_out = predict_data[:, 0] #Z
x_out = np.c_[x_out, predict_data[:, 1]] #A
x_out = np.c_[x_out, predict_data[:, 2]] #能量
"""
# 筛选input自由度   1, 18, 8,5,4
x_for_pre = np.array(predict_data[:, 0:2])#Z,A
x_for_pre = np.c_[x_for_pre, predict_data[:, 4]]#A+?+?
x_for_pre = np.c_[x_for_pre,predict_data[:, 5]]#???
#x_for_train suppose to be the main train data of the BNN of LBNN
x_for_pre = np.c_[x_for_pre, predict_data[:, 6]]
x_for_pre = np.c_[x_for_pre, predict_data[:, 7]]
#x_for_pre = np.c_[x_for_train, train_data[:,9]]#???
x_for_pre = x_for_pre.tolist()  # 将x从数组转成list



"""
#final dealth
# 数据缩放
cenx = np.std(x_for_train, axis=0)
avex = np.mean(x_for_train, axis=0)
x_for_train = (np.array(x_for_train) - avex) / cenx
x_for_pre = (np.array(x_for_pre) - avex) / cenx
#cent = np.std(t_for_train, axis=0)
#avet = np.mean(t_for_train, axis=0)
#经验公式中使用的数据不可进行缩放
#t_for_train = (np.array(t_for_train) - avet) / cent
#t_for_pre = (np.array(t_for_pre) - avet) / cent
"""
x_for_pre = np.array(x_for_pre) / np.mean(np.array(x_for_train))
x_for_train = np.array(x_for_train) / np.mean(np.array(x_for_train))
#ceny = np.std(y, axis=0)
#avey = np.mean(y, axis=0)
# y = (np.array(y) - avey) / ceny





#神经网络参数
#寻找最佳输入数据组合方法
# 用“for jj in range(23):”做循环，用“epochs = 1000”，“#if epoch > 990:”输出1000个迭代的误差。
# 修改 ”寻找最佳输入数据组合方法“ 框起来的部分，分别根据输出误差，选择1维，2维等输入维度，直到误差不再减小
#for jj in range(23):
input_nodes = 6
hidden_nodes = 15
Lorentz_units = 4
output_nodes = 1


samples_tot = 10
n_have_run = 0
y_samp1 = np.zeros((samples_tot*samples_tot, t_for_pre.shape[0], output_nodes))  # 建立一个存放预测值的数组
y_samp3 = np.zeros((samples_tot*samples_tot, t_for_pre.shape[0], 6))

n = t_for_train.shape[0]
indexList = np.arange(n)

#打乱数据，并区分训练数据与测试数据
for jj in range(samples_tot):
    np.random.shuffle(indexList)
    trainT = []
    trainX = []
    trainY = []
    #OUT_x = []
    for j in range(n):#此处进行随机赋值
        N_int =int( N_intensify[j] )
        for i in range(N_int):
            trainT.append(t_for_train[indexList[j]])
            trainX.append(x_for_train[indexList[j]])
            trainY.append(y[indexList[j]])
            #OUT_x.append(x_out[indexList[j]])

    #数据类型转换
    trainT = np.array(trainT)
    trainX = np.array(trainX)
    trainY = np.array(trainY)

    trainTT = torch.Tensor(trainT).type(torch.FloatTensor)
    trainXX = torch.Tensor(trainX).type(torch.FloatTensor)
    trainYY = torch.Tensor(trainY).type(torch.FloatTensor)

    if cuda_is_available:
        trainTT=trainTT.cuda()
        trainXX=trainXX.cuda()
        trainYY=trainYY.cuda()

    Err1 = []  # 输入一个数组用来储存均方根误差的值用于输出图像

    net = MLP_BBB(input_nodes, hidden_nodes, Lorentz_units, output_nodes, prior_var=1.0)
    optimizer = optim.Adam(net.parameters(), lr=0.01)# 构造一个优化器对象Optimizer
    epochs = 5001
    for epoch in range(epochs):# loop over the dataset multiple times
#        if epoch % 10 == 0 :
#            print("now is epoch", epoch)
        optimizer.zero_grad()

        # forward + backward + optimize

        loss = net.sample_elbo(trainXX, trainTT, trainYY, 1)
        if cuda_is_available:
            loss = loss.cuda()
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(parameters=net.parameters(), max_norm=10,norm_type=2)
        optimizer.step()
        # samples is the number of "predictions" we make for 1 x-value.
        samples = 100
        x_tmp = trainXX #理论x轴
        y_samp = np.zeros((samples, trainYY.shape[0], trainYY.shape[1]))#建立一个存放预测值的数组
        y_samp2 = np.zeros((samples, trainTT.shape[0], 5))


        """
        对于每千（1000）次的迭代进行均方根误差（标准差）计算并输出随机采样序数和迭代次数
        """
        if epoch % 1000 == 0:
            #if epoch > 1990:
            for s in range(samples):
                y_tmp = net(trainXX, trainTT).detach().cpu().numpy()
                y_samp[s] = y_tmp[:,0:1] #预测值
                y_samp2[s] = y_tmp[:, 1:6]
                #a = pow(np.mean(y_samp, axis=0) - trainY, 2.).mean(axis=0)
            err = pow(pow(np.mean(y_samp, axis=0) - trainY, 2.).mean(axis=0), 0.5)# prediction error #求均方根误差
            pp = np.mean(y_samp, axis=0)#提取预测值（数组）
            para = np.mean(y_samp2, axis=0)#提取预测值（数组）
            print("",epoch+1,"th epoch,Root Mean Square Error is ", err[0])#,"，随机打乱数据进行了",jj+1,"次；")


        """
        对每十万（100000）次的迭代进行测试数据预测及储存测试数据
        """
        """
        if epoch % 100000 == 0:
            n = t_for_train.shape[0]
            indexList = np.arange(n)
            T_test = []
            X_test = []
            Y_test = []
            OUT_test = []
            for j in range(n):  # 此处进行随机赋值
                if((x_out[j,0] == 53 and x_out[j,1] == 127) or (x_out[j,0] == 65 and x_out[j,1] == 159)):
                    T_test.append(t_for_train[j])
                    X_test.append(x_for_train[j])
                    Y_test.append(y[j])
                    OUT_test.append(x_out[j])
            T_test = np.array(T_test)
            X_test = np.array(X_test)
            Y_test = np.array(Y_test)
            OUT_test = np.array(OUT_test)
            preTT_test = torch.Tensor(T_test).type(torch.FloatTensor).cuda()
            preXX_test = torch.Tensor(X_test).type(torch.FloatTensor).cuda()
            samples = 100
            y_samp1_test = np.zeros((samples, X_test.shape[0], output_nodes))  # 建立一个存放预测值的数组
            y_samp2_test = np.zeros((samples, X_test.shape[0], 6))
            for s in range(samples):
                y_tmp = net(preXX_test, preTT_test).detach().cpu().numpy()
                y_samp1_test[s] = y_tmp[:, 0:1]  # 预测值
                y_samp2_test[s] = y_tmp[:, 1:7]
            pp_test = np.mean(y_samp1_test, axis=0)  # 提取预测值（数组）
            pre_out_test = OUT_test
            pre_out_test = np.c_[pre_out_test, np.power(10, Y_test)[:, 0]]  # 能量
            pre_out_test = np.c_[pre_out_test, np.power(10, pp_test)[:, 0]]  # 能量
            # pre_out = np.c_[pre_out, y[:,0]]  # 能量
            # pre_out = np.c_[pre_out, pp[:,0]]   # 能量
            pp1_test = np.mean(y_samp2_test, axis=0)  # 提取预测值（数组）
            pre_out_test = np.c_[pre_out_test, pp1_test]  # 能量
            pre_out_test = pre_out_test.tolist()
            np.savetxt("pre_BNN_test.txt", pre_out_test, fmt="%f", delimiter="  ")
            #pp_up = np.percentile(y_samp1, 97.5, axis=0)  # *maxabsy
        """


    if(err[0] > 0.21 ):
        continue#排除劣质拟合结果


    #preTT = torch.Tensor(t_for_train).type(torch.FloatTensor)
    #preXX = torch.Tensor(x_for_train).type(torch.FloatTensor)
    preTT = torch.Tensor(t_for_pre).type(torch.FloatTensor).cuda()
    preXX = torch.Tensor(x_for_pre).type(torch.FloatTensor).cuda()

    for s in range(samples_tot):
        y_tmp = net(preXX, preTT).detach().cpu().numpy()
        y_samp1[n_have_run*samples_tot +s] = y_tmp[:,0:1]  # 预测值
        y_samp3[n_have_run*samples_tot +s] = y_tmp[:, 1:7]
    n_have_run = n_have_run + 1
    if(n_have_run >= samples_tot):
        break


pre_out = x_for_pre

pp = np.mean(y_samp1, axis=0)#提取预测值（数组）
#pre_out = np.c_[pre_out, np.power(10,y)[:, 0]]  # 能量
pre_out = np.c_[pre_out, np.power(10,pp)[:, 0]/1000]  # 能量

#pre_out = np.c_[pre_out, y[:,0]]  # 能量
#pre_out = np.c_[pre_out, pp[:,0]]   # 能量
pp1 = np.mean(y_samp3, axis=0)#提取预测值（数组）
pre_out = np.c_[pre_out, pp1]  # 能量

pp_up = np.percentile(y_samp1, 97.5, axis=0)  # *maxabsy
#pp_up = np.percentile(y_samp1, 84.13, axis=0)  # *maxabsy
pp = pp_up
#pre_out = np.c_[pre_out, np.power(10,y)[:, 0]]  # 能量
pre_out = np.c_[pre_out, np.power(10,pp)[:, 0]/1000]  # 能量
#pre_out = np.c_[pre_out, y[:,0]]  # 能量
#pre_out = np.c_[pre_out, pp[:,0]]   # 能量
pp2 = np.percentile(y_samp3, 97.5, axis=0)#提取预测值（数组）
#pp2 = np.percentile(y_samp3, 84.13, axis=0)#提取预测值（数组）
pre_out = np.c_[pre_out, pp2]  # 能量

pp_down = np.percentile(y_samp1, 2.5, axis=0)  # *maxabsy
#pp_down = np.percentile(y_samp1, 15.87, axis=0)  # *maxabsy
pp = pp_down
#pre_out = np.c_[pre_out, np.power(10,y)[:, 0]]  # 能量
pre_out = np.c_[pre_out, np.power(10,pp)[:, 0]/1000]  # 能量
#pre_out = np.c_[pre_out, y[:,0]]  # 能量
#pre_out = np.c_[pre_out, pp[:,0]]   # 能量
pp3 = np.percentile(y_samp3, 2.5, axis=0)#提取预测值（数组）
#pp3 = np.percentile(y_samp3, 15.87, axis=0)#提取预测值（数组）
pre_out = np.c_[pre_out, pp3]  # 能量
pre_out = pre_out.tolist()
np.savetxt("pre_BNN.txt",pre_out, fmt="%f", delimiter="  ")





####################################################################################################
