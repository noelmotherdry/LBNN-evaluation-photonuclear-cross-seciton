

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
    def __init__(self, input_units, hidden_units  , output_units, noise_tol=0.1,  prior_var=1.):

        # initialize the network like you would with a standard multilayer perceptron, but using the BBB layer
        super().__init__()
        #产生一个可以优化的参数矩阵，大小为input_units * hidden_units
        #定义了这个参数矩阵，就相当于定义了一个映射，由input_units维度的输入数据，映射到hidden_units个节点参数
        self.hidden = Linear_BBB(input_units, hidden_units, prior_var=prior_var)#i*h矩阵，i节点h隐藏参数
        #self.hidden2 = Linear_BBB(hidden_units, hidden_units, prior_var=prior_var)
        # 由hidden_units个节点参数，映射到洛伦兹参数
        #self.hidden2 = Linear_BBB(hidden_units, hidden2_units, prior_var=prior_var)

        self.out = Linear_BBB(hidden_units,output_units, prior_var=prior_var)#h*L矩阵，h隐藏参数L洛伦兹参数
        # 由洛伦兹参数，映射到output_units节点的输出数据
        #self.Lorentz = Linear_BBB(hidden_units, Lorentz_units, prior_var=prior_var)
        # 应该是在这里插入物理公式定义
        self.noise_tol = noise_tol # we will use the noise tolerance to calculate our likelihood
        if cuda_is_available:
            self = self.cuda()

    def forward(self, x):
        # again, this is equivalent to a standard multilayer perceptron

        x = torch.sigmoid(self.hidden(x)) #hidden(x)：用x计算出节点参数；torch.sigmoid(t)则是做1/(1+exp(-t))计算
        #x = torch.sigmoid(self.hidden2(x))
        #x= torch.sigmoid(self.hidden2(x))

        x = torch.sigmoid(self.out(x)) #再用做了做1/(1+exp(-t))计算的节点，计算出输出数据y

        #应该是在这里插入物理公式
        return x

    def log_prior(self):
        # calculate the log prior over all the layers
        #self.out.
        return self.hidden.log_prior + self.out.log_prior

    def log_post(self):
        # calculate the log posterior over all the layers
        return self.hidden.log_post + self.out.log_post #+self.hidden2.log_post

    def sample_elbo(self, input, target, samples):
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
            outputs[i] = self(input) # make predictions

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

    def evaluate(self,y_samp):
        for s in range(samples):
            y_samp[s] = self(trainXX).detach().cpu().numpy()
            # a = pow(np.mean(y_samp, axis=0) - trainY, 2.).mean(axis=0)
        err = pow(pow(np.mean(y_samp, axis=0) - trainY, 2.).mean(axis=0), 0.5)  # prediction error #求均方根误差
        pp = np.mean(y_samp, axis=0)  # 提取预测值（数组）
        print("第", epoch + 1, "次迭代，均方根误差为", err[0], "，损失为", loss.item())  # ,"，随机打乱数据进行了",jj+1,"次；")
        return



def shuffle_data(x, y, N_intensify):
  """Randomly shuffles the data."""
  trainX = []
  trainY = []
  for j in range(n):  # 此处进行随机赋值
      N_int = int(N_intensify[j])
      for i in range(N_int):
          trainX.append(x[indexList[j]])
          trainY.append(y[indexList[j]])
  trainX = np.array(trainX)
  trainY = np.array(trainY)
  return trainX, trainY


def tensor_creator(data):
    trainXX = torch.Tensor(data).type(torch.FloatTensor)
    if cuda_is_available:
        trainXX=trainXX.cuda()
    return trainXX



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
    res =  np.transpose(np.array(res))
    x_for_train = res[:, 0:3]  # Z\A
    x_for_train = np.c_[x_for_train, res[:, 6]]
    x_for_train = np.c_[x_for_train, res[:, 7]]
    x_for_train = np.c_[x_for_train, res[:, 8]]
    x_for_train = np.c_[x_for_train, res[:, 9]]
    x_for_train = np.c_[x_for_train, res[:, 10]]
    # dealing with y and yerr's weight
    y = np.array(res[:, 4:5])
    yerr = np.array(res[:, 5:6])
    imax = yerr.shape[0]
    N_intensify = np.zeros((imax, 1))
    for i in range(imax):
        if (yerr[i] == 0):
            N_int = 1
        else:
            N_int = math.floor(0.1 * y[i] / yerr[i] + 0.5) + 1
        N_intensify[i] = N_int

    #print(x_for_train.shape)
    return x_for_train,y,N_intensify



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

x_for_train , y , N_intensify = read_train_data_file(path+"sign2n_EXFOR_with_mass.dat",path+"nulide_info.dat")


x_for_pre = read_predict_data_file(path+"sign2n_EXFOR_with_mass_to_BNN.dat",path+"nulide_info.dat")


x_for_pre = np.array(x_for_pre)


"""
cenx = np.std(x_for_train, axis=0)
avex = np.mean(x_for_train, axis=0)
x_for_train = (np.array(x_for_train) - avex) / cenx
x_for_pre = (np.array(x_for_pre) - avex) / cenx
"""

input_nodes = 8
hidden_nodes = 30
output_nodes = 1
samples_tot = 5
n_have_run = 0



y_samp1 = np.zeros((samples_tot*samples_tot, x_for_pre.shape[0], output_nodes))  # 建立一个存放预测值的数组

n = x_for_train.shape[0]
indexList = np.arange(n)

#打乱数据，并区分训练数据与测试数据
for jj in range(samples_tot):

    trainX , trainY = shuffle_data(x_for_train, y, N_intensify)
    trainXX = tensor_creator(trainX)
    trainYY = tensor_creator(trainY)

    Err1 = []  # 输入一个数组用来储存均方根误差的值用于输出图像

    net = MLP_BBB(input_nodes, hidden_nodes,output_nodes, prior_var=1.0)
    optimizer = optim.Adam(net.parameters(), lr=0.1)# 构造一个优化器对象Optimizer
    epochs = 3001
    for epoch in range(epochs):# loop over the dataset multiple times
#        if epoch % 10 == 0 :
#            print("now is epoch", epoch)

        # forward + backward + optimize
        optimizer.zero_grad()
        loss = net.sample_elbo(trainXX, trainYY, 1)
        if cuda_is_available:
            loss = loss.cuda()
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(parameters=net.parameters(), max_norm=10,norm_type=2)
        optimizer.step()
        # samples is the number of "predictions" we make for 1 x-value.
        samples = 100
        x_tmp = trainXX #理论x轴
        """
        对于每千（1000）次的迭代进行均方根误差（标准差）计算并输出随机采样序数和迭代次数
        """
        if epoch % 1000 == 0:
            net.evaluate(np.zeros((samples, trainYY.shape[0], trainYY.shape[1])))


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
        """

preX = x_for_pre

preXX = tensor_creator(preX)

samples = 100
#x_tmp = torch.linspace(-5,5,100).reshape(-1,1)
y_samp = np.zeros((samples,preXX.shape[0]))
for s in range(samples):
    y_tmp = net(preXX).detach().numpy()
    y_samp[s] = y_tmp.reshape(-1)

x1=[]
x2=[]
x3=[]
x4=[]
xp1=[]
xp2=[]
xp3=[]
xp4=[]
y1=[]
y2=[]
y3=[]
y4=[]
yp1=[]
yp2=[]
yp3=[]
yp4=[]
for  i in range(trainXX.shape[0]):
    if trainXX[i,3] == 1:
        x1.append(trainXX[i,2:3].numpy())
        y1.append(trainYY[i].numpy())
    elif trainXX[i,3] ==2:
        x2.append(trainXX[i,2:3].numpy())
        y2.append(trainYY[i].numpy())
    elif trainXX[i,3] ==3:
        x3.append(trainXX[i,2:3].numpy())
        y3.append(trainYY[i].numpy())
    elif trainXX[i,3] ==4:
        x4.append(trainXX[i,2:3].numpy())
        y4.append(trainYY[i].numpy())
preY=np.mean(y_samp, axis = 0)
for  i in range(preXX.shape[0]):
    if preXX[i,3] == 1:
        xp1.append(preXX[i,2:3].numpy())
        yp1.append(preY[i])
    elif preXX[i,3] ==2:
        xp2.append(preXX[i,2:3].numpy())
        yp2.append(preY[i])
    elif preXX[i,3] ==3:
        xp3.append(preXX[i,2:3].numpy())
        yp3.append(preY[i])
    elif preXX[i,3] ==4:
        xp4.append(preXX[i,2:3].numpy())
        yp4.append(preY[i])
plt.scatter(x1, y1, label='(γ,n)',color='#CCFFFF')
plt.scatter(x2,y2,label='(γ,2n)',color='#00CCFF')
plt.scatter(x3,y3,label='(γ,abs)',color='#0000FF')
plt.scatter(x4,y4,label='(γ,xn)',color='#000080')
plt.scatter(xp1, yp1, label='predict(γ,n)',color='#FFFF99')
plt.scatter(xp2,yp2,label='predict(γ,2n)',color='#FFCC00')
plt.scatter(xp3,yp3,label='predict(γ,abs)',color='#FF6600')
plt.scatter(xp4,yp4,label='predict(γ,xn)',color='#FF0000')
#plt.fill_between(trainTT[:,2:3].numpy().reshape(-1), np.percentile(y_samp, 2.5, axis = 0), np.percentile(y_samp, 97.5, axis = 0), alpha = 0.25, label='95% Confidence')
plt.legend()
plt.xlim(0,30)
plt.ylim(0,1450)
plt.xlabel("Incident γ energy(MeV)")
plt.ylabel("Cross section(mb)")
plt.title('P5884')
plt.show()

