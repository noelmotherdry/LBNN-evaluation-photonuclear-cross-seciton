###### 这是寻找最大关联的


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import math
#from scipy.stats import norm
import matplotlib.pyplot as plt

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

    def forward(self, input):
        #这个函数并未被调用，如何理解？
        """
          Optimization process
        """
        # sample weights
        w_epsilon = Normal(0.5, 1).sample(self.w_mu.shape)
        self.w = self.w_mu + torch.log(1+torch.exp(self.w_rho)) * w_epsilon

        # sample bias
        b_epsilon = Normal(0.5, 1).sample(self.b_mu.shape)
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
        self.hidden = Linear_BBB(input_units, hidden_units, prior_var=prior_var)
        #self.hidden2 = Linear_BBB(hidden_units, hidden_units, prior_var=prior_var)
        # 由hidden_units个节点参数，映射到洛伦兹参数
        self.out = Linear_BBB(hidden_units,Lorentz_units, prior_var=prior_var)
        # 由洛伦兹参数，映射到output_units节点的输出数据
        #self.Lorentz = Linear_BBB(hidden_units, Lorentz_units, prior_var=prior_var)
        # 应该是在这里插入物理公式定义
        self.noise_tol = noise_tol # we will use the noise tolerance to calculate our likelihood

    def forward(self, x, t):
        # again, this is equivalent to a standard multilayer perceptron
        x = torch.sigmoid(self.hidden(x)) #hidden(x)：用x计算出节点参数；torch.sigmoid(t)则是做1/(1+exp(-t))计算
        #x = torch.sigmoid(self.hidden2(x))
        x = torch.sigmoid(self.out(x)) #再用做了做1/(1+exp(-t))计算的节点，计算出输出数据y
        #x = self.Lorentz(x)
        x = Lorentzian_spectrum(x, t)
        #应该是在这里插入物理公式
        return x

    def log_prior(self):
        # calculate the log prior over all the layers
        return self.hidden.log_prior + self.out.log_prior #+self.hidden2.log_prior

    def log_post(self):
        # calculate the log posterior over all the layers
        return self.hidden.log_post + self.out.log_post #+self.hidden2.log_post

    def sample_elbo(self, input, input_t, target, samples):
        # we calculate the negative elbo, which will be our loss function
        #initialize tensors
        # outputs 有三个维度，第一个维度是抽样数samples；第二维度是训练数据的个数；第三个维度中的0:1是能谱，1:7是洛伦兹参数
        outputs = torch.zeros(samples, target.shape[0], target.shape[1]+7)
        log_priors = torch.zeros(samples)
        log_posts = torch.zeros(samples)
        log_likes = torch.zeros(samples)
        # make predictions and calculate prior, posterior, and likelihood for a given number of samples
        for i in range(samples):
            outputs[i] = self(input, input_t) # make predictions
            log_priors[i] = self.log_prior() # get log prior
            log_posts[i] = self.log_post() # get log variational posterior
            #outputs[i,:,0:1]是能谱，outputs[i,:,1:7]是洛伦兹参数；抽样只抽能谱的，因为洛伦兹参数是没有训练数据的
            log_likes[i] = Normal(outputs[i,:,0:1], self.noise_tol).log_prob(target).sum() # calculate the log likelihood
        # calculate monte carlo estimate of prior posterior and likelihood
        log_prior = log_priors.mean()
        log_post = log_posts.mean()
        log_like = log_likes.mean()
        # calculate the negative elbo (which is our loss function)
        loss = log_post - log_prior - log_like
        return loss

def Lorentzian_spectrum(x, t):
    sig_TRK = torch.mul(t[:,1:2], t[:,2:3]-t[:,1:2] )
    sig_TRK = 60.*torch.div(sig_TRK, t[:, 2:3])

    E1 = 20*x[:,1:2]+10
    E2 = 20*x[:, 4:5]+10

    a1 = 1*(x[:,2:3])+0.1
    a2 = 1*(x[:,5:6])+0.1
    Gam1 = torch.mul(a1,t[:,0:1])
    Gam2 = torch.mul(a2,t[:,0:1])


    G1 = 2*(x[:,0:1])+0.1
    G2 = 2*(x[:,3:4])+0.1


    E1_2 = torch.square(E1)
    t_2 = torch.square(t[:,0:1])
    t_2_Gam = torch.mul(t_2, Gam1)
    t_Gam = torch.mul(t[:,0:1], Gam1)
    t_2_E_2 = torch.sub(t_2, E1_2)
    t_2_E_Gam = torch.add( torch.square(t_2_E_2),torch.square(t_Gam) )
    F1 = 2./3.14159265*torch.div(t_2_Gam, t_2_E_Gam)
    sig_GDR1 = torch.mul(sig_TRK, G1)
    sig_GDR1 = torch.mul(sig_GDR1, F1)

    E1_2 = torch.square(E2)
    t_2 = torch.square(t[:,0:1])
    t_2_Gam = torch.mul(t_2, Gam2)
    t_Gam = torch.mul(t[:,0:1], Gam2)
    t_2_E_2 = torch.sub(t_2, E1_2)
    t_2_E_Gam = torch.add( torch.square(t_2_E_2),torch.square(t_Gam) )
    F2 = 2./3.14159265*torch.div(t_2_Gam, t_2_E_Gam)
    sig_GDR2 = torch.mul(sig_TRK, G2)
    sig_GDR2 = 2*torch.mul(sig_GDR2, F2)

    y = torch.add(sig_GDR1, sig_GDR2)
    y = torch.add(y,t[:,4:5])
    y = torch.log10(y)
    #parameters of Lorentzian shape
    # 把能谱y与洛伦兹参数合并成一个张量yy，输出yy
    #yy = torch.cat((y,torch.exp(x[:,0:1]), torch.exp(x[:,1:2]), torch.abs(x[:,2:3]), torch.abs(x[:,3:4]),1./torch.exp(x[:,4:5]),1./torch.exp(x[:,5:6]) ), 1)
    Gam1 = torch.mul(a1,E1)
    Gam2 = torch.mul(a2,E2)
    yy = torch.cat((y, G1, G2, torch.abs(E1), torch.abs(E2),Gam1, Gam2,sig_TRK), 1)
    return yy


#检查洛伦兹函数的

"""
t = []
t = np.arange(5, 40.0, 0.01).reshape(-1, 1)
tt = torch.from_numpy(t).type(torch.FloatTensor)
n = t.shape[0]
x = []
g1 = np.log(100.)
for i in range(n):
    x.append([g1, 15., 5.0,g1, 17., 5.0])
x = np.array(x)
xx = torch.from_numpy(x).type(torch.FloatTensor)
y = Lorentzian_spectrum(xx, tt)
yy = np.exp(y[:,0:1])
plt.legend()
plt.scatter(tt, yy)
plt.title('Posterior Predictive')
plt.show()
"""

#读取原始数据  1, 18, 17,5,4
#0iZ,1iA,2iN,3delta
#4Baverage,5QB_,6Q2B_,7Q4B_,8Sn,9Sp,10S2n,11S2p,12Qa,13Qep,14QB_n,15Qda,16Qpa,17Qna,18beta2,19beta4,20beta6,21Esh,22Edef
#23Ein, 24errEin, 25sig, 26errsig
label = ['iZ', 'iA', 'N_Z', 'delta', 'Baverage', 'QB_', 'Q2B_','Q4B_', 'Sn', 'Sp', 'S2n', 'S2p',
         'Qa', 'Qep', 'QB_n','Qda','Qpa','Qna', 'beta2', 'beta4', 'beta6', 'Esh', 'Edef']
fileName = 'F:/20200918-LBNN/code/sign2n_EXFOR_with_mass.dat'
n = len(open(fileName).readlines())
t_for_train = []
y = []
yerr = []
x0to23 = []
fr = open(fileName)
for line in fr.readlines():
    curLine = line.strip().split()
    x0to23.append(np.asfarray(curLine[0:23]))
    t_for_train.append(np.asfarray(curLine[23:24]))
    y.append(np.asfarray(curLine[25:26]))
    yerr.append(np.asfarray(curLine[26:27]))
x0to23 = np.array(x0to23)
y = np.array(y)*1000.
yerr = np.array(yerr)*1000.
imax = yerr.shape[0]
N_intensify = np.zeros((imax, 1))
for i in range(imax):
    if(yerr[i]==0):
        N_int = 1
    else:
        N_int = math.floor(0.1 * y[i] / yerr[i]  +0.5) + 1
    N_intensify[i] = N_int


t_for_train = np.array(t_for_train)
t_for_train = np.c_[t_for_train, x0to23[:, 0]]
t_for_train = np.c_[t_for_train, x0to23[:, 1]]
t_for_train = np.c_[t_for_train, x0to23[:, 18]]
t = t_for_train
imax = t.shape[0]
sig_QD = np.zeros((imax, 1))
for i in range(imax):
    if t[i, 0:1] > 2.224:
      sig_QD[i,0:1] = 397.8 *t[i, 1:2]*(t[i, 2:3] - t[i, 1:2]) / t[i, 2:3]
      sig_QD[i,0:1] = sig_QD[i,0:1] *(t[i, 0:1] - 2.224)**1.5 /t[i, 0:1]**3
    if t[i, 0:1] <= 20:
        sig_QD[i, 0:1] = sig_QD[i, 0:1]*math.exp(-73.3 / t[i, 0:1])
    if t[i, 0:1] > 20:
        sig_QD[i, 0:1] = sig_QD[i, 0:1] * (
                    0.083714 - 0.0098343 * t[i, 0:1] + 4.1222 * (t[i, 0:1] * 0.01) ** 2 - 3.4762 * (
                        t[i, 0:1] * 0.01) ** 3 + 0.93537 * (t[i, 0:1] * 0.01) ** 4)
t_for_train = np.c_[t_for_train, sig_QD[:, 0]]


fileName = 'F:/20200918-LBNN/code/sign2n_EXFOR_with_mass_to_BNN.dat'
n_pre = len(open(fileName).readlines())
input_pre_x0to23 = []
t_for_pre = []
fr = open(fileName)
for line in fr.readlines():
    curLine = line.strip().split()
    input_pre_x0to23.append(np.asfarray(curLine[0:23]))
    t_for_pre.append(np.asfarray(curLine[23:24]))
input_pre_x0to23 = np.array(input_pre_x0to23)
t_for_pre = np.array(t_for_pre)
t_for_pre = np.c_[t_for_pre, input_pre_x0to23[:, 0]]
t_for_pre = np.c_[t_for_pre, input_pre_x0to23[:, 1]]
t_for_pre = np.c_[t_for_pre, input_pre_x0to23[:, 18]]
t = t_for_pre
imax = t.shape[0]
sig_QD = np.zeros((imax, 1))
for i in range(imax):
    if t[i, 0:1] > 2.224:
      sig_QD[i,0:1] = 397.8 *t[i, 1:2]*(t[i, 2:3] - t[i, 1:2]) / t[i, 2:3]
      sig_QD[i,0:1] = sig_QD[i,0:1] *(t[i, 0:1] - 2.224)**1.5 /t[i, 0:1]**3
    if t[i, 0:1] <= 20:
        sig_QD[i, 0:1] = sig_QD[i, 0:1]*math.exp(-73.3 / t[i, 0:1])
    if t[i, 0:1] > 20:
        sig_QD[i, 0:1] = sig_QD[i, 0:1] * (
                    0.083714 - 0.0098343 * t[i, 0:1] + 4.1222 * (t[i, 0:1] * 0.01) ** 2 - 3.4762 * (
                        t[i, 0:1] * 0.01) ** 3 + 0.93537 * (t[i, 0:1] * 0.01) ** 4)
t_for_pre = np.c_[t_for_pre, sig_QD[:, 0]]

#输出数准备，包括弹核Z与A，靶核Z与A，入射能量，产物的Z和A
x_out = x0to23[:, 0] #Z
x_out = np.c_[x_out, x0to23[:, 1]] #A
x_out = np.c_[x_out, t_for_train[:, 0]] #能量
#x_out = x_out.tolist()
x_out_pre = input_pre_x0to23[:, 0] #Z
x_out_pre = np.c_[x_out_pre, input_pre_x0to23[:, 1]] #A
x_out_pre = np.c_[x_out_pre, t_for_pre[:, 0]] #能量
#x_out_pre = x_out_pre.tolist()

#转换成N_Z
x0to23[:, 2] = x0to23[:, 2] -x0to23[:, 0]
input_pre_x0to23[:, 2] = input_pre_x0to23[:, 2] -input_pre_x0to23[:, 0]

y = np.log10(y)

#寻找最佳输入数据组合方法
# 用“for jj in range(23):”做循环，用“epochs = 1000”，“#if epoch > 990:”输出1000个迭代的误差。
# 修改 ”寻找最佳输入数据组合方法“ 框起来的部分，分别根据输出误差，选择1维，2维等输入维度，直到误差不再减小
for jj in range(23):
  input_nodes = 5
  hidden_nodes = 30
  Lorentz_units = 7
  output_nodes = 1
  if jj == 1:
      continue
  if jj == 2:
      continue
  if jj == 18:
      continue
#a = 1
#if a == 1:
samples = 100
y_samp1 = np.zeros((samples*10, t_for_pre.shape[0], output_nodes))  # 建立一个存放预测值的数组
y_samp3 = np.zeros((samples*10, t_for_pre.shape[0], 6))
for jj in range(10):
  #筛选input自由度   1, 18, 17,5,4
  # 筛选input自由度   1, 18, 8,5,4
  x_for_train = x0to23[:, 1]
  x_for_train = np.c_[x_for_train, x0to23[:, 18]]
  x_for_train = np.c_[x_for_train, x0to23[:, 8]]
  x_for_train = np.c_[x_for_train, x0to23[:, 5]]
  #x_for_train = np.c_[x_for_train, x0to23[:, 24]]
  #x_for_train = np.c_[x_for_train, x0to23[:, 44]]
  #x_for_train = np.c_[x_for_train, x0to23[:, 45]]
  x_for_train = np.c_[x_for_train, x0to23[:,4]]

  x_for_pre = input_pre_x0to23[:, 1]
  x_for_pre = np.c_[x_for_pre, input_pre_x0to23[:, 18]]
  x_for_pre = np.c_[x_for_pre, input_pre_x0to23[:, 8]]
  x_for_pre = np.c_[x_for_pre, input_pre_x0to23[:, 5]]
  #x_for_pre = np.c_[x_for_pre, input_pre_x0to23[:, 24]]
  #x_for_pre = np.c_[x_for_pre, input_pre_x0to23[:, 44]]
  #x_for_pre = np.c_[x_for_pre, input_pre_x0to23[:, 45]]
  x_for_pre = np.c_[x_for_pre, input_pre_x0to23[:, 4]]

  # 数据缩放
  x_for_train = x_for_train.tolist()  # 将x从数组转成list
  cenx = np.std(x_for_train, axis=0)
  avex = np.mean(x_for_train, axis=0)
  x_for_train = (np.array(x_for_train) - avex) / cenx
  x_for_pre = (np.array(x_for_pre) - avex) / cenx

  cent = np.std(t_for_train, axis=0)
  avet = np.mean(t_for_train, axis=0)
  t_for_train = np.array(t_for_train)
  t_for_pre = np.array(t_for_pre)
  #t_for_train = (np.array(t_for_train) - avet) / cent
  #t_for_pre = (np.array(t_for_pre) - avet) / cent

  #ceny = np.std(y, axis=0)
  #avey = np.mean(y, axis=0)
  # y = (np.array(y) - avey) / ceny
  y = np.array(y)

  #打乱数据，并区分训练数据与测试数据
  n = t_for_train.shape[0]
  indexList = np.arange(n)
  np.random.shuffle(indexList)
  trainT = []
  trainX = []
  trainY = []
  OUT_x = []
  for j in range(n):#此处进行随机赋值
    N_int =int( N_intensify[j] )
    for i in range(N_int):
      trainT.append(t_for_train[indexList[j]])
      trainX.append(x_for_train[indexList[j]])
      trainY.append(y[indexList[j]])
      OUT_x.append(x_out[indexList[j]])

  #数据类型转换
  trainT = np.array(trainT)
  trainX = np.array(trainX)
  trainY = np.array(trainY)

  trainTT = torch.Tensor(trainT).type(torch.FloatTensor)
  trainXX = torch.Tensor(trainX).type(torch.FloatTensor)
  trainYY = torch.Tensor(trainY).type(torch.FloatTensor)

  Err1 = []  # 输入一个数组用来储存均方根误差的值用于输出图像
  net = MLP_BBB(input_nodes, hidden_nodes, Lorentz_units, output_nodes, prior_var=1.0)
  optimizer = optim.Adam(net.parameters(), lr=0.1)  # 构造一个优化器对象Optimizer
  epochs = 15001
  for epoch in range(epochs):  # loop over the dataset multiple times
    optimizer.zero_grad()
    # forward + backward + optimize
    loss = net.sample_elbo(trainXX, trainTT, trainYY, 1)
    loss.backward()
    optimizer.step()
    # samples is the number of "predictions" we make for 1 x-value.
    samples = 1
    x_tmp = trainXX #理论x轴
    y_samp = np.zeros((samples, trainYY.shape[0], trainYY.shape[1]))#建立一个存放预测值的数组
    y_samp2 = np.zeros((samples, trainTT.shape[0], 7))
    if epoch % 100 == 0:
    #if epoch > 1990:
      for s in range(samples):
        y_tmp = net(trainXX, trainTT).detach().numpy()
        y_samp[s] = y_tmp[:,0:1] #预测值
        y_samp2[s] = y_tmp[:, 1:8]
        #a = pow(np.mean(y_samp, axis=0) - trainY, 2.).mean(axis=0)
      err = pow(pow(np.mean(y_samp, axis=0) - trainY, 2.).mean(axis=0), 0.5)# prediction error #求均方根误差
      pp = np.mean(y_samp, axis=0)#提取预测值（数组）
      para = np.mean(y_samp2, axis=0)#提取预测值（数组）
      print(epoch, err[0],jj,label[jj])

    if epoch % 1000 == 0:
      n = t_for_train.shape[0]
      indexList = np.arange(n)
      T_test = []
      X_test = []
      Y_test = []
      OUT_test = []
      for j in range(n):  # 此处进行随机赋值
          if((x_out[j,0] == 53 and x_out[j,1] == 127) or (x_out[j,0] == 65 and x_out[j,1] == 159) or (x_out[j,0] == 50 and x_out[j,1] == 120)):
              T_test.append(t_for_train[j])
              X_test.append(x_for_train[j])
              Y_test.append(y[j])
              OUT_test.append(x_out[j])
      T_test = np.array(T_test)
      X_test = np.array(X_test)
      Y_test = np.array(Y_test)
      OUT_test = np.array(OUT_test)
      preTT_test = torch.Tensor(T_test).type(torch.FloatTensor)
      preXX_test = torch.Tensor(X_test).type(torch.FloatTensor)
      samples = 100
      y_samp1_test = np.zeros((samples, X_test.shape[0], output_nodes))  # 建立一个存放预测值的数组
      y_samp2_test = np.zeros((samples, X_test.shape[0], 6))
      for s in range(samples):
          y_tmp = net(preXX_test, preTT_test).detach().numpy()
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
      np.savetxt("F:/20200918-LBNN/code/pre_BNN_test.txt", pre_out_test, fmt="%f", delimiter="  ")
      pp_up = np.percentile(y_samp1, 97.5, axis=0)  # *maxabsy


  samples = 100
  #preTT = torch.Tensor(t_for_train).type(torch.FloatTensor)
  #preXX = torch.Tensor(x_for_train).type(torch.FloatTensor)
  preTT = torch.Tensor(t_for_pre).type(torch.FloatTensor)
  preXX = torch.Tensor(x_for_pre).type(torch.FloatTensor)
  pre_out = x_out_pre
  for s in range(samples):
    y_tmp = net(preXX, preTT).detach().numpy()
    y_samp1[jj*samples +s] = y_tmp[:,0:1]  # 预测值
    y_samp3[jj*samples +s] = y_tmp[:, 1:7]
pp = np.mean(y_samp1, axis=0)#提取预测值（数组）
#pre_out = np.c_[pre_out, np.power(10,y)[:, 0]]  # 能量
pre_out = np.c_[pre_out, np.power(10,pp)[:, 0]/1000]  # 能量
#pre_out = np.c_[pre_out, y[:,0]]  # 能量
#pre_out = np.c_[pre_out, pp[:,0]]   # 能量
pp1 = np.mean(y_samp3, axis=0)#提取预测值（数组）
pre_out = np.c_[pre_out, pp1]  # 能量

pp_up = np.percentile(y_samp1, 97.5, axis=0)  # *maxabsy
pp = pp_up
#pre_out = np.c_[pre_out, np.power(10,y)[:, 0]]  # 能量
pre_out = np.c_[pre_out, np.power(10,pp)[:, 0]/1000]  # 能量
#pre_out = np.c_[pre_out, y[:,0]]  # 能量
#pre_out = np.c_[pre_out, pp[:,0]]   # 能量
pp2 = np.percentile(y_samp3, 97.5, axis=0)#提取预测值（数组）
pre_out = np.c_[pre_out, pp2]  # 能量

pp_down = np.percentile(y_samp1, 2.5, axis=0)  # *maxabsy
pp = pp_down
#pre_out = np.c_[pre_out, np.power(10,y)[:, 0]]  # 能量
pre_out = np.c_[pre_out, np.power(10,pp)[:, 0]/1000]  # 能量
#pre_out = np.c_[pre_out, y[:,0]]  # 能量
#pre_out = np.c_[pre_out, pp[:,0]]   # 能量
pp3 = np.percentile(y_samp3, 2.5, axis=0)#提取预测值（数组）
pre_out = np.c_[pre_out, pp3]  # 能量
pre_out = pre_out.tolist()
np.savetxt("F:/20200918-LBNN/code/pre_BNN.txt",pre_out, fmt="%f", delimiter="  ")





####################################################################################################
