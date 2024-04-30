import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import math
import matplotlib.pyplot as plt
resss=[]
cuda_is_available = False#torch.cuda.is_available()
#from scipy.stats import norm


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
        self.w_mu = nn.Parameter(torch.zeros(output_features, input_features))
        self.w_rho = nn.Parameter(torch.zeros(output_features, input_features))

        #initialize mu and rho parameters for the layer's bias
        self.b_mu =  nn.Parameter(torch.zeros(output_features))
        self.b_rho = nn.Parameter(torch.zeros(output_features))

        #initialize weight samples (these will be calculated whenever the layer makes a prediction)
        self.w = None
        self.b = None

        # initialize prior distribution for all of the weights and biases
        self.prior = torch.distributions.Normal(0,prior_var)



    def forward(self, input):
        """
          Optimization process
        """
        # sample weights
        w_epsilon = Normal(0,1).sample(self.w_mu.shape)
        self.w = self.w_mu + torch.log(1+torch.exp(self.w_rho)) * w_epsilon

        # sample bias
        b_epsilon = Normal(0,1).sample(self.b_mu.shape)
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
    def __init__(self, input_units , hidden_units,hidden_units2, noise_tol=.1,  prior_var=1.):

        # initialize the network like you would with a standard multilayer perceptron, but using the BBB layer
        super().__init__()

        self.hidden = Linear_BBB(input_units ,hidden_units, prior_var=prior_var)
        self.hidden2 = Linear_BBB(hidden_units,hidden_units2,prior_var=prior_var)
        self.out = Linear_BBB(hidden_units2, 2, prior_var=prior_var)

        self.noise_tol = noise_tol # we will use the noise tolerance to calculate our likelihood
    def forward(self, x , t ):
        # again, this is equivalent to a standard multilayer perceptron
        x=self.hidden(x)
        x = torch.sigmoid(x)
        x = self.out(x)
        x = Rfunc(x,t)
        return x

    def log_prior(self):
        # calculate the log prior over all the layers
        return self.hidden.log_prior + self.out.log_prior

    def log_post(self):
        # calculate the log posterior over all the layers
        return self.hidden.log_post + self.out.log_post

    def sample_elbo(self, input_x , target_y, t , samples):
        # we calculate the negative elbo, which will be our loss function
        #initialize tensors
        outputs = torch.zeros(samples, target_y.shape[0])
        log_priors = torch.zeros(samples)
        log_posts = torch.zeros(samples)
        log_likes = torch.zeros(samples)
        # make predictions and calculate prior, posterior, and likelihood for a given number of samples
        for i in range(samples):
            outputs[i] = self(input_x,t).reshape(-1) # make predictions
            log_priors[i] = self.log_prior() # get log prior
            log_posts[i] = self.log_post() # get log variational posterior
            log_likes[i] = Normal(outputs[i], self.noise_tol).log_prob(target_y.reshape(-1)).sum() # calculate the log likelihood
        # calculate monte carlo estimate of prior posterior and likelihood
        log_prior = log_priors.mean()
        log_post = log_posts.mean()
        log_like = log_likes.mean()
        # calculate the negative elbo (which is our loss function)
        loss = log_post - log_prior - log_like
        return loss
"""
    def evaluate(self,y_samp):
        for s in range(samples):
            y_samp[s] = self(trainXX).detach().cpu().numpy()
            # a = pow(np.mean(y_samp, axis=0) - trainY, 2.).mean(axis=0)
        err = pow(pow(np.mean(y_samp, axis=0) - trainY, 2.).mean(axis=0), 0.5)  # prediction error #求均方根误差
        pp = np.mean(y_samp, axis=0)  # 提取预测值（数组）
        print("第", epoch + 1, "次迭代，均方根误差为", err[0], "，损失为", loss.item())  # ,"，随机打乱数据进行了",jj+1,"次；")
        return
"""

def Rfunc(x,reaction_channel):
    R=x[:,1:2]
    f = torch.zeros((reaction_channel.shape[0],1))
    if cuda_is_available:
        f = f.cuda()
    for  i in range(reaction_channel.shape[0]):

        if  reaction_channel[i] == 1:#represent reaction chanenl (g,n)
            f[i]=R[i]
        elif reaction_channel[i] == 2:#represent reaction chanenl (g,2n)
            f[i]=(1-R[i])
        elif reaction_channel[i] == 3:
            f[i]=1
        elif reaction_channel[i] == 4:#represent reaction chanenl (g,xn)
            f[i]=(2-R[i])
    sig=x[:,0:1]
    sig=torch.mul(sig,f)
    return sig


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
    #输出结果每列分别是Z，A，e，e_err，sig，sig_err，reaction_channel，delta，B，Sn，S2n
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
                exit("Process terminated")
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
    t =  res[:, 6]
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
    return x_for_train,y,N_intensify,t


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
    res = np.transpose(np.array(res))
    t=res[:,3:4]
    return res,t

path = ""

x_for_train , y , N_intensify , reaction_channel = read_train_data_file(path+"sign2n_EXFOR_with_mass.dat",path+"nulide_info.dat")
x_for_pre , reaction_channel_pre = read_predict_data_file(path+"sign2n_EXFOR_with_mass_to_BNN.dat",path+"nulide_info.dat")


input_nodes = 8
hidden_nodes = 20
hidden_nodes2 = 20

trainX = x_for_train
trainY = y


trainXX = tensor_creator(trainX)
trainYY = tensor_creator(trainY)
trainTT = tensor_creator(reaction_channel)
trainTT = tensor_creator(reaction_channel)
net = MLP_BBB(input_nodes,hidden_nodes,hidden_nodes2, prior_var=10)
optimizer = optim.Adam(net.parameters(), lr=.1)
epochs = 3001
for epoch in range(epochs):  # loop over the dataset multiple times
    optimizer.zero_grad()
    # forward + backward + optimize
    loss = net.sample_elbo(trainXX , trainYY,trainTT, 1)
    loss.backward()
    optimizer.step()
    if epoch % 1000 == 0:
        y_samp = net(trainXX,trainTT).detach().cpu().numpy()
        err = pow(pow(np.mean(y_samp, axis=0) - y, 2.).mean(axis=0), 0.5)  # prediction error #求均方根误差
        pp = np.mean(y_samp, axis=0)  # 提取预测值（数组）
        print("第", epoch + 1, "次迭代，均方根误差为", err[0], "，损失为", loss.item())  # ,"，随机打乱数据进行了",jj+1,"次；")

print('Finished Training')




preX = x_for_pre


preXX = tensor_creator(preX)
preTT = tensor_creator(reaction_channel_pre)
samples = 100
#x_tmp = torch.linspace(-5,5,100).reshape(-1,1)
y_samp = np.zeros((samples,preXX.shape[0]))
for s in range(samples):
    y_tmp = net(preXX,preTT).detach().numpy()
    y_samp[s] = y_tmp.reshape(-1)
    """
plt.scatter(trainXX[:,2:3].numpy(), trainYY.numpy(), label='Mean Posterior Predictive',color='green')
#plt.fill_between(trainTT[:,2:3].numpy().reshape(-1), np.percentile(y_samp, 2.5, axis = 0), np.percentile(y_samp, 97.5, axis = 0), alpha = 0.25, label='95% Confidence')
plt.legend()
plt.plot(trainXX[:,2:3].numpy().reshape(-1), np.mean(y_samp, axis = 0), alpha = 0.25, label='95% Confidence')
plt.title('P5884')
plt.show()
"""
#print("etexd")
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
for  i in range(trainTT.shape[0]):
    if trainTT[i] == 1:
        x1.append(trainXX[i,2:3].numpy())
        y1.append(trainYY[i].numpy())
    elif trainTT[i] ==2:
        x2.append(trainXX[i,2:3].numpy())
        y2.append(trainYY[i].numpy())
    elif trainTT[i] ==3:
        x3.append(trainXX[i,2:3].numpy())
        y3.append(trainYY[i].numpy())
    elif trainTT[i] ==4:
        x4.append(trainXX[i,2:3].numpy())
        y4.append(trainYY[i].numpy())
preY=np.mean(y_samp, axis = 0)
for  i in range(preTT.shape[0]):
    if preTT[i] == 1:
        xp1.append(preXX[i,2:3].numpy())
        yp1.append(preY[i])
    elif preTT[i] ==2:
        xp2.append(preXX[i,2:3].numpy())
        yp2.append(preY[i])
    elif preTT[i] ==3:
        xp3.append(preXX[i,2:3].numpy())
        yp3.append(preY[i])
    elif preTT[i] ==4:
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
plt.legend()
plt.xlim(0,30)
plt.ylim(0,1450)
plt.xlabel("Incident γ energy(MeV)")
plt.ylabel("Cross section(mb)")
plt.show()
# samples is the number of "predictions" we make for 1 x-value.
