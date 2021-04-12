import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# You can write code above the if-main block.
def add_col_name(df):
  #新增欄位名稱，將第一列資料歸位(本來在欄位名稱)
  bad_col_name = list(df.columns)
  name_map = {bad_col_name[i]:new_name for i,new_name in enumerate(list('開高低收'))}#{ ... : 開高收低}
  df.rename(columns=name_map,inplace=True)
  new_dataframe = pd.DataFrame([bad_col_name],columns=list('開高收低'))
  new_dataframe = new_dataframe.append(df,ignore_index=True)
  # print(new_dataframe)
  return new_dataframe

class simpleLSTM(nn.Module):
  def __init__(self, input_size=1, hidden_size=128, num_layers=1, num_classes=1):
    super(simpleLSTM, self).__init__()
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
    self.fc2 = nn.Linear(hidden_size, int(hidden_size/4))
    self.fc1 = nn.Linear(int(hidden_size/4), num_classes)
    self.relu = nn.ReLU()
  def forward(self, x):
    # x shape (batch, time_step, input_size)
    # out shape (batch, time_step, output_size)
    # h_n shape (n_layers, batch, hidden_size)
    # h_c shape (n_layers, batch, hidden_size)
    # 初始化hidden和memory cell参数
    h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
    c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

    # forward propagate lstm
    out, (h_n, h_c) = self.lstm(x, (h0, c0))

    # 選取最后一个时刻的输出
    out = self.fc2(out[:, -1, :])
    out = self.relu(out)
    out = self.fc1(out)
    return out


class StockData(Dataset):
  def __init__(self,path,num_day):
    df = pd.read_csv(path,encoding='utf8')
    df = self.add_col_name(df)
    self.days = num_day#要往前看幾天
    self.open = df['開'].to_list()
    self.high = df['高'].to_list()
    self.close = df['收'].to_list()
    self.low = df['低'].to_list()
    self.open_dif = [float(self.open[i]) for i in range(0,len(self.open))]
    
  def add_col_name(self,df):
    #新增欄位名稱，將第一列資料歸位(本來在欄位名稱)
    bad_col_name = list(df.columns)
    name_map = {bad_col_name[i]:new_name for i,new_name in enumerate(list('開高低收'))}#{ ... : 開高收低}
    df.rename(columns=name_map,inplace=True)
    new_dataframe = pd.DataFrame([bad_col_name],columns=list('開高收低'))
    new_dataframe = new_dataframe.append(df,ignore_index=True)
    # print(new_dataframe)
    return new_dataframe

  def __getitem__(self,i):
    #LSTM data
    data = self.open_dif[i:i+self.days]
    label = self.open_dif[i+self.days]

    data = torch.tensor(data).float()
    label = torch.tensor(label).float()
    data = data.unsqueeze(1)
    # label = label.unsqueeze(1)
    # print(data.shape)
    return data,label
  def __len__(self):
    return len(self.open_dif)-self.days


def load_data(file_name):
    backward_day = 200
    return StockData(file_name,backward_day)

def train(model,train_loader,test_loader):
    criterion = nn.MSELoss(reduction='mean')
    optimizer = optim.Adam(model.parameters(),lr=0.01,weight_decay=0.001)
    scheduler = CosineAnnealingWarmRestarts(optimizer,T_0=5,T_mult=2)
    if(torch.cuda.is_available()):
        criterion = criterion.cuda()

    
    max_loss = 10**10
    for epochs in range(2000):
        running_loss = 0
        count = 0
        if (epochs==1500):
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr']*0.1
            model.train()
        #training
        for i in train_loader:
            
            data,label = i

            if(torch.cuda.is_available()):
                data = data.cuda()
                label = label.cuda()
            optimizer.zero_grad()
            output = model(data)
            # print(torch.ones(output.shape).shape)
            loss = criterion(output,label)
            
            # loss.backward(torch.ones(label.shape).cuda())
            loss.backward()
            optimizer.step()
            running_loss += loss.mean()
            count+=1
        scheduler.step()
        
        print('epoch:',epochs,'loss:',(running_loss/count).tolist())

        #save best
        pred_model = []
        label_model = []
        model.eval()

        for i in test_loader:
        
            data,label = i
            if(torch.cuda.is_available()):
                data = data.cuda()
            pred = model(data)
            # print(pred)
            pred_model += pred.detach().cpu().numpy().squeeze().tolist()
            label_model += label.detach().cpu().numpy().squeeze().tolist()
            test_loss = nn.MSELoss(reduction='mean')
            loss = test_loss(torch.tensor(np.array(pred_model).astype(np.float)),torch.tensor(np.array(label_model).astype(np.float)))
            if (max_loss>loss):
                model_name = 'best_model'
                torch.save(model,model_name)
                max_loss = loss
                print('save!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! val_loss:',loss.tolist(),end='\n')

def predict_future(train_input,real_set):
  #input:
    #train_input: np.array: [num_backward_days] 1dim
    #real_set: np.array: [num_test_day] 1dim
  #output:
    #result_list: np.array: [num_test_day-1,2] 2dim 每次預測後兩天，最後一天不用預測
  #從train_set取資料預測test(real)的第一筆後，才可將第一筆真實的test(real)資料加入input讓模型預測第二筆
  train_input = torch.tensor(train_input).unsqueeze(1)
  real_set = torch.tensor(real_set)
  result_list = []
  pred_days = len(real_set)-1#總天數-1
  each_step_day = 2#每次往後預測幾天
  for i in range(pred_days):
    input_temp = train_input
    result_pair_temp = []
    for j in range(each_step_day):#一次兩天
      data = input_temp.unsqueeze(0)
      if(torch.cuda.is_available()):
        data = data.cuda()
      pred = model(data)
      input_temp = torch.cat((input_temp[-199:],pred.detach().cpu()),0)#將預測後的一天加入data
      result_pair_temp.append(pred.detach().cpu().squeeze().tolist())

    result_list.append(result_pair_temp)
    # print(real_set[i])
    train_input = torch.cat((train_input,real_set[i].unsqueeze(0).unsqueeze(0)),0)
    # print(pre_input)
  result_list = np.array(result_list)
  return result_list

if __name__ == '__main__':
    # You should not modify this part.
    import argparse
    import os 
    print('current path:',os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('--training',
                       default='training_data.csv',
                       help='input training data file name')
    parser.add_argument('--testing',
                        default='testing_data.csv',
                        help='input testing data file name')
    parser.add_argument('--output',
                        default='output.csv',
                        help='output file name')
    args = parser.parse_args()
    
    # The following part is an example.
    # You can modify it at will.

    #data
    dataset = load_data(args.training)

    train_size = int(len(dataset)*0.7)

    training_data = torch.utils.data.Subset(dataset,range(0,train_size))
    val_data = torch.utils.data.Subset(dataset,range(train_size,len(dataset)))

    train_loader = DataLoader(training_data,batch_size=128,shuffle=False)
    val_loader = DataLoader(val_data,batch_size=128,shuffle=False)

    print('train size ',len(training_data))
    print('val size:',len(val_data))

    #train
    model = simpleLSTM()
    if(torch.cuda.is_available()):
        model = model.cuda()
    # train(model,train_loader,test_loader)
    
    #predict
    testing_data = pd.read_csv(args.testing)
    testing_data = add_col_name(testing_data)
    testing_data = testing_data['開'].to_list()
    testing_data = [float(testing_data[i]) for i in range(1,len(testing_data))]#str to float

    model_name = 'very_best_model'
    if(torch.cuda.is_available()):
      model = torch.load(model_name)
    else:
      model = torch.load(model_name, map_location='cpu')
    train_input = val_data[len(val_data)-1][0].squeeze()#使用train data的最後199筆
    result_list = predict_future(train_input,testing_data)

    #output answer
    with open(args.output, 'w') as output_file:
      cur_hold = 0
      for row in result_list:
        # We will perform your action as the open price in the next day.
        if((row[1]-row[0])>0 and cur_hold<1):#股票為0或-1時可買
          action = 1
          cur_hold+=1
        elif((row[1]-row[0])<0 and cur_hold>-1):#股票為1或0時可賣
          action = -1
          cur_hold-=1
        else:
          action = 0
        output_file.write(str(action)+'\n')

        # this is your option, you can leave it empty.
        