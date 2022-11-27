#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing the libraries

import numpy as np
import matplotlib.pyplot as plt 
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adam
from PIL import Image
from tabulate import tabulate
import torch.nn.functional as F
import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#visualization of the images in the data set

image = Image.open(r"D:\Uni-Siegen\4th Sem (SS 2022)\RAML\DeepFake\Dataset\train\train_bicubic\imagewoof\0.jpg")
print('The format of the image is: ',image.format)
print('The mode the image is: ',image.mode)
print('The size of the images in the data set is : ',image.size)
#plt.imshow(image)
#plt.show()




#defining the path of the training and testing data

data_dir = 'D:/Uni-Siegen/4th Sem (SS 2022)/RAML/DeepFake/Dataset/'
train_data_path = ['train/train_bicubic','train/train_bilinear','train/train_pixelshuffle','train/combo']
test_data_path = ['test/bicubic', 'test/bilinear', 'test/pixelshuffle','test/combo']

"""
#selecting of the desired data set for training ans testing the model

print('\nThe Training and Test Data is as follows') 
table_1 = [[1,'Bicubic','Bicubic'],[2, 'Bilinear','Bilinear'], [3, 'Pixel shuffle','Pixel shuffle'],[4,'All the above']]
header = ["Sr No", "Training Data", "Testing Data"]
print(tabulate(table_1, header))



#assignment of the value of our selection


train_path = int(input(('Enter the Sr No. of the data set for training: ')))-1
test_path =  int(input(('Enter the Sr No. of the data set for testing: ')))-1


#selecting of the desired data set for training ans testing the model


print('Select the Training and Testing datasets: ')
display_data_1 = wdg.Dropdown(
    options=[('Bicubic', 0), ('Bilinear', 1), ('Pixel shuffle', 2), ('All the above', 3)],
    value = 0,
    description='Training:',
)
display_data_2 = wdg.Dropdown(
    options=[('Bicubic', 0), ('Bilinear', 1), ('Pixel shuffle', 2), ('All the above', 3)],
    value = 0,
    description='Testing:',
)
display(wdg.HBox([display_data_1, display_data_2]))
#display(display_data_1)
#display(display_data_2)



#assignment of the value of our selection

train_path = display_data_1.value
test_path = display_data_2.value
#print(train_path,test_path)
"""
#declaration of the lists to aggregate the losses and accuracies

combo_train_losses = []
combo_train_accu = []
combo_test_losses = []
combo_test_accu = []
combo_validation_losses = []
combo_validation_accu = []

#iterate over different training paths to train over Bicibic, Bilinear, Pixel Shuffle and combinesd data.       
for idx_tr,trn in enumerate(train_data_path):
    
        #initilization of the hyperparamaters

        batch_size = 40
        num_epochs = 50
        best_accuracy=0.0
        min_validation_loss = np.inf
        the_last_loss = 100
        patience = 2
        trigger_times = 0
        final_epoch = 0
        image_size = [32,32]

        #declaration of lists to append the performance measurement parameters
        
        train_losses = []
        train_accu = []
        test_losses = []
        test_accu = []
        validation_losses = []
        validation_accu = []


        data_transforms = {
                'train': transforms.Compose([
                #transforms.Grayscale(1),
                transforms.Resize(image_size), 
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()]),
        
                'test': transforms.Compose([
                #transforms.Grayscale(1),
                transforms.Resize(image_size) ,
                transforms.ToTensor()])}
            
        train_dataset = datasets.ImageFolder(os.path.join(data_dir,trn),data_transforms['train'])
        
        #Split the training data into traning and validation datasets in the 80:20 ratio
        percentage_split = 0.8
        val_percentage_split = round(1 - percentage_split, 1)
        val_data_len = int(val_percentage_split*train_dataset.__len__())
        train_data_len = train_dataset.__len__() - val_data_len
        #print(train_data_len,val_data_len)
        
        train_dataset, validation_dataset = random_split(train_dataset,[train_data_len,val_data_len])
        
        """
        train_sampler = SubsetRandomSampler(list(range(1600-320)))
        validation_sampler = SubsetRandomSampler(list(range(320)))
        
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, sampler = train_sampler)
        validation_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, sampler = validation_sampler)
        """
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size)
        validation_loader = DataLoader(dataset=validation_dataset, batch_size=batch_size)
        
        
        num_train_sam = train_loader.__len__()*train_loader.batch_size
        num_valid_sam = validation_loader.__len__()*validation_loader.batch_size
        
        print('\nThere are',num_train_sam, 'train samples')
        print('There are',num_valid_sam, 'validation samples')
        
        
        #displaying the labels for real and fake images
        
        print('\nThe Classification is as follows') 
        table_2 = [[0,'Real Image'],[1, 'Fake Image']]
        header = ["Label", "Image Classification"]
        print(tabulate(table_2, header))
        
        """
        dataiter = iter(train_loader)
        images, labels = dataiter.next()
        #print(labels.shape)
        label= labels.numpy()
        #print(label)

        index = 0

        for index in range(len(label)):
            if (label[index] != 0):
                label[index] = 1
        """
        
        #visualizing the images and their respective labels of the loaded images
        
        dataiter = iter(train_loader)
        images, labels = dataiter.next()
        print('\nThe loaded batch of the images is a tensor with size: ',images.shape)
        #print(labels.shape)
        #
        label= labels.numpy()
        print('The loaded batch of the images is a tensor with labels: ', label[:4])
        #print(label[:4])
        
        
        classes = ['Real image', 'Fake image']
        
        fig_1 = plt.figure()
        for i in range(4):
          plt.subplot(2,2,i+1)
          plt.tight_layout()
          plt.imshow(images[i][0], cmap='gray', interpolation='none')
          plt.title("Reality: {}".format(classes[labels[i]]))
          plt.xticks([])
          plt.yticks([])
        fig_1
        
        """
        fig_2 = plt.figure()
        for i, (images, labels) in enumerate(train_loader):
            plt.subplot(2,2,i+1)
            plt.tight_layout()
            plt.imshow(images[i][0], cmap='gray', interpolation='none')
            plt.title("Reality: {}".format(classes[labels[i]]))
            plt.xticks([])
            plt.yticks([])
        fig_2
        
        """
        print('\n')
        
        #defining the CNN model
        
        class ConvNet(nn.Module):
            def __init__(self):
                super(ConvNet,self).__init__()
                #Input image shape: (50,3,32,32)
                self.conv1=nn.Conv2d(in_channels=3,out_channels=12,kernel_size=3,stride=1,padding=1)
                #image shape: (50,12,32,32)
                self.bn1=nn.BatchNorm2d(num_features=12)
                #image shape: (50,12,32,32)
                self.relu1=nn.ReLU()
                #image shape: (50,12,32,32) 
                self.pool=nn.MaxPool2d(kernel_size=2)
                #image shape: (50,12,16,16)
                self.conv2=nn.Conv2d(in_channels=12,out_channels=20,kernel_size=3,stride=1,padding=1)
                #image shape: (50,20,16,16)
                self.relu2=nn.ReLU()
                #image shape: (50,20,16,16)   
                self.conv3=nn.Conv2d(in_channels=20,out_channels=32,kernel_size=3,stride=1,padding=1)
                #image shape: (50,32,16,16)
                self.bn3=nn.BatchNorm2d(num_features=32)
                #image shape: (50,32,16,16)
                self.relu3=nn.ReLU()
                #image shape: (50,32,16,16)
                
                """
                self.pool=nn.MaxPool2d(kernel_size=2)
                        
                self.conv4=nn.Conv2d(in_channels=32,out_channels=45,kernel_size=3,stride=1,padding=1)
             
                self.bn4=nn.BatchNorm2d(num_features=45)
              
                self.relu4=nn.ReLU()
             
                """
                
                self.fc=nn.Linear(in_features=16 * 16 * 32,out_features=2)
                #self.fc1 = nn.Linear(in_features=16 * 16 * 32,out_features=5000)
                #self.fc2 = nn.Linear(in_features=5000,out_features=2)
                #self.fc3 = nn.Linear(in_features=2500,out_features=1250)
                #self.fc4 = nn.Linear(in_features=1250,out_features=2)
                
                
            def forward(self,input):
                output=self.conv1(input)
                output=self.bn1(output)
                output=self.relu1(output)
                    
                output=self.pool(output)
                    
                output=self.conv2(output)
                output=self.relu2(output)
                
                
                output=self.conv3(output)
                output=self.bn3(output)
                output=self.relu3(output)
                
                """
                output=self.pool(output)
                
               
                output=self.conv4(output)
                output=self.bn4(output)
                output=self.relu4(output)
                """  
                    
                output=output.view(-1, 32 * 16 * 16)
                   
                    
                output=self.fc(output)
                #output = F.relu(self.fc1(output))
                #output = F.relu(self.fc2(output))
                #output = F.relu(self.fc3(output))
                #output = F.relu(self.fc4(output))
                output = torch.sigmoid(output)
                
                return output
         
            
        model = ConvNet().to(device)
        
        #defining the loss function and the optimizer
        
        loss_function=nn.CrossEntropyLoss().to(device)
        #optimizer = torch.optim.SGD(model.parameters(), lr = 0.0001,weight_decay=0.001)
        optimizer=Adam(model.parameters(),lr=0.0001,weight_decay=0.001)
        
        print(device)    
        
        #training the model
        for epoch in range(num_epochs):
            
            model.train()
            train_accuracy=0.0
            train_loss=0.0
            
            for i, (images,labels) in enumerate(train_loader):
                if torch.cuda.is_available():
                    images=Variable(images.cuda())
                    labels=Variable(labels.cuda())
                    
                optimizer.zero_grad()
                
                outputs=model(images)
                loss = loss_function(outputs,labels)
                loss.backward()
                optimizer.step()
                
                
                train_loss+= loss.cpu().data*images.size(0)
                _,prediction=torch.max(outputs.data,1)
                
                train_accuracy+=int(torch.sum(prediction==labels.data))
                
            train_accuracy = train_accuracy/num_train_sam*100
            train_loss = train_loss/num_train_sam
            train_accu.append(train_accuracy)
            train_losses.append(train_loss)
            
             
            #validating the model
            
            model.eval()
            validation_accuracy=0.0
            validation_loss = 0.0
         
            for i, (images,labels) in enumerate(validation_loader):
                if torch.cuda.is_available():
                    images=Variable(images.cuda())
                    labels=Variable(labels.cuda())
                    
                outputs=model(images)
                loss = loss_function(outputs,labels)
                
                validation_loss+= loss.cpu().data*images.size(0)
                _,prediction=torch.max(outputs.data,1)
                
        
                validation_accuracy+=int(torch.sum(prediction==labels.data))
                   
            validation_accuracy=validation_accuracy/num_valid_sam*100
            validation_loss = validation_loss/num_valid_sam
            validation_losses.append(validation_loss)
            validation_accu.append(validation_accuracy)
                
           
                
            #print('\n') 
            if epoch < 9:
                print('Epoch:',epoch+1,'     Train Loss:',format(train_loss.item(),".4f"),'    Train Accuracy:',format(train_accuracy,".4f"),'     Validation Loss:',format(validation_loss.item(),".4f"),'    Validation Accuracy:',format(validation_accuracy,".4f"))
                #print(tabulate([epoch,train_loss,train_accuracy,test_accuracy], headers = ['Epoch','Train Loss','Train Accuracy','Test Accuracy']))
            else:
                print('Epoch:',epoch+1,'    Train Loss:',format(train_loss.item(),".4f"),'    Train Accuracy:',format(train_accuracy,".4f"),'     Validation Loss:',format(validation_loss.item(),".4f"),'    Validation Accuracy:',format(validation_accuracy,".4f"))       
        
        
            #Implementation of Early stop to prevent overfitting
            
            if validation_loss > the_last_loss:
                trigger_times += 1
                print('Patirnce:', trigger_times)
        
                if trigger_times >= patience:
                    final_epoch = epoch
                    print('Early stopping\n\nStart of the testing process.\n')
                    break
        
            else:
                print('Patirnce: 0')
                trigger_times = 0
        
            the_last_loss = validation_loss
                
        
        combo_train_losses.append(train_loss.item())
        combo_validation_losses.append(validation_loss.item())
        combo_train_accu.append(train_accuracy)
        combo_validation_accu.append(validation_accuracy)
        
       
        #iterate over different testing paths to test over Bicibic, Bilinear, Pixel Shuffle and combinesd data.
        
        for idx_tst,tst in enumerate(test_data_path):
            
            model.eval()
            test_accuracy=0.0
            test_loss = 0.0
            
            test_dataset = datasets.ImageFolder(os.path.join(data_dir,tst),data_transforms['test'])
            test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
            num_test_sam = test_loader.__len__()*test_loader.batch_size
            print('There are',num_test_sam, 'test samples\n')
            
            for i, (images,labels) in enumerate(test_loader):
                if torch.cuda.is_available():
                    images=Variable(images.cuda())
                    labels=Variable(labels.cuda())

                outputs=model(images)
                loss = loss_function(outputs,labels)

                test_loss+= loss.cpu().data*images.size(0)
                _,prediction=torch.max(outputs.data,1)


                test_accuracy+=int(torch.sum(prediction==labels.data))


            test_accuracy=test_accuracy/num_test_sam*100
            test_loss = test_loss/num_test_sam


            #test_losses.append(test_loss)
            #test_accu.append(test_accuracy)

            table_1 = ['Bicubic','Bilinear','Pixel shuffle','Combined data']

            print('After carring out the training with ' + str(table_1[idx_tr]) + ' dataset and testing the model with ' +str(table_1[idx_tst]) + ' we obtain:')
            if epoch < 9:
                print('Test Loss:',format(test_loss.item(),".4f"),  '     Test Accuracy:',format(test_accuracy,".4f"))
                #print(tabulate([epoch,train_loss,train_accuracy,test_accuracy], headers = ['Epoch','Train Loss','Train Accuracy','Test Accuracy']))
            else:
                print('Test Loss:',format(test_loss.item(),".4f"),  '     Test Accuracy:',format(test_accuracy,".4f")) 
             
            #appending all the losses and accuracy to the respective lists
            combo_test_losses.append(test_loss.item())
            combo_test_accu.append(test_accuracy)
            combo_train_losses.append(0)
            combo_validation_losses.append(0)
            combo_train_accu.append(0)
            combo_validation_accu.append(0)
            
            
            plt.rcParams["figure.figsize"] = (10,6)
            
            #plotting the training vs validation lossses
            
            fig_3 = plt.figure() 
            plt.plot(train_losses,'-o')
            #plt.plot(test_losses,'-o')
            plt.plot(validation_losses,'-o')
            plt.xlabel('epoch')
            plt.ylabel('losses')
            plt.legend(['Train','Validation'])
            plt.title('Train vs Validation losses')
            plt.show()

            #plotting the training vs validation accuracies
            
            fig_4 = plt.figure() 
            plt.plot(train_accu,'-o')
            #plt.plot(test_accu,'-o')
            plt.plot(validation_accu,'-o')
            plt.xlabel('epoch')
            plt.ylabel('accuracy')
            plt.legend(['Train', 'Validation'])
            plt.title('Train vs Validation Accuracy')
            plt.show()


# In[2]:


#representation of the losses and accuracies in a table

table_3 = ['Bicubic','Bicubic','Bicubic','Bicubic','Bilinear','Bilinear','Bilinear','Bilinear','Pixel shuffle','Pixel shuffle','Pixel shuffle','Pixel shuffle','Combined data','Combined data','Combined data','Combined data']
table_4 = ['Bicubic','Bilinear','Pixel shuffle','Combined data','Bicubic','Bilinear','Pixel shuffle','Combined data','Bicubic','Bilinear','Pixel shuffle','Combined data','Bicubic','Bilinear','Pixel shuffle','Combined data']

print('\nFinal Conclusion\n\n') 

final_table = []

for i in range(16):
    final_table.append([table_3[i],table_4[i],combo_train_losses[i],combo_train_accu[i],combo_test_losses[i],combo_test_accu[i]])
#print(final_table)

header = ["Training Data","Test Data", "Training Loss", "Training Accuracy", "Testing Loss", "Testing Accuracy"]
print(tabulate(final_table, header))

table_3 = ['Bicubic','Bicubic','Bicubic','Bicubic','Bilinear','Bilinear','Bilinear','Bilinear','Pixel shuffle','Pixel shuffle','Pixel shuffle','Pixel shuffle','Combined data','Combined data','Combined data','Combined data']
table_4 = ['Bicubic','Bilinear','Pixel shuffle','Combined data','Bicubic','Bilinear','Pixel shuffle','Combined data','Bicubic','Bilinear','Pixel shuffle','Combined data','Bicubic','Bilinear','Pixel shuffle','Combined data']

print('\n\n') 

final_table_2 = []

for i in range(16):
    final_table_2.append([table_3[i],table_4[i],combo_validation_losses[i],combo_validation_accu[i]])
#print(final_table)

header_2 = ["Training Data","Test Data","Validation Loss", "Validation Accuracy"]
print(tabulate(final_table_2, header_2))


# In[ ]:





# In[ ]:





# In[ ]:




