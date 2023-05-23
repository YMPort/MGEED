import argparse
import numpy as np
import pandas as pd
import torch
import torchvision
from torchvision import models, transforms
from utils import resnet_extend
from PIL import Image
import os

parser = argparse.ArgumentParser(description='')
parser.add_argument('--epoch', default=10)
parser.add_argument('--img_dir', type=str, default='..')
parser.add_argument('--annotation_file', type=str, default='label.csv')
parser.add_argument('--model_weights', type=str, default='weights.pth')
args = parser.parse_args()

class AR_Dataset(torch.utils.data.Dataset):
    def __init__(self, annotation_file, img_dir, num_classes=9, transforms=None):
        self.img_labels = pd.read_csv(annotation_file)
        self.img_dir = img_dir
        self.transforms = transforms
        self.num_classes = num_classes
        
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        im_path = os.path.join(self.img_dir,self.img_labels.iloc[idx,1])
        image = Image.open(im_path)
        Emotion = self.img_labels.iloc[idx, 92]-1
        VA = [self.img_labels.iloc[idx, 93], self.img_labels.iloc[idx, 94]]
        OMG = [self.img_labels.iloc[idx, i] for i in range(2,22)]
        EEG = [self.img_labels.iloc[idx, i] for i in range(22,92)]
        sub = self.img_labels.iloc[idx, 0]
        if self.transforms:
            image = self.transforms(image)
            Emotion = torch.as_tensor(Emotion, dtype=torch.int64)
            VA = torch.as_tensor(VA, dtype=torch.float32)
            OMG = torch.as_tensor(OMG, dtype=torch.float32)
            EEG = torch.as_tensor(EEG, dtype=torch.float32)
            sub = torch.as_tensor(sub, dtype=torch.int64)
        return image, OMG, EEG, Emotion, VA, sub

class RecogModel(torch.nn.Module):
    def __init__(self, num_classes=9):
        super(RecogModel, self).__init__()
        self.num_classes = num_classes
        self.ResNet = models.resnet101(pretrained=True)
        self.fc_extend = torch.nn.Linear(2048+90, num_classes)
    
    def forward(self, im, omg, eeg):
        for param in self.ResNet.parameters():
            param.requires_grad = False
        
        bound_method = resnet_extend.forward.__get__(self.ResNet, self.ResNet.__class__)
        setattr(self.ResNet, 'forward', bound_method)
        x = self.ResNet(im)
        omg = torch.nn.functional.normalize(omg, p=2.0)
        eeg = torch.nn.functional.normalize(eeg, p=2.0)
        x = torch.cat((x, omg, eeg), dim=1)
        
        return self.fc_extend(x)

def train_test(data_root, data_file, train_ind, test_ind, num_epochs=15, save_weights_path=None):
    data_transforms = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    dataset = AR_Dataset(data_file, data_root, transforms=data_transforms)
    train_dataset = torch.utils.data.Subset(dataset,train_ind)
    test_dataset = torch.utils.data.Subset(dataset,test_ind)
    num_classes = dataset.num_classes
    emo = dataset.img_labels.Emotion[train_ind]
    count_emo = emo.value_counts().sort_index().tolist()
    emo=np.array(emo.tolist())-1

    #class_weight = np.log(np.max(count_emo)/count_emo)+1
    class_weight = 1.0 / np.array(count_emo)
    sample_weight = class_weight[emo]
    sample_weight = torch.tensor(sample_weight,dtype=torch.float)#.to(device)
    sampler = torch.utils.data.WeightedRandomSampler(sample_weight, 6400, replacement=True)
    
    test_size = len(test_dataset)
    data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, sampler=sampler)
    data_loader_test = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    #class_weight = torch.tensor(class_weight,dtype=torch.float).to(device)
    model = RecogModel(num_classes)
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9,weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    ce_loss = torch.nn.CrossEntropyLoss() #weight = class_weight
    mse_loss = torch.nn.MSELoss()

    for epoch in range(num_epochs):
        model.train()
        for images, omg, eeg, emotion, va, sub in data_loader:
            images = images.to(device)
            omg = omg.to(device)
            eeg = eeg.to(device)
            emotion = emotion.to(device)
            va = va.to(device)
            _, counts = np.unique(emotion.cpu().numpy(), return_counts=True)
            optimizer.zero_grad()
            outputs = model(images, omg, eeg)
            _, preds = torch.max(outputs[:,:num_classes-2], 1)
            class_loss = ce_loss(outputs[:,:num_classes-2],emotion)
            regre_loss = mse_loss(outputs[:,num_classes-2:],va)
            loss = class_loss + regre_loss
            #print(class_loss.item(),regre_loss.item())
            loss.backward()
            optimizer.step()
            #batch_loss = loss.item() * images.size(0)
            #batch_corrects = torch.sum(preds==emotion.data)
            #print("loss: {:.4f} Acc: {:.4f}".format(batch_loss, batch_corrects.double()/images.size(0)))
        lr_scheduler.step()
        
        model.eval()
        confusion_matrix = torch.zeros(num_classes-2, num_classes-2)
        v_mse = 0.0
        a_mse = 0.0
        for images, omg, eeg, emotion, va, sub in data_loader_test:
            images = images.to(device)
            omg = omg.to(device)
            eeg = eeg.to(device)
            emotion = emotion.to(device)
            va = va.to(device)
            outputs = model(images, omg, eeg)
            vmse = mse_loss(outputs[:,num_classes-2],va[:,0])
            amse = mse_loss(outputs[:,num_classes-1],va[:,1])
            v_mse += vmse.item() * images.size(0)
            a_mse += amse.item() * images.size(0)
            _, emo_preds = torch.max(outputs[:,:num_classes-2], 1)
            for t, p in zip(emotion.view(-1), emo_preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
        #acc_per_emo = confusion_matrix.diag()/confusion_matrix.sum(1)
        #acc = confusion_matrix.diag().sum()/test_size
        print("valence_mse: {:.4f} arousal_mse: {:.4f} conf: {}".format(v_mse/test_size, a_mse/test_size, confusion_matrix))
    return model, confusion_matrix, vmse, amse

def evaluate(data_root,data_file):
    img_labels = pd.read_csv(data_file)
    ind = img_labels.SubSet
    indices1 = ind[ind==1].index
    indices2 = ind[ind==2].index
    _, confusion_m1, vmse1, amse1 = train_test(data_root, data_file, indices2, indices1)
    _, confusion_m2, vmse2, amse2 = train_test(data_root, data_file, indices1, indices2)
    cm_tensor = confusion_m1 + confusion_m2
    cm = cm_tensor.cpu().detach().numpy()
    v_rms = torch.sqrt((vmse1+vmse2)/torch.sum(cm_tensor))
    v_rms = v_rms.cpu().detach().numpy()
    a_rms = torch.sqrt((amse1+amse2)/torch.sum(cm_tensor))
    a_rms = a_rms.cpu().detach().numpy()
    tpp = np.diag(cm)/np.sum(cm,1)
    tn=np.sum(cm,0)-np.diag(cm)
    n=np.sum(cm)-np.sum(cm,1)
    tnn=tn/n
    b_acc = (tpp+tnn)/2
    return cm, v_rms, a_rms, b_acc
    
cm, v_rms, a_rms, b_acc = evaluate(args.img_dir,args.annotation_file)