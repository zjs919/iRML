import torch
import torch.nn.functional as F
from sklearn import metrics

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def test(model, features, adj, labels, idx_test):
    model.eval()
    output = model(features, adj)
    pred_labels=torch.argmax(output,axis=1)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    
    #acc_test = metrics.accuracy_score(labels[idx_test].cpu().detach().numpy(), pred_labels[idx_test].cpu().detach().numpy())
    #f1_test=metrics.f1_score(labels[idx_test].cpu().detach().numpy(), pred_labels[idx_test].cpu().detach().numpy(),average='weighted')
    #auc_test=metrics.roc_auc_score(one_hot(labels[idx_test].cpu().detach().numpy()), output[idx_test].cpu().detach().numpy(),multi_class='ovr',average='weighted')
    
    return loss_test.item(), acc_test.item()#, f1_test, auc_test

def Block_matrix_train(epoch, model, optimizer, features, adj, labels, split_data_index, idx_train):
    model.train()
    optimizer.zero_grad()

    output = model(features[split_data_index], adj[split_data_index][:, split_data_index]) #adj, keep block matrix
    
    loss_train = F.nll_loss(output[idx_train], labels[split_data_index][idx_train])
    
    
    acc_train = accuracy(output[idx_train], labels[split_data_index][idx_train])
    

    loss_train.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    #print("epoch", epoch, 
    #      "train", loss_train.item(), acc_train.item())
    return loss_train.item(), acc_train.item()


def Block_matrix_train_batch(epoch, model, optimizer, features, adj, labels, split_data_index_list, idx_train_list ):
    model.train()
    optimizer.zero_grad()
    
    loss_train = None
    count = 0 
    for i in range(len(split_data_index_list)):
        split_data_index = split_data_index_list[i]
        idx_train = idx_train_list[i]
        output = model(features[split_data_index], adj[split_data_index][:, split_data_index]) #adj, keep block matrix
        if loss_train == None:
            loss_train = len(idx_train) * F.nll_loss(output[idx_train], labels[split_data_index][idx_train])
        else:
            loss_train += len(idx_train) * F.nll_loss(output[idx_train], labels[split_data_index][idx_train])
        acc_train = accuracy(output[idx_train], labels[split_data_index][idx_train])
        
        count += len(idx_train)
       
    
    loss_train /= count
    
    loss_train.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    #print("epoch", epoch, 
    #      "train", loss_train.item(), acc_train.item())
    return loss_train.item(), acc_train.item()
