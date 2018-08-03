import os
import data_preparation
from data_preparation import THCNewsDataSet, batch_iter
import torch
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from config import Config
from textcnn import TextCNN
import torch.nn as nn
import torch.nn.functional as F

device = torch.device(
    "cuda:0") if torch.cuda.is_available() else torch.device("cpu")


model = TextCNN()

model = model.to(device)

opt = optim.Adam(model.parameters())

criterion = nn.CrossEntropyLoss()


def save_model(model, model_name="best_model_sofa.pkl", model_save_dir="./trained_models/"):
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    path = os.path.join(model_save_dir, model_name)

    torch.save(model.state_dict(), path)
    print("saved model state dict at :"+path)


def load_model(model, model_path="./trained_models/best_model_sofa.pkl"):
    if not os.path.exists(model_path):
        raise RuntimeError("model path not exist...")
    model.load_state_dict(torch.load(model_path))
    print("loaded model state dict from:"+model_path)


def train():
    num_epochs = Config.num_epochs
    train_dataset = THCNewsDataSet()
    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=Config.batch_size, shuffle=True, num_workers=4)
    val_dataset = THCNewsDataSet(Config.val_path)
    val_dataloader = DataLoader(
        dataset=val_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=4)
    test_dataset = THCNewsDataSet(Config.test_path)
    test_dataloader = DataLoader(
        dataset=test_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=4)

    flag = True
    print("start training ....")

    best_val_acc = 0.0

    for epoch in range(1, num_epochs+1):
        num_train = 0
        num_train_correct = 0
        epoch_loss = 0.0
        model.train()
        for batch_x, batch_y in train_dataloader:
            # batch_x = batch_x.to(device)
            # batch_y = batch_y.to(device)
            batch_x = torch.tensor(batch_x, device=device, dtype=torch.long)
            batch_y = torch.tensor(batch_y, device=device, dtype=torch.long)
            opt.zero_grad()
            out_put = model(batch_x)
            loss = criterion(out_put, batch_y)
            epoch_loss += loss.item()/len(batch_x)

            num_train += batch_x.size(0)
            with torch.no_grad():
                out_labels = torch.argmax(out_put, dim=1)
                out_correct = (out_labels == batch_y).sum()
                num_train_correct += out_correct.item()

            loss.backward()
            opt.step()
        print("epoch:{}/{} ,loss:{} ,accuracy:{} {}/{}".format(epoch,
                                                               num_epochs,
                                                               epoch_loss/epoch,
                                                               num_train_correct/num_train,
                                                               num_train_correct, num_train))
        model.eval()
        num_val = 0
        num_val_correct = 0
        with torch.no_grad():
            for val_x, val_y in val_dataloader:
                val_x = val_x.to(device)
                val_y = val_y.to(device)

                val_out = model(val_x)
                num_val += val_x.size(0)
                val_labels = torch.argmax(val_out, dim=1)
                val_correct = (val_labels == val_y).sum()
                num_val_correct += val_correct.item()
        val_acc = num_val_correct / num_val

        print("val accuracy:{:.4f}  {}/{}".format(val_acc,
                                                  num_val_correct, num_val))
        if val_acc > best_val_acc:
            print("val improved .....")
            save_model(model)
            best_val_acc = val_acc
            print()

    print("train done ...")
    print("load best for testing...")
    load_model(model)
    model.eval()
    num_test = 0
    num_test_correct = 0
    with torch.no_grad():
        for test_x, test_y in test_dataloader:
            test_x = test_x.to(device)
            test_y = test_y.to(device)

            test_out = model(test_x)
            num_test += test_x.size(0)
            test_labels = torch.argmax(test_out, dim=1)
            test_correct = (test_labels == test_y).sum()
            num_test_correct += test_correct.item()

    print("test accuracy:{:.4f}  {}/{}".format(num_test_correct /
                                               num_test, num_test_correct, num_test))


if __name__ == "__main__":
    train()
