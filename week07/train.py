import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from sennet import SenResNet

# Function to train the model
def train_model(model, train_loader, dev_loader, criterion, optimizer, num_epochs=10, device='cuda:0', checkpoint_dir='checkpoints', scheduler=None):
    model.to(device)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    best_dev_acc = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        #train
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            # stats
            running_loss += loss.item() * X_batch.size(0)
            _, predicted = outputs.max(1)
            total += y_batch.size(0)
            correct += predicted.eq(y_batch).sum().item()
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}')
        
        #dev
        model.eval()
        dev_loss = 0.0
        dev_correct = 0
        dev_total = 0
        
        with torch.no_grad():
            for X_dev, y_dev in dev_loader:
                X_dev, y_dev = X_dev.to(device), y_dev.to(device)
                outputs = model(X_dev)
                loss = criterion(outputs, y_dev)
                
                dev_loss += loss.item() * X_dev.size(0)
                _, predicted = outputs.max(1)
                dev_total += y_dev.size(0)
                dev_correct += predicted.eq(y_dev).sum().item()
        
        dev_loss = dev_loss / len(dev_loader.dataset)
        dev_acc = dev_correct / dev_total

        if scheduler:
            scheduler.step(dev_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Dev Loss: {dev_loss:.4f}, Dev Acc: {dev_acc:.4f}')
        
        # save checkpoint every 20
        if (epoch+1) % 20 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': epoch_loss,
                'dev_loss': dev_loss,
                'dev_acc': dev_acc,
            }
            torch.save(checkpoint, f'{checkpoint_dir}/checkpoint_epoch_{epoch+1}.pth')
        
        # save model
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            torch.save(model.state_dict(), f'{checkpoint_dir}/best_model.pth')
            print(f'Saved new best model and backbone with dev accuracy: {dev_acc:.4f}')
    
    # final final model and backbone
    torch.save(model.state_dict(), f'{checkpoint_dir}/FULL_model.pth')
    
    return model

def main():
    # Data paths
    imagenet_dir = '/media/data/kuba/imgnet_dataset'
    train_dir = os.path.join(imagenet_dir, 'train')
    dev_dir = os.path.join(imagenet_dir, 'val')
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    dev_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # data set up
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    dev_dataset = datasets.ImageFolder(dev_dir, transform=dev_transform)
    
    batch_size = 64  
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=4, pin_memory=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=4, pin_memory=True)
    
    # other set up
    device = 'cuda:1'
    model = SenResNet(num_classes=1000)  
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    
    train_model(model, train_loader, dev_loader, criterion, optimizer, num_epochs=90, device=device, checkpoint_dir='imagenet_checkpoints', scheduler=scheduler)

if __name__ == "__main__":
    main()