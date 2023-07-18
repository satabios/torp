# Torch Model Compression(tomoco)

This is a Pytorch Helper package aimed to aid the workflow of deep learning model development and deployment. 


1. (1) This packages has boiler plate defintions that can ease the development of torch model development
2. (2) Pruning Techniques are being imported from Tomoco Package
3. (3) Model Quantization and Deployment features are in the development pipeline which will be available for use soon.
## Package install:

```python

pip install torp

```


## Usage:

```python
from torp import train, evaluate






lr = 0.001 
n_classes = 10			 # Intended for output classes
epochs = 5                         # Set no. of training epochs
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    # Pick the device availble
batch_size = 64			# Set batch size
optim = 0
training =1                        # Set training to 1 if you would like to train post to prune
criterion = nn.CrossEntropyLoss()  # Set your criterion here

train_dataset = CIFAR10(root='data/', download=True, transform=transforms.ToTensor())
valid_dataset = CIFAR10(root='data/',  download=True,train=False, transform=transforms.ToTensor())

# define the data loaders
train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=config.batch_size, shuffle=False)


#Use a cutom model or pull a model from a repository

res50 = timm.create_model("resnet50", pretrained=True).to(config.device)

optim =  torch.optim.Adam(res50.parameters(), config.lr=0.001,  amsgrad=True) 
train(  model, train_loader, criterion, optim, scheduler, callbacks = None)

evaluate(model, valid_loader)

```



### To-Do

- [x] Universal Channel-Wise Pruning
- [x] Update Tutorials
- [+] Fine Grained Purning (In-Progress)
- [ ] Quantisation
- [ ] Universal AutoML package
- [ ] Introduction of Sparsification in Pipeline