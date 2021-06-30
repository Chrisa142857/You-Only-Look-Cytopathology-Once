## Four Classifiers at WSI-level
+ SVM
    + sklearn.SVM.svc
+ RNN
    + torch.nn.RNN
+ LSTM
    + torch.nn.LSTM
+ Transformer
    + https://github.com/lucidrains/vit-pytorch

## Train for WSI classification

For RNN, LSTM, Transformer using
```
python train.py
```

For SVM using
```
python svm_train.py
```

## Testing and collecting results from the best model
```
python test_roc.py
```
