# NeuroSAM
Model 1: MAE=5.3
batch_size: 16, optimizer: AdamW, loss: MSE, lr: 1e-4, split: 0.7 

Model 2: MAE=6.x
batch_size: 16, optimizer: AdamW, loss: MSE, lr: 1e-4, split: 0.7

Model 3: MAE=5.3
batch_size: 32, optimizer: AdamW, loss: MSE, lr: 1e-5, split:0.8

Model 4: MAE 5.x
batch_size: 16, optimizer: AdamW, loss: MSE, lr: 1e-4, split: 0.8 uniform age dist

Model 5: MAE=3.77
batch_size: 16, optimizer: AdamW, loss: MSE, lr: 1e-4 (scheduler), split:0.8, full sample of 15300 subjects

Model 6: MAE=4.7
batch_size: 16, optimizer: AdamW, loss: MSE, lr: 1e-4 (scheduler=cosine), split:0.8, full sample of 15000 subjects (JUK and RRIB not included on training)

Model 7: MAE=4.41 (??????)
same as 5 but patience=15 instead of 20 and JUK and RRIB not included on training

model 8: MAE= 6.3
the same but simplified model (less learnable parameters) and not using the paper's datasets, now training on the whole database and testing with externals loss= smoothL1

model 9: MAE= 6.5
same model as model 8 but with the original size of learnable parameters, batch_size=8