# BYOL
## A BYOL implementation in pytorch [Bootstrap Your Own Latent A New Approach to Self-Supervised Learning](https://arxiv.org/pdf/2006.07733.pdf)


![byol architecture](https://github.com/markpesic/byol/blob/master/images/byol.png?raw=true)


## Model
```python
from BYOL.byol import BYOL

model = BYOL(input_size=2048,
    hidden_size=4096,
    output_size=256,
    depth_proj=2,
    depth_pred=2,
    closedFormPredicator = False,
    EAvg = True,
    t = 0.996,
    backend='resnet50',
    pretrained=False)
   
```

## Training
```python
import torch
from BYOL.byol import BYOL
from BYOL.utils import criterion, get_byol_transforms
#train_loader, size, mean, std, lr and device given by the users

t, t1, _ = get_byol_transforms(size, mean, std)

model = BYOL(input_size=2048,
    hidden_size=4096,
    output_size=256,
    depth_proj=2,
    depth_pred=2,
    closedFormPredicator = False,
    EAvg = True,
    t = 0.996,
    backend='resnet50',
    pretrained=False)
    
model = model.to(device)
optimizer = torch.optim.SGD( model.parameters(), lr=lr, momentum= 0.9, weight_decay=1.5e-4)

for epoch in range(30):
    model.train()
    for batch, _ in train_loader:
        batch = batch.to(device)
        x = transform(batch)
        x1 = transform1(batch)

        onlinex, onlinex1, targetx, targetx1 = model(x, x1)
        loss = criterion(onlinex, targetx1, onlinex1, targetx)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        byol.updateTargetNetwork()

```

## Citation
```bibtex
@misc{grill2020bootstrap,
    title = {Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning},
    author = {Jean-Bastien Grill and Florian Strub and Florent Altché and Corentin Tallec and Pierre H. Richemond and Elena Buchatskaya and Carl Doersch and Bernardo Avila Pires and Zhaohan Daniel Guo and Mohammad Gheshlaghi Azar and Bilal Piot and Koray Kavukcuoglu and Rémi Munos and Michal Valko},
    year = {2020},
    eprint = {2006.07733},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG}
}
```
