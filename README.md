# BYOL
# A BYOL implementation in pytorch [Bootstrap Your Own Latent A New Approach to Self-Supervised Learning](https://arxiv.org/pdf/2006.07733.pdf)


![byol architecture](https://github.com/markpesic/byol/blob/master/images/byol.png?raw=true)


To instantiate a byol model
```python
from BYOL.byol import BYOL

model = BYOL(input_size=2048,
    hidden_size=4096,
    output_size=256,
    depth_proj=2,
    depth_pred=2,
    closedFormPredicator = False,
    EAvg = True,
    t = 0.996)
    
   
```
