
Input Layer: Input Shape: [None, 3, 32, 32]
Conv2D: Filter Shape: [128, 3, 3, 3], pad 1
BN
Binarization
Conv2D: Filter: [128, 128, 3, 3], pad 1
MaxPool (2x2)
BN
Binarization
Conv2D: Filter: [256, 128, 3, 3], pad 1
BN
Binarization
Conv2D: Filter: [256, 256, 3, 3], pad 1
MaxPool (2x2)
BN
Binarization
Conv2D: Filter: [512, 256, 3, 3], pad 1
BN
Binarization
Conv2D: Filter:[512, 512, 3, 3], pad 1
MaxPool (2x2)
BN
Binarization
Dense: Nodes: 1024
BN
Binarization
Dense: Nodes: 1024
BN
Binarization
Dense: Nodes: 10
BN


