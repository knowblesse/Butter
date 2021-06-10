"""
목적 : 주어진 비디오 데이터에서 위치와 head direction을 뽑아내는 알고리즘의 작성

ResNet 등의 pretrained CNN을 활용해서 데이터 처리 예정.

Method 1 : 전체 파일을 그대로 넣어서 데이터 처리.

Method 2 : opencv로 ROI를 작게 만들어서 처리
- 세영이 blob detection 파트 제작 중

Conv2DTranspose layer?? maybe expanding the pre-convolved layer?
    Transposed convolution layer (deconvolution)
    
layers.Embedding
layers.concatenate : merge all available feature into a single large vector
layers.add([output1, output2])
"""