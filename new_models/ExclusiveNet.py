from random import randrange
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from .registry import register_model


def fit_channels(layers, ratio):
    layers[1][0].block[0][0] = nn.Conv2d(in_channels=32 + int(ratio * 3), out_channels=32, kernel_size=(3,3), padding=(1,1))

    layers[2][0].block[0][0] = nn.Conv2d(in_channels=16 + int(ratio * 35), out_channels=96, kernel_size=(1,1))

    layers[3][0].block[0][0] = nn.Conv2d(in_channels=24 + int(ratio * 51), out_channels=144, kernel_size=(1,1))

    layers[4][0].block[0][0] = nn.Conv2d(in_channels= 40 + int(ratio * 75), out_channels=240, kernel_size=(1,1))

    layers[5][0].block[0][0] = nn.Conv2d(in_channels= 80 + int(ratio * 115), out_channels=480, kernel_size=(1,1))

    layers[6][0].block[0][0] = nn.Conv2d(in_channels= 112 + int(ratio * 195), out_channels=672, kernel_size=(1,1))

    layers[7][0].block[0][0] = nn.Conv2d(in_channels= 192 + int(ratio * 307), out_channels=1152, kernel_size=(1,1))

    layers[8][0] = nn.Conv2d(in_channels= 320 + int(ratio * 499), out_channels=1280, kernel_size=(1,1))
    
    return layers

def greater(val1, val2, list):
    return len([x for x in list if x >= val1 and x < val2])

class ExclusiveNet_R(nn.Module):
    def __init__(self, skip_ratio=0.5):
        super().__init__()
        
        self.ratio = skip_ratio
        baseline = torchvision.models.efficientnet_b0()

        self.features = baseline.features
        self.features = fit_channels(self.features, self.ratio)

        self.layer_pools = [0, 2, 3, 4, 6] 
        if self.ratio < 0.334:
            self.layer_pools = self.layer_pools[1:]

        self.classifier = baseline.classifier
        self.classifier[1] = nn.Linear(in_features=1280 + int(self.ratio * 819), out_features=10, bias=True) #change for imagenet

        ##create random permutation for each flow forward channel
        self.permutations = None
        
        self.permutations = {}
        channels = [3, 32, 16, 24, 40, 80, 112, 192, 320, 1280]
        for i in range(10):
            for j in range(i, 10):
                self.permutations[i*8 + j] = torch.randperm(channels[i])[:int(channels[i] * self.ratio)]




    def forward(self, x):
        #there might be a more efficient implementation possible
        prev = [x]
        for i in range(len(self.features)):

            prev_pooled = [z[:,self.permutations[idx*8 + i], :, :] for idx, z in enumerate(prev[:-1])]
          
            prev_pooled = [z for z in prev_pooled if z.shape[1] > 0]
            for idx in range(len(prev_pooled)):
                val = greater(idx, i, self.layer_pools)
                
                p = F.adaptive_avg_pool2d
                
                size = int(prev_pooled[idx].shape[-1] / 2**val)
                prev_pooled[idx] = p(prev_pooled[idx], (size, size))
            inx = torch.cat(prev_pooled + [prev[-1]], dim=1) 
            

            prev.append(self.features[i](inx))

        prev_pooled = [z[:,torch.randperm(z.shape[1])[:int(z.shape[1] * self.ratio)], :, :] for z in prev[:-1]]
        prev_pooled = [z for z in prev_pooled if z.shape[1] > 0]
        for idx in range(len(prev_pooled)):
                val = greater(idx, 9, self.layer_pools)
                p = F.adaptive_avg_pool2d
                
                size = int(prev_pooled[idx].shape[-1] / 2**val)
                prev_pooled[idx] = p(prev_pooled[idx], (size, size))

        prev_pooled = [z for z in prev_pooled if z.shape[1] > 0]
        x = torch.cat(prev_pooled + [prev[-1]], dim=1) 
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def greater(self, val1, val2, list):
        return len([x for x in list if x > val1 and x < val2])

@register_model
class ExclusiveNet(nn.Module):
    #Current model working with 2,3 skips, check for 5 and 10
    def __init__(self, skip=5, exclusive_forward=False, skip_start=False, out_c=1000): 
        #TODO: if exlusive forward, remove all channels of indices greater than number of featuers and not greater than layer number
        super().__init__()
        self.skip = skip
        self.exclusive_forward = exclusive_forward
        self.skip_start = skip_start

        baseline = torchvision.models.efficientnet_b0()

        self.features = baseline.features
        self.features = self.fit_channels(self.features, skip)

        self.layer_pools = [0, 2, 3, 4, 6] #can remove 0 at start
        if not self.skip_start:
            self.layer_pools = self.layer_pools[1:]

        self.classifier = baseline.classifier
        start = 3 if self.skip_start else 0
        if skip == 2:
            self.classifier[1] = nn.Linear(in_features=1280 + (len(self.features) * (skip-1)) - 1 + start, out_features=out_c, bias=True) #need to check for skip=2, can change out_features
        elif skip == 1:
            self.classifier[1] = nn.Linear(in_features=1280 + len(self.features) * skip - 1 + start, out_features=out_c, bias=True) #out classes changed for imagenet, should be parameter
        elif skip == 3:
            self.classifier[1] = nn.Linear(in_features=1288 + start, out_features=out_c, bias=True)
        else:
            self.classifier[1] = nn.Linear(in_features=1288 + start, out_features=out_c, bias=True)

    def forward(self, x):
        start = [x] if self.skip_start else [] #we would want a third option allowing skip from start only to first 4 layers
        prev = []
        for i in range(len(self.features)):
            prev_pooled = [start[0]] if self.skip_start else []
            for j in range(len(prev[:-1])):
                d = i - j
                if d == 0:
                    prev_pooled.append(prev[:-1][j][:,(-d-1)*self.skip:,:,:])
                else:
                    prev_pooled.append(prev[:-1][j][:,(-d-1)*self.skip:-d*self.skip,:,:])
            for idx in range(len(prev_pooled)):
                val = self.greater(idx, i, self.layer_pools)
               
                p = F.adaptive_avg_pool2d
                size = int(prev_pooled[idx].shape[-1] / 2**val)
                prev_pooled[idx] = p(prev_pooled[idx], (size, size))
            

            if len(prev) == 0:
                inx = x
            else:
                temp = prev[-1] if not self.exclusive_forward else prev[-1][:,:-(8-i)*self.skip,:,:] if i < 8 else prev[-1]
                inx = torch.cat(prev_pooled + [temp], dim=1) 

            prev.append(self.features[i](inx))

        n = len(self.features)
        
        prev_pooled = start + [z[:,-n-1:-n, :, :] for z in prev[:-1]]
        for idx in range(len(prev_pooled)):
                val = self.greater(idx, 9, self.layer_pools)
               
                p = F.adaptive_avg_pool2d
                
                size = int(prev_pooled[idx].shape[-1] / 2**val)
                prev_pooled[idx] = p(prev_pooled[idx], (size, size))

        prev_pooled = [z for z in prev_pooled if z.shape[1] > 0]
        x = torch.cat(prev_pooled + [prev[-1]], dim=1) 
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return self.classifier(x)



    def fit_channels(self, layers, skip):
        start = 3 if self.skip_start else 0
        ex_f = 0 if not self.exclusive_forward else self.skip
        
        layers[0][0] = nn.Conv2d(3, 32 + skip * 8, (3,3), (2,2), (1,1), bias=False)
        layers[0][1] = nn.BatchNorm2d(32 + skip * 8)

        layers[1][0].block[0][0] = nn.Conv2d(in_channels=32 + start + skip*8 - 7*ex_f, out_channels=32, kernel_size=(3,3), padding=(1,1), bias=False) #can add skip first input to here

        #fix option: add 1X1 conv to wanted number of channels at end of last layer
        layers[1][-1].block[-1][0] = nn.Conv2d(in_channels=32, out_channels= 16 + skip * 7, kernel_size=(1,1), bias=False)
        layers[1][-1].block[-1][1] = nn.BatchNorm2d(16 + skip * 7)

        layers[2][0].block[0][0] = nn.Conv2d(in_channels=16 + skip * 8 + start - 6 * ex_f, out_channels=96, kernel_size=(1,1), bias=False) #first layer has 1 skip and the rest can have 2
        
        #for in-block residual connection
        layers[2][0].block[-1][0] = nn.Conv2d(in_channels=96, out_channels=24 + skip * 6, kernel_size=(1,1), stride=(1,1), bias=False)
        layers[2][0].block[-1][1] = nn.BatchNorm2d(24 + skip * 6)
        layers[2][1].block[0][0] = nn.Conv2d(24 + skip * 6, 144, (1,1))

        layers[2][-1].block[-1][0] = nn.Conv2d(in_channels=144, out_channels=24 + skip * 6, kernel_size=(1,1), stride=(1,1), bias=False)
        layers[2][-1].block[-1][1] = nn.BatchNorm2d(24 + skip * 6)


        #layers[2][-1].block[-1][0] = nn.Conv2d(in_channels=144, out_channels= 24 + skip * 2, kernel_size=(1,1), bias=False)
        #layers[2][-1].block[-1][1] = nn.BatchNorm2d(24 + skip * 2)


        layers[3][0].block[0][0] = nn.Conv2d(in_channels=24 + skip * 8 + start - 5 * ex_f, out_channels=144, kernel_size=(1,1), bias=False)
        
        for i in range(len(layers[3])):
            c = layers[3][i].block[-1][0]
            cout = c.out_channels
           
            layers[3][i].block[-1][0] = nn.Conv2d(c.in_channels, cout + skip * 5, (1,1), (1,1), bias=False)
            layers[3][i].block[-1][1] = nn.BatchNorm2d(cout + skip * 5)
            if i > 0: 
                temp_c = layers[3][i].block[0][0]
                layers[3][i].block[0][0] = nn.Conv2d(cout + skip * 5, temp_c.out_channels, (1,1), (1,1), bias=False)

        layers[4][0].block[0][0] = nn.Conv2d(in_channels= 40 + skip * 8 + start - 4 * ex_f, out_channels=240, kernel_size=(1,1), bias=False)

        for i in range(len(layers[4])):
            c = layers[4][i].block[-1][0]
            cout = c.out_channels
           
            layers[4][i].block[-1][0] = nn.Conv2d(c.in_channels, cout + skip * 4, (1,1), (1,1), bias=False)
            layers[4][i].block[-1][1] = nn.BatchNorm2d(cout + skip * 4)
            if i > 0: 
                temp_c = layers[4][i].block[0][0]
                layers[4][i].block[0][0] = nn.Conv2d(cout + skip * 4, temp_c.out_channels, (1,1), (1,1), bias=False)

        layers[5][0].block[0][0] = nn.Conv2d(in_channels= 80 + skip * 8 + start - 3*ex_f, out_channels=480, kernel_size=(1,1),bias=False)

        for i in range(len(layers[5])):
            c = layers[5][i].block[-1][0]
            cout = c.out_channels
           
            layers[5][i].block[-1][0] = nn.Conv2d(c.in_channels, cout + skip * 3, (1,1), (1,1), bias=False)
            layers[5][i].block[-1][1] = nn.BatchNorm2d(cout + skip * 3)
            if i > 0: 
                temp_c = layers[5][i].block[0][0]
                layers[5][i].block[0][0] = nn.Conv2d(cout + skip * 3, temp_c.out_channels, (1,1), (1,1), bias=False)

        layers[6][0].block[0][0] = nn.Conv2d(in_channels= 112 + skip * 8 + start -2*ex_f, out_channels=672, kernel_size=(1,1), bias=False)

        for i in range(len(layers[6])):
            c = layers[6][i].block[-1][0]
            cout = c.out_channels
           
            layers[6][i].block[-1][0] = nn.Conv2d(c.in_channels, cout + skip * 2, (1,1), (1,1), bias=False)
            layers[6][i].block[-1][1] = nn.BatchNorm2d(cout + skip * 2)
            if i > 0: 
                temp_c = layers[6][i].block[0][0]
                layers[6][i].block[0][0] = nn.Conv2d(cout + skip * 2, temp_c.out_channels, (1,1), (1,1), bias=False)

        layers[7][0].block[0][0] = nn.Conv2d(in_channels= 192 + skip * 8 + start - ex_f, out_channels=1152, kernel_size=(1,1), bias=False)

        for i in range(len(layers[7])):
            c = layers[7][i].block[-1][0]
            cout = c.out_channels
           
            layers[7][i].block[-1][0] = nn.Conv2d(c.in_channels, cout + skip * 1, (1,1), (1,1), bias=False)
            layers[7][i].block[-1][1] = nn.BatchNorm2d(cout + skip * 1)
            if i > 0: 
                temp_c = layers[7][i].block[0][0]
                layers[7][i].block[0][0] = nn.Conv2d(cout + skip * 1, temp_c.out_channels, (1,1), (1,1), bias=False)

        layers[8][0] = nn.Conv2d(in_channels= 320 + skip * 8 + start, out_channels=1280, kernel_size=(1,1), bias=False)
        
        return layers

    def greater(self, val1, val2, list):
        return len([x for x in list if x >= val1 and x < val2]) if self.skip_start else len([x for x in list if x > val1 and x < val2])

##For case of skip larger than 10
class ExclusiveNetDense(nn.Module):
    #Current model working with 2,3 skips, check for 5 and 10
    def __init__(self, skip=50, exlusive_forward=False, skip_start=False, out_c=1000): 
        #TODO: if exlusive forward, remove all channels of indices greater than number of featuers and not greater than layer number
        super().__init__()
        self.skip = skip
        self.exclusive_forward = exlusive_forward
        self.skip_start = skip_start

        baseline = torchvision.models.efficientnet_b0()

        self.features = baseline.features
        self.features = self.fit_channels(self.features, skip)

        self.layer_pools = [0, 2, 3, 4, 6] #can remove 0 at start
        if not self.skip_start:
            self.layer_pools = self.layer_pools[1:]

        self.classifier = baseline.classifier
        start = 3 if self.skip_start else 0
        if skip == 2:
            self.classifier[1] = nn.Linear(in_features=1280 + (len(self.features) * (skip-1)) - 1 + start, out_features=out_c, bias=True) #need to check for skip=2, can change out_features
        elif skip == 1:
            self.classifier[1] = nn.Linear(in_features=1280 + len(self.features) * skip - 1 + start, out_features=out_c, bias=True) #out classes changed for imagenet, should be parameter
        elif skip == 3:
            self.classifier[1] = nn.Linear(in_features=1288 + start, out_features=out_c, bias=True)
        else:
            self.classifier[1] = nn.Linear(in_features=1288 + start, out_features=out_c, bias=True)

    def forward(self, x):
        start = [x] if self.skip_start else [] #we would want a third option allowing skip from start only to first 4 layers
        prev = []
        for i in range(len(self.features)):
            prev_pooled = [start[0]] if self.skip_start else []
            for j in range(len(prev[:-1])):
                d = i - j
                if d == 0:
                    prev_pooled.append(prev[:-1][j][:,(-d-1)*self.skip:,:,:])
                else:
                    prev_pooled.append(prev[:-1][j][:,(-d-1)*self.skip:-d*self.skip,:,:])
            for idx in range(len(prev_pooled)):
                val = self.greater(idx, i, self.layer_pools)
               
                p = F.adaptive_avg_pool2d
                size = int(prev_pooled[idx].shape[-1] / 2**val)
                prev_pooled[idx] = p(prev_pooled[idx], (size, size))
            

            if len(prev) == 0:
                inx = x
            else:
                #temp = prev[-1] if not self.exclusiveforward else prev[-1][:]
                inx = torch.cat(prev_pooled + [prev[-1]], dim=1) 

            prev.append(self.features[i](inx))

        n = len(self.features)
        
        prev_pooled = start + [z[:,-n-1:-n, :, :] for z in prev[:-1]]
        for idx in range(len(prev_pooled)):
                val = self.greater(idx, 9, self.layer_pools)
               
                p = F.adaptive_avg_pool2d
                
                size = int(prev_pooled[idx].shape[-1] / 2**val)
                prev_pooled[idx] = p(prev_pooled[idx], (size, size))

        prev_pooled = [z for z in prev_pooled if z.shape[1] > 0]
        x = torch.cat(prev_pooled + [prev[-1]], dim=1) 
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def fit_channels(self, layers, skip): #if skip > 30 then use +1 for skip multiplier
            start = 3 if self.skip_start else 0
            
            layers[0][0] = nn.Conv2d(3, 32 + skip * 9, (3,3), (2,2), (1,1), bias=False)
            layers[0][1] = nn.BatchNorm2d(32 + skip * 9)

            layers[1][0].block[0][0] = nn.Conv2d(in_channels=32 + start + skip*9, out_channels=32, kernel_size=(3,3), padding=(1,1), bias=False) #can add skip first input to here

            #fix option: add 1X1 conv to wanted number of channels at end of last layer
            layers[1][-1].block[-1][0] = nn.Conv2d(in_channels=32, out_channels= 16 + skip * 8, kernel_size=(1,1), bias=False)
            layers[1][-1].block[-1][1] = nn.BatchNorm2d(16 + skip * 8) #if exclusive forward we will reduce the skip*7 from the input of next layer

            layers[2][0].block[0][0] = nn.Conv2d(in_channels=16 + skip * 9 + start, out_channels=96, kernel_size=(1,1), bias=False) #first layer has 1 skip and the rest can have 2
            
            #for in-block residual connection
            layers[2][0].block[-1][0] = nn.Conv2d(in_channels=96, out_channels=24 + skip * 7, kernel_size=(1,1), stride=(1,1), bias=False)
            layers[2][0].block[-1][1] = nn.BatchNorm2d(24 + skip * 7)
            layers[2][1].block[0][0] = nn.Conv2d(24 + skip * 7, 144, (1,1))

            layers[2][-1].block[-1][0] = nn.Conv2d(in_channels=144, out_channels=24 + skip * 7, kernel_size=(1,1), stride=(1,1), bias=False)
            layers[2][-1].block[-1][1] = nn.BatchNorm2d(24 + skip * 7)


            #layers[2][-1].block[-1][0] = nn.Conv2d(in_channels=144, out_channels= 24 + skip * 2, kernel_size=(1,1), bias=False)
            #layers[2][-1].block[-1][1] = nn.BatchNorm2d(24 + skip * 2)


            layers[3][0].block[0][0] = nn.Conv2d(in_channels=24 + skip * 9 + start, out_channels=144, kernel_size=(1,1), bias=False)
            
            for i in range(len(layers[3])):
                c = layers[3][i].block[-1][0]
                cout = c.out_channels
            
                layers[3][i].block[-1][0] = nn.Conv2d(c.in_channels, cout + skip * 6, (1,1), (1,1), bias=False)
                layers[3][i].block[-1][1] = nn.BatchNorm2d(cout + skip * 6)
                if i > 0: 
                    temp_c = layers[3][i].block[0][0]
                    layers[3][i].block[0][0] = nn.Conv2d(cout + skip * 6, temp_c.out_channels, (1,1), (1,1), bias=False)

            layers[4][0].block[0][0] = nn.Conv2d(in_channels= 40 + skip * 9 + start, out_channels=240, kernel_size=(1,1), bias=False)

            for i in range(len(layers[4])):
                c = layers[4][i].block[-1][0]
                cout = c.out_channels
            
                layers[4][i].block[-1][0] = nn.Conv2d(c.in_channels, cout + skip * 5, (1,1), (1,1), bias=False)
                layers[4][i].block[-1][1] = nn.BatchNorm2d(cout + skip * 5)
                if i > 0: 
                    temp_c = layers[4][i].block[0][0]
                    layers[4][i].block[0][0] = nn.Conv2d(cout + skip * 5, temp_c.out_channels, (1,1), (1,1), bias=False)

            layers[5][0].block[0][0] = nn.Conv2d(in_channels= 80 + skip * 9 + start, out_channels=480, kernel_size=(1,1),bias=False)

            for i in range(len(layers[5])):
                c = layers[5][i].block[-1][0]
                cout = c.out_channels
            
                layers[5][i].block[-1][0] = nn.Conv2d(c.in_channels, cout + skip * 4, (1,1), (1,1), bias=False)
                layers[5][i].block[-1][1] = nn.BatchNorm2d(cout + skip * 4)
                if i > 0: 
                    temp_c = layers[5][i].block[0][0]
                    layers[5][i].block[0][0] = nn.Conv2d(cout + skip * 4, temp_c.out_channels, (1,1), (1,1), bias=False)

            layers[6][0].block[0][0] = nn.Conv2d(in_channels= 112 + skip * 9 + start , out_channels=672, kernel_size=(1,1), bias=False)

            for i in range(len(layers[6])):
                c = layers[6][i].block[-1][0]
                cout = c.out_channels
            
                layers[6][i].block[-1][0] = nn.Conv2d(c.in_channels, cout + skip * 3, (1,1), (1,1), bias=False)
                layers[6][i].block[-1][1] = nn.BatchNorm2d(cout + skip * 3)
                if i > 0: 
                    temp_c = layers[6][i].block[0][0]
                    layers[6][i].block[0][0] = nn.Conv2d(cout + skip * 3, temp_c.out_channels, (1,1), (1,1), bias=False)

            layers[7][0].block[0][0] = nn.Conv2d(in_channels= 192 + skip * 9 + start, out_channels=1152, kernel_size=(1,1), bias=False)

            for i in range(len(layers[7])):
                c = layers[7][i].block[-1][0]
                cout = c.out_channels
            
                layers[7][i].block[-1][0] = nn.Conv2d(c.in_channels, cout + skip * 2, (1,1), (1,1), bias=False)
                layers[7][i].block[-1][1] = nn.BatchNorm2d(cout + skip * 2)
                if i > 0: 
                    temp_c = layers[7][i].block[0][0]
                    layers[7][i].block[0][0] = nn.Conv2d(cout + skip * 2, temp_c.out_channels, (1,1), (1,1), bias=False)

            layers[8][0] = nn.Conv2d(in_channels= 320 + skip * 9 + start, out_channels=1280, kernel_size=(1,1), bias=False)
            
            return layers
            
    def greater(self, val1, val2, list):
        return len([x for x in list if x >= val1 and x < val2]) if self.skip_start else len([x for x in list if x > val1 and x < val2])


if __name__ == "__main__":
    model = ExclusiveNet(skip=10, skip_start=False, exclusive_forward=True)
    x = torch.randn(32, 3, 128, 128)
    res = model(x)
    print(res.shape)




