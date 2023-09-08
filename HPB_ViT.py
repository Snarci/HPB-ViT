import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.nn.functional as F
import math
from transformers import AutoModelForImageClassification,AutoFeatureExtractor,AutoModel
from transformers import pipeline
# helpers
from modules import *
import re

def get_largest_number(string):
    # use regular expressions to extract all the numbers from the string
    numbers = re.findall(r'\d+', string)
    
    # convert the numbers to integers and find the largest number
    if numbers:
        largest_number = max(int(number) for number in numbers)
        return largest_number
    else:
        return None

class patch_expander(nn.Module):
    def __init__(self, num_patches_in, num_patches_out, size_in, size_out):
        super().__init__()
        self.mpl_channels = nn.Linear(num_patches_in, num_patches_out)
        self.mpl_size = nn.Linear(size_in, size_out)


    def forward(self, x):
        x = x.transpose(1,2)
        x = self.mpl_channels(x)
        x = x.transpose(1,2)
        x = self.mpl_size(x)
        return x
    

class teacher_encoder(nn.Module):
    def __init__(self, model_name, channels_out, dim_out, freezed_teacher=True):
        super().__init__()
        self.model_name = model_name
        #i need to extract features from the teacher
        self.feature_extractor = AutoModel.from_pretrained(model_name,trust_remote_code=True, output_hidden_states=True)
        #get the last hidden state number of channels and size

        #check if the config has the image size attribute
        if hasattr(self.feature_extractor.config, 'image_size'):
            iamge_size = self.feature_extractor.config.image_size
        else:
            iamge_size = get_largest_number(model_name)

        img = torch.rand(2,3,iamge_size,iamge_size)
        x = self.feature_extractor(img)
        #if its not a tensor take the output_hidden_states
        if not isinstance(x, torch.Tensor):
            x = x.last_hidden_state
        #if it has only a shape of 2 then expand it to 3
        if len(x.shape) == 2:
            x = torch.unsqueeze(x, 1)
        #if the tensor has more than 3 dimensions then i need to merge the last dimensions
        if len(x.shape) > 3:
            x = x.flatten(2)
        channels_last_hidden_state = x.shape[-2]
        size_last_hidden_state = x.shape[-1]
        #now i need to expand the channels and the size
        self.patch_expander = patch_expander(channels_last_hidden_state, channels_out, size_last_hidden_state, dim_out)
    def forward(self, x):
        x = self.feature_extractor(x)
        if not isinstance(x, torch.Tensor):
            x = x.last_hidden_state
        if len(x.shape) == 2:
            x = torch.unsqueeze(x, 1)
        if len(x.shape) > 3:
            x = x.flatten(2)
        x = self.patch_expander(x)  
        
        return x
    

class HPB_ViT(nn.Module):
    def __init__(self,dim=768,init_channels=256,mlp_dim=2048,num_classes=4,teacher_name='',freezed_teacher=True,dropout=0.0,exp=False):
        super().__init__()
        debug=False
        self.num_stages = 4
        #self.depths = [4,4,8,4]
        #self.depths = [1,1,1,1]
        #self.depths = [2,2,2,2] #buono max 86.2
        self.depths = [2,2,4,2]
        #self.depths = [4,4,4,4]
        # an array of all number of patches at each stage the image size gets halved each stage
        self.num_patches = [(init_channels//(4**i)) for i in range(self.num_stages)]
        #print(self.num_patches)
        self.teacher = teacher_encoder(teacher_name, init_channels, dim, freezed_teacher)
        #path embedding for each stage the first is a PatchEmbed and the rest are HRec
        self.patch_embeddings = nn.ModuleList([
            HRecV2( 
                in_dim=dim,
                embed_dim=1024,
                inner_dim=mlp_dim) for i in range(self.num_stages)])
        #transformer for each stage
        self.transformers = nn.ModuleList([
            Transformer(
                dim=dim,
                depth=self.depths[i],
                heads=8,
                dim_head=64,
                mlp_dim=mlp_dim,
                dropout=dropout) for i in range(self.num_stages)])
        #now class and position embeddings for each stage
        self.cls_tokens = nn.ParameterList([nn.Parameter(torch.randn(1, 1, dim)) for i in range(self.num_stages)])
        self.pos_embedding = nn.ParameterList([nn.Parameter(torch.randn(1, self.num_patches[i] + 1, dim)) for i in range(self.num_stages)])
        #now the mlp head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim*(self.num_stages+2)),
            nn.Linear(dim*(self.num_stages+2), num_classes)
        )

        if debug:
            print(self.num_patches)
        #print number of parameters of the teacher
        print("Number of parameters of the teacher: ", sum(p.numel() for p in self.teacher.parameters() if p.requires_grad))
        
    def forward(self, x):
        #print(x.shape) 
        x = self.teacher(x)
        # collect mean of all the patches
        techer_class_tokens = x.mean(dim=1)

        #print(x.shape)
        class_tokens = []
        for i in range(self.num_stages):
            #print("stage ",i)
            #print(x.shape)
            x = self.patch_embeddings[i](x)
            #print(x.shape)
            b, n, _ = x.shape
            cls_tokens = repeat(self.cls_tokens[i], '() n d -> b n d', b=b)
            x = torch.cat((cls_tokens, x), dim=1)
            x += self.pos_embedding[i][:, :(n + 1)]
            x = self.transformers[i](x)
            #print(x.shape)
            class_tokens.append(x[:, 0])
            #remove the class tokens from x
            x = x[:, 1:]
        # now take the class tokens and concat them
        t = torch.cat(class_tokens, dim=1)
        #print (t.shape)
        #remove dimension 1 from x
        x = x.squeeze(1)
        #print (x.shape)
        x = torch.cat((t, x), dim=1)
        x = torch.cat((techer_class_tokens, x), dim=1)
        #print(x.shape)
        x = self.mlp_head(x)
        #print(x.shape)
        return x
        
        


