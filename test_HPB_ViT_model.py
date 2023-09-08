from HPB_ViT import HPB_ViT
import torch

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = 10
    img_dim = 224
    img = torch.rand(2,3,img_dim,img_dim).to(device)
    #if img is <0 or >1 normalize it
    img = torch.clamp(img, 0, 1)
    path_teacher="google/vit-base-patch16-224"
    dim = 384
    mpl_dim = dim*4
    num_classes = 4
    init_channels = 256
    model = HPB_ViT(
            dim=dim,
            init_channels=init_channels,
            mlp_dim=mpl_dim,
            num_classes=num_classes,
            teacher_name=path_teacher,
            freezed_teacher=False,
            exp=False
            ).to(device)
    x= model(img)
    print(x.shape)
    #print number of parameters
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))


    #simulate backprop
    loss = x.sum()
    loss.backward()
    print(loss)
    print("Backprop done")



   

