from vit import *


model_vit = ViT(
        image_size = 256,
        patch_size = 32,
        num_classes = 145,
        dim = 1024,
        depth = 6,
        heads = 16,
        mlp_dim = 2048,
        dropout = 0.1,
        emb_dropout = 0.1
    )

print(model_vit)

torch.save(model_vit, "pt/model_vit")