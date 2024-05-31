class Hyperparameters:
    def __init__(self):
        # Hyperparameters and paths
        self.img_size = 128
        self.patch_size = 8
        self.in_channels = 3
        self.emb_dim = 1024
        self.num_heads = 4
        self.num_layers = 18
        self.forward_expansion = 4
        self.num_classes = self.img_size * self.img_size * self.in_channels  # Assuming next frame prediction with same size
        self.batch_size = 4

