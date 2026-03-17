import torch

ckpt = torch.load('checkpoints/best_model.pth', map_location='cpu')
print("=" * 50)
print("BEST MODEL CHECKPOINT INFO")
print("=" * 50)
print(f"Epoch: {ckpt['epoch']}")
print(f"Best Accuracy: {ckpt['best_acc']:.2f}%")
print(f"Optimizer State: {'present' if 'optimizer_state_dict' in ckpt else 'missing'}")
print(f"Model Parameters: {len(ckpt['model_state_dict'])} layers")
print("=" * 50)
print("\n✅ These are the weights exported to fpga_weights/")
