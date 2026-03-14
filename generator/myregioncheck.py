import torch

pt = r"C:\Users\thodo\Desktop\DIPLOMATIKI ERGASIA\EFARMOGI\TS_TRAJGEN DATASET DONLOADED FROM USER\TS-TrajGen_Dataset\Porto_Taxi\save\Porto_Taxi\my_region_generator.pt"
sd = torch.load(pt, map_location="cpu")

print("has w1:", "w1" in sd)
print("has function_g:", any(k.startswith("function_g.") for k in sd.keys()))
print("has function_h:", any(k.startswith("function_h.") for k in sd.keys()))
print("sample function_h keys:", [k for k in sd.keys() if k.startswith("function_h.")][:5])
