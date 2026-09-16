import torch

from kornia.feature import LightGlue


print("=" * 60)
print(" ALIKED LIGHTGLUE CONFIGURATION")
print("=" * 60)


matcher = LightGlue(
    features="aliked"
).to("cpu")

matcher.eval()


print("\nLightGlue configuration:")
print(matcher.conf)


print("\n" + "=" * 60)
print(" IMPORTANT PARAMETERS")
print("=" * 60)

print("features:", matcher.features)
print("input_dim:", matcher.conf.input_dim)
print("descriptor_dim:", matcher.conf.descriptor_dim)
print("n_layers:", matcher.conf.n_layers)
print("num_heads:", matcher.conf.num_heads)
print("depth_confidence:", matcher.conf.depth_confidence)
print("width_confidence:", matcher.conf.width_confidence)
print("filter_threshold:", matcher.conf.filter_threshold)
print("weights:", matcher.conf.weights)


print("\n" + "=" * 60)
print(" MODEL PARAMETERS")
print("=" * 60)

total_params = sum(
    p.numel()
    for p in matcher.parameters()
)

print(
    "Total parameters:",
    total_params
)


print("\n" + "=" * 60)
print(" DESCRIPTOR PROJECTION")
print("=" * 60)

print(
    "Input projection:",
    matcher.input_proj
)


print("\n" + "=" * 60)
print(" TEST COMPLETE")
print("=" * 60)