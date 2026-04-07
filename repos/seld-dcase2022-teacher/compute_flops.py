import torch
from thop import profile
import seldnet_model
import parameters

params = parameters.get_params('3')

batch_size = 1
data_in = (batch_size, 7, params['feature_sequence_length'], params['nb_mel_bins'])
data_out = (batch_size, params['label_sequence_length'], params['unique_classes'] * 3 * 3)

model = seldnet_model.CRNN(data_in, data_out, params)
model.eval()

x = torch.randn(*data_in)

macs, params_count = profile(model, inputs=(x,), verbose=False)

print("Input shape:", data_in)
print("MACs:", macs)
print("Approx FLOPs:", 2 * macs)
print("Params from thop:", params_count)