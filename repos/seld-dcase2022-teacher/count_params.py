import torch
import seldnet_model
import parameters

def main():
    params = parameters.get_params('3')

    # Use the exact shapes printed by train_seldnet.py:
    # data_in: (batch, 7, feature_seq_len=250, mel_bins=64)
    # data_out: (batch, label_seq_len=50, 108)
    in_shape = (params['batch_size'], 7, params['feature_sequence_length'], params['nb_mel_bins'])
    out_shape = (params['batch_size'], params['label_sequence_length'], params['unique_classes'] * 3 * 3)

    model = seldnet_model.CRNN(in_shape, out_shape, params)

    state = torch.load(params['pretrained_model_weights'], map_location='cpu')
    model.load_state_dict(state)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Total parameters:", total)
    print("Trainable parameters:", trainable)

if __name__ == "__main__":
    main()