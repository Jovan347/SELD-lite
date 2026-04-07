import seldnet_model
import parameters

def main():
    params = parameters.get_params('3')

    in_shape = (params['batch_size'], 7, params['feature_sequence_length'], params['nb_mel_bins'])
    out_shape = (params['batch_size'], params['label_sequence_length'], params['unique_classes'] * 3 * 3)

    model = seldnet_model.CRNN(in_shape, out_shape, params)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Total parameters (architecture):", total)
    print("Trainable parameters:", trainable)

if __name__ == "__main__":
    main()