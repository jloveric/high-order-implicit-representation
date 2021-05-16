from single_text_dataset import encode_input_from_text, decode_output_to_text
import torch

def test_encode_decode():
    encoding, raw_text = encode_input_from_text(
        text_in="hello ripley!", features=10)
    assert len(encoding) == 10
    assert raw_text == "lo ripley!"

    input_vals = torch.rand(128)
    values, indices, ascii = decode_output_to_text(encoding=input_vals, topk=5)
    
    assert values.shape[0] == 5
    assert indices.shape[0] == 5
    assert len(ascii) == 5

    print('result', values, indices, ascii)
    print('encoding', encoding)
    print('raw_text', raw_text)
