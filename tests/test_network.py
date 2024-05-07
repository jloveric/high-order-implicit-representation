from high_order_implicit_representation.networks import GenerativeNetwork
import torch

def test_generative_network():
    batch_size = 7
    embedding_size = 10
    input_size = 2
    
    net = GenerativeNetwork(
        embedding_size=embedding_size,
        input_size=input_size,
        output_size=5,
        mlp_width=10,
        mlp_layers=1,
        input_segments=10,
        mlp_segments=2,
        n=3,
    )
    
    embedding = torch.rand(batch_size, embedding_size)
    position = torch.rand(batch_size, input_size)

    result = net(embedding, position)

    print('result', result.shape)