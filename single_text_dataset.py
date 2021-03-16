def generate_dataset(text_in: str, features: int, targets: int):

    udata = text_in.decode("utf-8")
    text = udata.encode("ascii", "ignore")

    final = len(text)-(targets+features)
    feature_list = []
    target_list = []
    for i in range(final):
        nf = [ord(val) for val in text[i:(i+features)]]
        feature_list.append(nf)
        nt = [ord(val) for val in text[(i+features):(i+features+targets)]]
        target_list.append(nt)

    return feature_list, target_list


def dataset_from_file(filename: str, features: int, targets: int):
    with open(filename, "r") as f:
        return generate_dataset(text_in=f.read(), features=features, targets=targets)
