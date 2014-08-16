import h5py


def load_sim(fn):
    with h5py.File(fn, "r") as f:
        n = f.attrs["length"]
        tags = f["tags"][:n]
        features = f["features"][:n, :]

    return tags, features


if __name__ == "__main__":
    tags, features = load_sim("simulation.h5")
    print(tags.shape)
    print(features.shape)
