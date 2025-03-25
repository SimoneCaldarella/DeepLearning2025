import time
import random
import torchvision.transforms as transforms

from tqdm import tqdm
from torch.utils.data import Subset
from torchvision.datasets import Flowers102

# I know guys, but they added only one month ago in pytorch
CLASSES = [
    "pink primrose",
    "hard-leaved pocket orchid",
    "canterbury bells",
    "sweet pea",
    "english marigold",
    "tiger lily",
    "moon orchid",
    "bird of paradise",
    "monkshood",
    "globe thistle",
    "snapdragon",
    "colt's foot",
    "king protea",
    "spear thistle",
    "yellow iris",
    "globe-flower",
    "purple coneflower",
    "peruvian lily",
    "balloon flower",
    "giant white arum lily",
    "fire lily",
    "pincushion flower",
    "fritillary",
    "red ginger",
    "grape hyacinth",
    "corn poppy",
    "prince of wales feathers",
    "stemless gentian",
    "artichoke",
    "sweet william",
    "carnation",
    "garden phlox",
    "love in the mist",
    "mexican aster",
    "alpine sea holly",
    "ruby-lipped cattleya",
    "cape flower",
    "great masterwort",
    "siam tulip",
    "lenten rose",
    "barbeton daisy",
    "daffodil",
    "sword lily",
    "poinsettia",
    "bolero deep blue",
    "wallflower",
    "marigold",
    "buttercup",
    "oxeye daisy",
    "common dandelion",
    "petunia",
    "wild pansy",
    "primula",
    "sunflower",
    "pelargonium",
    "bishop of llandaff",
    "gaura",
    "geranium",
    "orange dahlia",
    "pink-yellow dahlia?",
    "cautleya spicata",
    "japanese anemone",
    "black-eyed susan",
    "silverbush",
    "californian poppy",
    "osteospermum",
    "spring crocus",
    "bearded iris",
    "windflower",
    "tree poppy",
    "gazania",
    "azalea",
    "water lily",
    "rose",
    "thorn apple",
    "morning glory",
    "passion flower",
    "lotus",
    "toad lily",
    "anthurium",
    "frangipani",
    "clematis",
    "hibiscus",
    "columbine",
    "desert-rose",
    "tree mallow",
    "magnolia",
    "cyclamen",
    "watercress",
    "canna lily",
    "hippeastrum",
    "bee balm",
    "ball moss",
    "foxglove",
    "bougainvillea",
    "camellia",
    "mallow",
    "mexican petunia",
    "bromelia",
    "blanket flower",
    "trumpet creeper",
    "blackberry lily",
]
        

def load_splits(data_dir: str = "./data", verbose: bool = True) -> Subset:
    """Load the base and novel splits of the Flowers102 dataset.
    Args:
        data_dir (str): Directory where the dataset will be stored.
        verbose (bool): If True, print status messages.
    Returns:
        tuple: A tuple containing the training, validation, and base + novel test sets.
    """
    start_time = time.time()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    train = Flowers102(root=data_dir, split="train", download=True, transform=transform)
    valid = Flowers102(root=data_dir, split="val", download=True, transform=transform)
    test = Flowers102(root=data_dir, split="test", download=True, transform=transform)

    train_split = _split_data(train, ["base"])
    if verbose: print("Train base set loaded!", flush=True)
    valid_split = _split_data(valid, ["base"])
    if verbose: print("Valid base set loaded!", flush=True)
    test_split = _split_data(test, ["base", "novel"])
    if verbose: print("Test base and novel set loaded!", flush=True)

    end_time = time.time()
    if verbose: print(f"Data loaded in {end_time - start_time:.2f} seconds.", flush=True)

    return train_split["base"], valid_split["base"], test_split["base"], test_split["novel"]


def _split_data(dataset, splits=["base", "novel"], K=10):

    dataset.class_to_idx = {CLASSES[idx]: idx for idx in dataset._labels}

    # Sort classes by name, standard in Few-Shot Learning
    sorted_classes = sorted(dataset.class_to_idx.items(), key=lambda x: x[0])

    # Get ordered labels_idx
    sorted_labels_idx = [dataset.class_to_idx[class_name] for class_name, _ in sorted_classes]

    # Ensure samples are ordered by class (required for base2novel split)  
    samples_idx = list(range(len(dataset)))
    sorted_samples_idx = sorted(samples_idx, key=lambda x: sorted_labels_idx.index(dataset._labels[x]))

    # Split the dataset into two halves
    num_classes = len(sorted_labels_idx)
    first_half_classes = set(sorted_labels_idx[:num_classes // 2])
    second_half_classes = set(sorted_labels_idx[num_classes // 2:])

    splits_dict = {}

    if "base" in splits:
        # Collect samples belonging to the first half classes
        first_half_indices = [sample_idx for sample_idx in sorted_samples_idx 
                              if dataset._labels[sample_idx] in first_half_classes]
        
        first_class2samples = {idx: [] for idx in first_half_classes}

        for sample_idx in first_half_indices:
            class_idx = dataset._labels[sample_idx]
            first_class2samples[class_idx].append(sample_idx)

        # Select K random samples per class
        # oxford only has 10 samples per class, 
        # however this may be useful for other datasets
        selected_indices_base = []
        for class_idx in first_half_classes:
            class_samples = first_class2samples[class_idx]
            selected_samples = random.sample(class_samples, K)
            selected_indices_base.extend(selected_samples)

        base_set = Subset(dataset, selected_indices_base)
        base_set.class2idx = {CLASSES[idx]: idx for idx in first_half_classes}

        splits_dict.update({"base": base_set})

    if "novel" in splits:
        # Collect samples belonging to the second half classes
        second_half_indices = [sample_idx for sample_idx in sorted_samples_idx 
                               if dataset._labels[sample_idx] in second_half_classes]
        
        second_class2samples = {idx: [] for idx in second_half_classes}

        for sample_idx in second_half_indices:
            class_idx = dataset._labels[sample_idx]
            second_class2samples[class_idx].append(sample_idx)

        selected_indices_novel = []
        for class_idx in second_half_classes:
            class_samples = second_class2samples[class_idx]
            selected_samples = random.sample(class_samples, K)
            selected_indices_novel.extend(selected_samples)

        novel_set = Subset(dataset, selected_indices_novel)
        novel_set.class2idx = {CLASSES[idx]: idx for idx in second_half_classes}
        
        splits_dict.update({"novel": novel_set})

    # Returning base and novel classes dataset
    return splits_dict
