import torch
import open_clip

from tqdm import tqdm
from flowers_utils import load_splits
from torch.utils.data import DataLoader



TEMPLATE = lambda C: f"a photo of a {C}, a type of flower."


def eval(model, dataset, batch_size=512, device="cuda"):

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    idx2class = {idx: class_name for class_name, idx in dataset.class2idx.items()}

    # Text inputs
    class_names = list(dataset.class2idx)
    text_inputs = open_clip.tokenize([TEMPLATE(name) for name in class_names]).to(device)

    # Evaluate the dataset
    correct = 0
    with torch.no_grad():
        for batch in tqdm(dataloader):
            images, labels_idx = batch
            labels = torch.Tensor([class_names.index(idx2class[idx.item()]) for idx in labels_idx]).long()
    
            image_inputs = images.to(device)

            # Get embeddings
            image_features = model.encode_image(image_inputs)
            text_features = model.encode_text(text_inputs)

            # Normalize
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            # Compute similarity
            logits_per_image = (image_features @ text_features.T).softmax(dim=-1)
            predicted_class = logits_per_image.argmax(dim=-1).to("cpu")

            # Save accuracy rate in batch
            correct += (predicted_class == labels).sum().item()

    accuracy = correct / len(dataset)
    return accuracy


def main():
    # Zero-shot evaluation with OpenCLIP
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
    model = model.to(device)

    # Load the dataset
    _, _, test_base_set, test_novel_set = load_splits(transform=preprocess)

    # Evaluate on base classes
    print(f"\n🧠 Zero-shot evaluation on Base Classes with {len(test_base_set)} samples.")
    accuracy = eval(model, test_base_set, device=device)
    print(f"🔍 Accuracy {accuracy*100:.2f}% on the base classes.")

    # Evaluate on novel classes
    print(f"\n🧠 Zero-shot evaluation on Novel Classes with {len(test_novel_set)} samples.")
    accuracy = eval(model, test_novel_set, device=device)
    print(f"🔍 Accuracy {accuracy*100:.2f}% on the novel classes.")


if __name__ == "__main__":
    main()
