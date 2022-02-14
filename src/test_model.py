import torch


def test_model(model, tokenizer, dataset, device, batch_size: int = 20):
    correct_count = 0
    for batch_start, batch_end in zip(
        range(0, len(dataset), batch_size), range(batch_size, len(dataset), batch_size)
    ):
        data = dataset[batch_start:batch_end]
        x = data["text"]
        y_true = torch.tensor(data["label"]).to(device)
        input = tokenizer(x, padding=True, return_tensors="pt").to(device)
        logits = model(**input)[0]
        y_pred = torch.argmax(logits, dim=1)
        correct_count += torch.sum(y_pred == y_true)

    acc = correct_count / len(dataset)
    print(acc)
