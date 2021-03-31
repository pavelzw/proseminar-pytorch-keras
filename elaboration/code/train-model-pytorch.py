loss_criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr_rate)

epochs = 3

for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    for images, labels in train_loader:
        optimizer.zero_grad()

        outputs = model(images)
        loss = loss_criterion(outputs, labels)
        loss.backward()
        optimizer.step()
