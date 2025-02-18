import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


def starting_train(train_dataset, val_dataset, model, hyperparameters, n_eval):
    """
    Trains and evaluates a model.

    Args:
        train_dataset:   PyTorch dataset containing training data.
        val_dataset:     PyTorch dataset containing validation data.
        model:           PyTorch model to be trained.
        hyperparameters: Dictionary containing hyperparameters.
        n_eval:          Interval at which we evaluate our model.
    """

    # Get keyword arguments
    batch_size, epochs = hyperparameters["batch_size"], hyperparameters["epochs"]

    # Initialize dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True )

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # Initalize optimizer (for gradient descent) and loss function
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()


    if torch.cuda.is_available(): # Check if GPU is available
        device = torch.device('cuda:0')
        print("Using GPU")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    step = 0
    print(device)
    model = model.to(device) # Move model to GPU if available
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")

        # Loop over each batch in the dataset
        for batch in tqdm(train_loader):
            # Backpropagation and gradient descent:

            #print('batch' + batch)
            images, labels = batch
            # print(type(images))
            # print(type(labels))

            # Move inputs over to GPU
            images = images.to(device)
            labels = labels.to(device)

            # Forward propagation
            outputs = model(images) # Same thing as model.forward(images)

            # What shape does outputs have?

            # Backprop
            loss = loss_fn(outputs, labels)
            loss.backward()       # Compute gradients
            optimizer.step()      # Update all the weights with the gradients you just calculated
            optimizer.zero_grad() # Clear gradients before next iteration      

            # Periodically evaluate our model + log to Tensorboard
            if step % n_eval == 0:
                # TODO:
                # Compute training loss and accuracy.
                # Log the results to Tensorboard.

                # TODO:
                # Compute validation loss and accuracy.
                # Log the results to Tensorboard.
                # Don't forget to turn off gradient calculations!
                pass
                # evaluate(val_loader, model, loss_fn)

            step += 1

        print('Epoch:', epoch, 'Loss:', loss.item())      
        print()


def compute_accuracy(outputs, labels):
    """
    Computes the accuracy of a model's predictions.

    Example input:
        outputs: [0.7, 0.9, 0.3, 0.2]
        labels:  [1, 1, 0, 1]

    Example output:
        0.75
    """

    n_correct = (torch.round(outputs) == labels).sum().item()
    n_total = len(outputs)
    return n_correct / n_total


def evaluate(val_loader, model, loss_fn):
    """
    Computes the loss and accuracy of a model on the validation dataset.

    TODO!
    """

    loss = 0
    total_labels = []
    total_outputs = []
    with torch.no_grad():
        for batch in tqdm(val_loader):
            images, labels = val_loader
            outputs = model(images)
            loss += loss_fn(outputs, labels).mean().item()
            total_labels += labels
            total_outputs += outputs
    
    accuracy = compute_accuracy(total_outputs, total_labels)
    
 
