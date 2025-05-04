import torch
from tqdm import tqdm
from utilities import save_checkpoint, save_sample_as_numpy
from vae_loss_function import vae_loss_function, compute_chunk_values


# Helper function to freeze parameters
def freeze_parameters(module):
    for param in module.parameters():
        param.requires_grad = False




# Training and validation with VAE
def train_and_validate(model,
                       train_loader, 
                       val_loader,
                       start_epoch,
                       epochs,
                       criterion, 
                       optimizer,
                       device,
                       sample_rate, 
                       checkpoint_folder,
                       music_out_folder, 
                       FREEZE_ENCODER_DECODER_AFTER=10):
    # Move the model to the correct device
    model = model.to(device)

    for epoch in range(start_epoch, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")

        # Training
        model.train()
        train_loss = 0
        for inputs, targets in tqdm(train_loader, desc="Training"):

            # Move inputs and targets to the correct device
            total_loss = torch.tensor(0.).to(device)
            inputs, targets = inputs.to(device), targets.to(device)

            # Preprocess inputs and targets
            inputs = inputs    +1
            targets = targets  +1

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            reconstructed, outputs = model(inputs)

            #reconstructed_c = compute_chunk_values(reconstructed, 200, reduction="max")
            #inputs_c = compute_chunk_values(inputs, 200, reduction="max")
            #outputs_c = compute_chunk_values(outputs, 200, reduction="max")
            #targets_c = compute_chunk_values(targets, 200, reduction="max")

           # factor = 30
           # task_loss_c = factor * criterion(outputs_c, targets_c)
           # rec_loss_c = factor * criterion(inputs_c, reconstructed_c)
            rec_loss = criterion(inputs, reconstructed)




            # Apply exponential transformation to reconstructed, outputs, and targets
           # reconstructed, outputs, targets = (
           #     torch.exp(2 * reconstructed),
           #     torch.exp(2 * outputs),
           #     torch.exp(2 * targets),
           # )

            # Compute VAE loss
            #total_loss, reconstruction_loss, kl_loss = vae_loss_function(
            #    reconstructed, inputs, mu, logvar, criterion
            #)

            # Add task-specific loss if applicable
            if epoch >= FREEZE_ENCODER_DECODER_AFTER:
                task_loss = criterion(outputs, targets)
                total_loss += task_loss #+ task_loss_c
            else:
                task_loss = 0.0
                


            total_loss += rec_loss # rec_loss_c + 
            # Backward pass
            total_loss.backward()

            # Optimizer step
            optimizer.step()

            # Accumulate the training loss
            train_loss += total_loss.item()

        # Calculate average training loss
        avg_train_loss = train_loss / len(train_loader)
        print(
            f"Training Loss: {avg_train_loss:.4f}, "
            f"Reconstruction Loss: {rec_loss:.4f}, "
            f"Task Loss: {task_loss:.4f}" #KL Loss: {kl_loss:.4f},
        )


        # Save checkpoint and validation sample every 10 epochs
        if epoch % 10 == 0:
            save_checkpoint(model, optimizer, epoch, checkpoint_folder)
            save_sample_as_numpy(model, val_loader, device, music_out_folder, epoch, prefix = 'tr_')

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc="Validation"):
                # Move inputs and targets to the correct device
                inputs, targets = inputs.to(device), targets.to(device)

                # Forward pass
                reconstructed, outputs = model(inputs)
                rec_loss = criterion(inputs, reconstructed)

                # Compute VAE loss
               # total_loss, reconstruction_loss, kl_loss = vae_loss_function(
               #     reconstructed, inputs, mu, logvar, criterion
               # )

                # Add task-specific loss if applicable
                task_loss = criterion(outputs, targets) if epoch >= FREEZE_ENCODER_DECODER_AFTER else 0.0
                total_loss += task_loss

                # Accumulate the validation loss
                val_loss += total_loss.item()

        # Calculate average validation loss
        avg_val_loss = val_loss / len(val_loader)
        print(
            f"Validation Loss: {avg_val_loss:.4f}, "
            f"Reconstruction Loss: {rec_loss:.4f}, "
            f" Task Loss: {task_loss:.4f}" #KL Loss: {kl_loss:.4f},
        )

        # Save checkpoint and validation sample every 10 epochs
        if epoch % 10 == 0:
            save_checkpoint(model, optimizer, epoch, checkpoint_folder)
            save_sample_as_numpy(model, val_loader, device, music_out_folder, epoch)
