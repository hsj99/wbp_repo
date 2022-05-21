import os
import time
import torch
import argparse
import warnings
import torch.nn.functional as F
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import pandas as pd
# utils
from KIE.utils.logger import create_logger
from KIE.utils.checkpointer import Checkpointer
# configs
from KIE.configs import configs
# data
from KIE.data.generate import CustomDataset, TrainBatchCollator
from KIE.data.preprocessing import create_dataloader
# model
from KIE.model import CharacterLevelCNNHighwayBiLSTM
# misc
from KIE.utils.misc import AverageMeter, get_process_time
# plotting
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")

# arg
parser = argparse.ArgumentParser(description="Character-Level for Text Classification")

parser.add_argument("--config-file", action="store", help="The path to the configs file.")
parser.add_argument("--save-steps", default=0, type=int, metavar="N",
                    help="After a certain number of epochs, a checkpoint is saved")
parser.add_argument("--log-steps", default=10, type=int, help="Print logs every log steps")
parser.add_argument("--resume", action="store",
                    help="Resume training from a path to the given checkpoint.")
parser.add_argument("--use-cuda", action="store_true",
                    help="Enable or disable the CUDA training. By default it is disable.")
parser.add_argument("--gpu-device", default=0, type=int, help="Specify the GPU ID to use. By default, it is the ID 0")


args = parser.parse_args()

# Put warnings to silence if any.
warnings.filterwarnings("ignore")

def main():
    
    # CUDA
    use_cuda = torch.cuda.is_available() and args.use_cuda
    torch.autograd.set_detect_anomaly(True)

    # output directory
    output_dir = os.path.normpath(configs.OUTPUT_DIR)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # logger
    logger = create_logger(name="Training phase", output_dir=output_dir, log_filename="training-log")

    # checkpointer
    checkpointer = Checkpointer(output_dir, logger)

    # dataloader arguments
    dataloader_args = dict(configs.DATALOADER.TRAINING)

    # adding collate_fn to dataloader args
    dataloader_args["collate_fn"] = TrainBatchCollator(class_labels_padding_value=configs.MODEL.LOSS.IGNORE_INDEX)

    # The declaration and tensor type of the CPU/GPU device.
    device = torch.device("cpu")  # By default, the device is CPU.
    if not use_cuda:
        torch.set_default_tensor_type("torch.FloatTensor")
    else:
        device = torch.device("cuda")
        torch.cuda.set_device(args.gpu_device)
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
            
        # When one does not care about reproducibility,
        # this two lines below will find the best algorithm for your GPU.
        torch.backends.cudnn.enabled = True
        
    dataloader_args["generator"] = torch.Generator(device=device)

    # generate train data, perform split and define model loss function
    logger.info("Splitting the data into Train/Validation...")
    class_dict = dict(configs.DATASET.CLASS_NAMES)
    train_dataset = CustomDataset(new_dir=dict(configs.DATASET.TRAIN), class_dict=class_dict,
                                    validation_args={"val_folder": dict(configs.DATASET.VAL), "ratio": 0.1})
    train_loader = create_dataloader(dataset=train_dataset, is_train=True, **dataloader_args)
    train_criterion = torch.nn.CrossEntropyLoss(ignore_index=configs.MODEL.LOSS.IGNORE_INDEX, reduction="none")
    vocabulary = train_dataset.vocabulary
    text_max_length = train_dataset.text_max_length

    # generate validation data, perform split and define model loss function
    logger.info("Declaring and initialising the validation dataset...")
    validation_dataset = CustomDataset(new_dir=dict(configs.DATASET.VAL), class_dict=class_dict)
    validation_loader = create_dataloader(dataset=validation_dataset, is_train=False, **dataloader_args)
    validation_criterion = torch.nn.CrossEntropyLoss(ignore_index=configs.MODEL.LOSS.IGNORE_INDEX, reduction="mean")

    # define model and save model to device
    logger.info("Declaring and initialising the model and optimizer...")
    model_params = dict(configs.MODEL.PARAMS)
    vocab_size = len(vocabulary)
    model = CharacterLevelCNNHighwayBiLSTM(n_classes=configs.DATASET.NUM_CLASSES,
                                            max_seq_length=text_max_length,
                                            char_vocab_size=vocab_size, **model_params)
    model = model.to(device)

    # optimiser
    optimizer = torch.optim.AdamW(model.parameters(),
                                lr=configs.SOLVER.LR,
                                weight_decay=configs.SOLVER.WEIGHT_DECAY,
                                amsgrad=configs.SOLVER.AMSGRAD)

    # learning rate
    logger.info("Declaring and initialising the learning rate scheduler...")
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                        milestones=configs.SOLVER.LR_DECAY_STEPS,
                                                        gamma=configs.SOLVER.GAMMA)

    # resume training from last checkpoint if resume not None
    start_epoch = 1
    checkpoint_data = {}
    best_validation_f1_score = total_loss_plotter_window = None

    if args.resume is not None:
            
        logger.info("Resuming the training...")
            
        checkpoint_data = checkpointer.load(file_to_load=args.resume, map_location=device)
            
        start_epoch = checkpoint_data["epoch"]
            
        total_loss_plotter_window = checkpoint_data.get("total_loss_plot_win")
        best_validation_f1_score = checkpoint_data.get("best_validation_f1_score")
            
        optimizer_state_dict = checkpoint_data.get("optimizer_state_dict")
        if optimizer_state_dict is not None:
            optimizer.load_state_dict(optimizer_state_dict)
            
        model_state_dict = checkpoint_data.get("model_state_dict")
        if model_state_dict is not None:
            model.load_state_dict(model_state_dict)

    logger.info("Starting Training...")
    
    if best_validation_f1_score is None:
        best_validation_f1_score = 0.0

    accumulated_epoch = 0
    
    training_start_time = time.time()

    ###############################
    epoch_acc_scores = []
    epoch_loss_scores = []
    ###############################

    # saving training information to logs and checkpointing model files

    for current_epoch in range(start_epoch, configs.SOLVER.MAX_EPOCHS + 1):
        
        training_loss = train(model=model, optimizer=optimizer, train_loader=train_loader,
                              criterion=train_criterion, device=device, current_epoch=current_epoch,
                              logger=logger)
        
        logger.info("[TRAINING] Total loss: {total_loss:.6f}\n\n".format(total_loss=training_loss))
        
        logger.info("Validation in progress...")
        if use_cuda:
            torch.cuda.empty_cache()  # speed up evaluation after k-epoch training has just finished.
        validation_f1_score, validation_loss, acc_scores, loss_scores, true_classes, predicted_classes = validate(model=model, validation_loader=validation_loader,
                                                        criterion=validation_criterion, device=device,
                                                        current_epoch=current_epoch, logger=logger)
        #Saving models based on the highest F1-score.
        if validation_f1_score > best_validation_f1_score:
            best_validation_f1_score = validation_f1_score
            checkpoint_data.update({"best_validation_f1_score": best_validation_f1_score})
            checkpointer.save(name="BEST_MODEL_BASED_ON_F1_SCORE", is_best=True, data=checkpoint_data)
        
        # Saving the lr scheduler.
        lr_scheduler.step(current_epoch)
        checkpoint_data.update({"lr_scheduler_state_dict": lr_scheduler.state_dict()})
        
        # Saving useful data.
        checkpoint_data.update({
            "epoch": current_epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
        })
        
        if current_epoch > 0 and args.save_steps > 0 and current_epoch % args.save_steps == 0:
            
            logger.info("Saving a checkpoint at epoch {0}...".format(current_epoch))
            
            checkpointer.save(name="{0}_{1}_CHECKPOINT_EPOCH_{2}".format(
                configs.MODEL.NAME.upper(),
                configs.DATASET.NAME.upper(),
                current_epoch),
                data=checkpoint_data
            )    
        
        # Computing time...
        accumulated_epoch += 1
        elapsed_time, left_time, estimated_finish_time = get_process_time(start_time=training_start_time,
                                                                          current_iteration=accumulated_epoch,
                                                                          max_iterations=configs.SOLVER.MAX_EPOCHS)
        logger.info("Elapsed time: {et} seconds || "
                    "Remaining time: {lt} seconds || "
                    "ETA: {eta}\n\n".format(
            et=elapsed_time,
            lt=left_time,
            eta=estimated_finish_time
        ))

        ####################################
        epoch_acc_scores.append(acc_scores)
        epoch_loss_scores.append(loss_scores)
        ####################################

    logger.info("Training has ended. Saving the final checkpoint...")
    checkpointer.save(name="{0}_FINAL_CHECKPOINT".format(
        configs.MODEL.NAME.upper()),
        data=checkpoint_data
    )
    #########################
    print("Epoch Acc Scores", epoch_acc_scores)
    print("Epoch Loss Scores", epoch_loss_scores)
    #########################
    
    epochs_range = range(1, configs.SOLVER.MAX_EPOCHS + 1)
    plt.figure(figsize=(15,5))
    plt.subplot(1,2,1)
    plt.plot(epochs_range, epoch_acc_scores)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy')

    plt.subplot(1,2,2)
    plt.plot(epochs_range, epoch_loss_scores)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Model Loss')
    plt.savefig('KIE\\outputs\\acc_loss_graph.png')
    plt.show()

    #################################
    # save classification report
    cl_report = classification_report(true_classes, predicted_classes, output_dict=True)
    df = pd.DataFrame(cl_report).transpose()
    df.to_csv('KIE\\outputs\\classification_report.csv')

    #################################
    # save heatmap    
    heatmap = sns.heatmap(confusion_matrix(true_classes, predicted_classes), cmap='coolwarm', cbar=True, annot=True, fmt='1', center=True)
    figure = heatmap.get_figure()    
    figure.savefig('KIE\\outputs\\heatmap.png', dpi=400)
    #################################

    if use_cuda:
        torch.cuda.empty_cache()



# train the model

def train(model: torch.nn.Module, optimizer, train_loader,
          criterion: torch.nn.Module, device: torch.device,
          current_epoch: int, logger):
    training_losses = AverageMeter(fmt=":.6f")
    
    # Activating Dropout layer if any...
    model = model.train()
    
    for current_iteration, batch_samples in enumerate(train_loader):
        
        text_features, text_class_labels, class_weights = batch_samples
        
        text_features = text_features.to(device)
        text_class_labels = text_class_labels.to(device)
        
        # Clearing out the model's gradients before doing backprop.
        # 'set_to_none=True' here can modestly improve performance
        optimizer.zero_grad(set_to_none=True)
        
        predictions = model(text_features)
        
        class_weights = class_weights.contiguous().view(-1)
        
        text_class_labels = text_class_labels.contiguous().view(-1)
        
        predictions = predictions.contiguous().view(-1, configs.DATASET.NUM_CLASSES)
        
        loss = (class_weights * criterion(predictions, text_class_labels)).mean()
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=5.0, norm_type=2.0)
        
        optimizer.step()

        training_losses.update(loss.detach().clone(), n=text_features.size(0))
        
        if current_iteration > 0 and args.log_steps > 0 and current_iteration % args.log_steps == 0:
            logger.info("Epoch: {curr_epoch}/{maximum_epochs} || "
                        "Iterations: {iter} || "
                        "Learning_rate: {lr} || "
                        "Loss: {loss}\n\n".format(
                curr_epoch=current_epoch,
                maximum_epochs=configs.SOLVER.MAX_EPOCHS,
                iter=current_iteration,
                lr=optimizer.param_groups[0]["lr"],
                loss=training_losses
            ))
    return training_losses.global_avg


# validate the model

@torch.no_grad()
def validate(model: torch.nn.Module, validation_loader,
             criterion: torch.nn.Module, device: torch.device,
             current_epoch: int, logger):
    validation_losses = AverageMeter(fmt=":.6f")
    
    # Disabling Dropout layer if any...
    model = model.eval()
    
    true_classes = []
    predicted_classes = []
    ####################################
    acc_scores = []
    loss_scores = []
    ####################################
    
    for current_iteration, batch_samples in enumerate(validation_loader):
        
        text_features, text_class_labels, class_weights = batch_samples

        ############################
        iter_acc_scores = []
        iter_loss_scores = []
        ############################
        
        # Putting into the right device...
        text_features = text_features.to(device)
        text_class_labels = text_class_labels.to(device)
        
        text_class_labels = text_class_labels.view(-1)
        
        outputs = model(text_features)
        
        loss = criterion(outputs.view(-1, configs.DATASET.NUM_CLASSES), text_class_labels)
        ######################
        iter_loss_scores.append(loss.tolist())
        ##########################
        validation_losses.update(loss.detach().cpu().numpy(), text_features.size(0))
        idx = torch.where(text_class_labels == configs.MODEL.LOSS.IGNORE_INDEX)[0]
        text_class_labels[idx] = 0
        true_classes.extend(text_class_labels.detach().cpu().numpy())
        
        _, predictions = torch.max(F.softmax(outputs, dim=2), dim=2)
        predictions = predictions.contiguous().view(-1)
        predicted_classes.extend(predictions.detach().cpu().numpy())
        
        if current_iteration > 0 and args.log_steps > 0 and current_iteration % args.log_steps == 0:
            logger.info("Epoch: {curr_epoch}/{maximum_epochs} || "
                        "Iteration: {iter} || "
                        "Loss: {loss}\n\n".format(
                curr_epoch=current_epoch,
                maximum_epochs=configs.SOLVER.MAX_EPOCHS,
                iter=current_iteration,
                loss=validation_losses
            ))
        ####################################
        acc = accuracy_score(true_classes, predicted_classes)
        iter_acc_scores.append(acc)
        ######################################

    acc_scores.append(sum(iter_acc_scores)/len(iter_acc_scores))
    loss_scores.append(sum(iter_loss_scores)/len(iter_loss_scores))

    #####################################


    validation_loss = validation_losses.global_avg
    validation_f1_score = f1_score(true_classes, predicted_classes, average="weighted")
    results = {
        "loss": validation_loss,
        "precision": precision_score(true_classes, predicted_classes, average="weighted"),
        "recall": recall_score(true_classes, predicted_classes, average="weighted"),
        "f1": validation_f1_score
    }
    
    report = classification_report(true_classes, predicted_classes)
    logger.info("\n" + report)
    
    logger.info("***** Validation results ******")
    for key in sorted(results.keys()):
        logger.info("  %s = %s", key, str(results[key]))

    return validation_f1_score, validation_loss, acc_scores, loss_scores, true_classes, predicted_classes


# execute the file

if __name__ == '__main__':
    
    # Guarding against bad arguments.
    
    if args.save_steps < 0:
        raise ValueError("{0} is an invalid value for the argument: --save-steps".format(args.save_steps))
    elif args.log_steps < 0:
        raise ValueError("{0} is an invalid value for the argument: --log-steps".format(args.log_steps))
    elif args.resume is not None and not os.path.isfile(args.resume):
        raise ValueError("The path to the checkpoint data file is wrong!")
    
    gpu_devices = list(range(torch.cuda.device_count()))
    if len(gpu_devices) != 0 and args.gpu_device not in gpu_devices:
        raise ValueError("Your GPU ID is out of the range! You may want to check with 'nvidia-smi'")
    elif args.use_cuda and not torch.cuda.is_available():
        raise ValueError("The argument --use-cuda is specified but it seems you cannot use CUDA!")
    
    if args.config_file is not None:
        if not os.path.isfile(args.config_file):
            raise ValueError("The configs file is wrong!")
        else:
            configs.merge_from_file(args.config_file)
    configs.freeze()
    
    main()