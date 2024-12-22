import matplotlib.pyplot as plt

import re

def parse_results(file_path):
    results = []
    with open(file_path, 'r') as file:
        epoch = None
        for line in file:
            epoch_match = re.match(r'Epoch (\d+)', line)
            if epoch_match:
                epoch = int(epoch_match.group(1))
            elif re.match(r'-+', line):
                continue
            else:
                time_match = re.match(r'Time:\s+(\d+)', line)
                if time_match:
                    time = int(time_match.group(1))
                    train_loss = float(re.search(r'Train loss:\s+([\d.]+)', line).group(1))
                    val_loss = float(re.search(r'Val loss:\s+([\d.]+)', line).group(1))
                    batch = re.search(r'Batch:\s+(\d+)\s+/\s+(\d+)', line)
                    batch_num = int(batch.group(1))
                    total_batches = int(batch.group(2))
                    results.append({
                        'epoch': epoch,
                        'time': time,
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'batch_num': batch_num,
                        'total_batches': total_batches
                    })
    return results

def plot(file_path):
    parsed_results = parse_results(file_path)

    # Extracting data for plotting
    time = [result['time'] for result in parsed_results]
    train_loss = [result['train_loss'] for result in parsed_results]
    val_loss = [result['val_loss'] for result in parsed_results]

    # Plotting the data
    plt.figure(figsize=(10, 5))
    plt.plot(time, train_loss, label='Train Loss', marker='o')
    plt.plot(time, val_loss, label='Validation Loss', marker='o')

    # Adding labels and title
    plt.xlabel('Time')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Time')
    plt.legend()

    # Display the plot
    plt.show()