import torch
import random
def generate_vocab(filename):
    with open(filename) as vocab_filename:
        all_text = vocab_filename.read()
        unique_chars = set(all_text)
        unique_chars = sorted(list(unique_chars))
        unique_chars = ''.join(unique_chars)
        return unique_chars, len(unique_chars) + 1, all_text #add one to chars for EOS

def generate_bars(all_text):
    bars = all_text.split('\n[')
    return bars

def generate_bar(): return '[' + random.choice(bars)

all_letters, n_letters, all_text  = generate_vocab('data/all_merged.txt')
bars = generate_bars(all_text)


def letter_to_index(letter):
    return all_letters.find(letter)

def letter_to_tensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letter_to_index(letter)] = 1
    return tensor

def line_to_tensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letter_to_index(letter)] = 1
    return tensor

def random_bar(text):
    # from the text, pull a random bar
    print('barrrr')

def generate_input_tensor(line):
    return line_to_tensor(line)

def generate_target_tensor(line):
    target_line = line[1:]
    target_tensor = line_to_tensor(line)
    #create EOF 'letter'
    eof_tensor = torch.zeros(1, n_letters)
    eof_tensor[0][len(all_letters)-1] = 1
    eof_tensor = eof_tensor.unsqueeze(0)
    target_tensor = torch.cat((target_tensor, eof_tensor), 0)
    return target_tensor

def random_training_example():
    line = generate_bar()
    target_tensor = generate_target_tensor(line)
    input_tensor = generate_input_tensor(line)
    # Make these long, for nn.NLLLoss
    input_tensor, target_tensor = input_tensor.long(), target_tensor.long()
    return input_tensor, target_tensor


if __name__ == "__main__":
    random_training_example()
