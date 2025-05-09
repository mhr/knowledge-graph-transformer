import random
import numpy as np

def load_entity_list(entity_file):
    """Load entity list from entity mapping file."""
    entities = []
    with open(entity_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                entity, _ = parts
                entities.append(entity)
    return entities

def load_triples(data_file):
    """Load triples (head, relation, tail) from data file."""
    triples = []
    with open(data_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 3:
                head, relation, tail = parts
                triples.append((head, relation, tail))
    return triples

def create_tokenized_dataset(data_file, entity_file, verify_negatives=True, max_length=128, seed=42):
    # Establish deterministic behavior with rigorous entropy control
    random.seed(seed)
    np.random.seed(seed)
    
    # Load entities and triples
    entities = load_entity_list(entity_file)
    triples = load_triples(data_file)
    triples_set = set(triples) if verify_negatives else set()
    
    # Construct vocabulary (token → index mapping)
    vocabulary = {'[PAD]': 0, '[CLS]': 1, '[UNK]': 2, '?': 3, 'Is': 4, 'the': 5, 'of': 6}
    idx = len(vocabulary)
    
    # Add entities and relations to vocabulary
    for h, r, t in triples:
        for token in [h, r, t]:
            if token not in vocabulary:
                vocabulary[token] = idx
                idx += 1
    
    # Create inverse mapping (index → token)
    inverse_vocabulary = {idx: token for token, idx in vocabulary.items()}
    
    tokenized_examples = []
    labels = []
    
    # Generate positive and negative examples
    for head, relation, tail in triples:
        # Positive example
        pos_query = f"[CLS] Is {head} the {relation} of {tail} ?"
        pos_tokens = pos_query.split()
        pos_indices = [vocabulary.get(token, vocabulary['[UNK]']) for token in pos_tokens]
        
        # Apply sequence normalization (padding/truncation)
        if len(pos_indices) < max_length:
            pos_indices = pos_indices + [vocabulary['[PAD]']] * (max_length - len(pos_indices))
        else:
            pos_indices = pos_indices[:max_length]
            
        tokenized_examples.append(pos_indices)
        labels.append(1)
        
        # Generate negative example via contrastive sampling
        neg_triple = None
        max_attempts = 50
        attempts = 0
        
        while neg_triple is None and attempts < max_attempts:
            # Stochastic corruption strategy with uniform probability
            if random.random() < 0.5:
                false_head = random.choice(entities)
                while false_head == head:
                    false_head = random.choice(entities)
                candidate = (false_head, relation, tail)
            else:
                false_tail = random.choice(entities)
                while false_tail == tail:
                    false_tail = random.choice(entities)
                candidate = (head, relation, false_tail)
            
            if not verify_negatives or candidate not in triples_set:
                neg_triple = candidate
            
            attempts += 1
        
        if neg_triple is None:
            neg_triple = candidate
        
        # Process negative example
        neg_head, neg_relation, neg_tail = neg_triple
        neg_query = f"[CLS] Is {neg_head} the {neg_relation} of {neg_tail} ?"
        neg_tokens = neg_query.split()
        neg_indices = [vocabulary.get(token, vocabulary['[UNK]']) for token in neg_tokens]
        
        # Apply sequence normalization
        if len(neg_indices) < max_length:
            neg_indices = neg_indices + [vocabulary['[PAD]']] * (max_length - len(neg_indices))
        else:
            neg_indices = neg_indices[:max_length]
            
        tokenized_examples.append(neg_indices)
        labels.append(0)
    
    # Apply deterministic permutation
    combined = list(zip(tokenized_examples, labels))
    np.random.shuffle(combined)
    tokenized_examples, labels = zip(*combined)
    
    # Convert to NumPy arrays for tensor operations
    tokenized_examples = np.array(tokenized_examples, dtype=np.int32)
    labels = np.array(labels, dtype=np.int32)
    
    return tokenized_examples, labels, vocabulary, inverse_vocabulary

def main(max_length=128, seed=0, root_dir="KGDatasets/Kinship", split="train"):
    data_file = f"{root_dir}/{split}.txt"
    entity_file = f"{root_dir}/entity2id.txt"
    
    # Call the enhanced function that returns integer-encoded sequences and vocabulary
    tokenized_sequences, labels, token2idx, idx2token = create_tokenized_dataset(
        data_file, entity_file, max_length=max_length, seed=seed
    )
    
    # Print examples of integer-encoded sequences
    print("Sample of generated dataset:")
    for i in range(min(5, len(tokenized_sequences))):
        # For display purposes, convert back to tokens temporarily
        original_tokens = [idx2token.get(idx, '[UNK]') for idx in tokenized_sequences[i] if idx != token2idx['[PAD]']]
        tokens_str = ' '.join(original_tokens)
        # Show both integer sequence and original tokens
        integer_seq = ' '.join(map(str, tokenized_sequences[i][:10])) + "..." if len(tokenized_sequences[i]) > 10 else ' '.join(map(str, tokenized_sequences[i]))
        print(f"Example {i+1}: {integer_seq}")
        print(f"         (Original: {tokens_str} | Label: {labels[i]})")
    
    # Dataset statistics
    positive_count = sum(labels)
    negative_count = len(labels) - positive_count
    print(f"\nDataset statistics:")
    print(f"Total examples: {len(labels)}")
    print(f"Positive examples: {positive_count} ({positive_count/len(labels):.2%})")
    print(f"Negative examples: {negative_count} ({negative_count/len(labels):.2%})")
    print(f"Vocabulary size: {len(token2idx)}")
    
    return tokenized_sequences, labels, token2idx, idx2token