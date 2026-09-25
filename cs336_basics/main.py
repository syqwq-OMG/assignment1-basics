from cs336_basics.BPE.bpe import BPE
import timeit

if __name__ == '__main__':
    bpe = BPE(special_tokens=['<|endoftext|>'])
    
    start_time = timeit.default_timer()
    bpe.train('tests/fixtures/TinyStories-train.txt', vocab_size=10000)
    end_time = timeit.default_timer()

    print(f"Training time: {end_time - start_time:.2f} seconds")

    bpe.save("tinystories_train")