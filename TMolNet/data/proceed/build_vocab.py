import argparse
import pickle
from collections import Counter
from build_vocab_until import WordVocab




def main():
    parser = argparse.ArgumentParser(description='Build a vocabulary pickle')
    parser.add_argument('--corpus_path', '-c', type=str, default='./data/esol.txt', help='path to th ecorpus')
    parser.add_argument('--out_path', '-o', type=str, default='./data/esol_vocab.pkl', help='output file')
    parser.add_argument('--min_freq', '-m', type=int, default=1, help='minimum frequency for vocabulary')
    parser.add_argument('--vocab_size', '-v', type=int, default=None, help='max vocabulary size')
    parser.add_argument('--encoding', '-e', type=str, default='utf-8', help='encoding of corpus')
    args = parser.parse_args()

    with open(args.corpus_path, "r", encoding=args.encoding) as f:
        vocab = WordVocab(f, max_size=args.vocab_size, min_freq=args.min_freq)
    args.out_path = '{}_vocab.pkl'.format(args.corpus_path[:-4])
    print("VOCAB SIZE:", len(vocab))
    vocab.save_vocab(args.out_path)


if __name__ == '__main__':
    main()
    print("运行完成")
