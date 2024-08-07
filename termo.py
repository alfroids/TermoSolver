import argparse
import os
from itertools import product
from time import time

import numpy as np
from numpy.typing import NDArray
from unidecode import unidecode


# WORD LIST FUNCTIONS
def get_patterns() -> NDArray[np.str_]:
    return np.sort(
        np.array(["".join(p) for p in product("=+-", repeat=WORD_SIZE)], dtype=np.str_)
    )


def get_word_list() -> NDArray[np.str_]:
    words = np.loadtxt(WORD_LIST_PATH, dtype=np.str_)
    words = np.unique(np.strings.lower(words))
    words = words[np.strings.isalpha(words)]
    words = words[np.strings.str_len(words) == WORD_SIZE]

    return words


def compute_pattern_matrix() -> NDArray[np.int_]:
    if os.path.exists(PATTERN_MATRIX_PATH):
        mtx: NDArray[np.int_] = np.load(PATTERN_MATRIX_PATH)

        assert mtx.shape == (
            N_WORDS,
            N_WORDS,
        ), "The size of the given pattern matrix is different from the word list."

    else:
        mtx: NDArray[np.int_] = np.zeros((N_WORDS, N_WORDS), dtype=np.int_)

        try:
            from tqdm import tqdm

            for i in tqdm(range(N_WORDS)):
                word: str = WORD_LIST[i]
                patterns: list[str] = [compute_pattern(word, w) for w in WORD_LIST]
                mtx[i] = np.searchsorted(PATTERNS, patterns)

        except ImportError:
            for i in range(N_WORDS):
                word: str = WORD_LIST[i]
                patterns: list[str] = [compute_pattern(word, w) for w in WORD_LIST]
                mtx[i] = np.searchsorted(PATTERNS, patterns)

        if PATTERN_MATRIX_PATH:
            np.save(PATTERN_MATRIX_PATH, mtx)

    return mtx


# TODO: optimize
def compute_pattern(guess: str, correct: str) -> str:
    clist: list[str] = list(correct)
    plist: list[str] = [""] * WORD_SIZE

    for k in range(WORD_SIZE):
        if guess[k] not in correct:
            plist[k] = "-"
        elif guess[k] == correct[k]:
            plist[k] = "="
            clist[k] = ""

    for k in range(WORD_SIZE):
        if plist[k]:
            continue
        if guess[k] in clist:
            plist[k] = "+"
            clist[clist.index(guess[k])] = ""
        else:
            plist[k] = "-"

    pattern = "".join(plist)

    assert pattern in PATTERNS

    return pattern


# USER INTERFACE FUNCTIONS
def print_suggestions(
    suggestions: NDArray[np.int_],
    possible: NDArray[np.bool_],
    entropies: NDArray[np.float64] | None = None,
) -> None:
    print("My suggestions:")

    for i in range(suggestions.size):
        word: str = str(WORD_LIST[suggestions[i]])
        out: str = f"        [{i + 1}] {word}"
        if not possible[:, suggestions[i]].any():
            out += "*"
        else:
            out += " "
        if VERBOSE >= 1 and entropies is not None:
            out += f" ({entropies[i]:.3f} bits)"
        print(out)


def input_guess(suggestions: NDArray[np.int_]) -> int:
    while True:
        guess: str = input("Your guess: ")

        try:
            i: int = int(guess)

            if 1 <= i <= suggestions.size:
                return suggestions[i - 1]

        except ValueError:
            guess = unidecode(guess.lower())
            i: int = int(np.searchsorted(WORD_LIST, guess))

            if WORD_LIST[i] == guess:
                return i


def input_result(game: int) -> int:
    while True:
        pattern: str = input(f"G{game}) Result: ")

        i: int = int(np.searchsorted(PATTERNS, pattern))

        if PATTERNS[i] == pattern:
            return i


# ENTROPY CALCULATION FUNCTIONS
def get_suggestions(
    possible: NDArray[np.bool_],
    victories: NDArray[np.bool_],
    possible_only: bool = True,
) -> tuple[NDArray[np.int_], NDArray[np.float64] | None]:
    for i in range(N_GAMES):
        if possible[i].sum() == 1 and not victories[i]:
            return np.argwhere(possible[i]).flatten(), None

    word_list: NDArray[np.bool_] = (
        possible.any(axis=0) if possible_only else np.ones(N_WORDS, dtype=bool)
    )

    entropies: NDArray[np.float64] = np.zeros(N_WORDS)
    for i in range(N_GAMES):
        ent_game = compute_entropies(word_list, possible[i])
        entropies += ent_game

    idx: NDArray[np.int_] = np.argsort(entropies)[-1 : -N_SUGGESTIONS - 1 : -1]

    return idx, entropies[idx]


def compute_entropies(
    word_list: NDArray[np.bool_], possible: NDArray[np.bool_]
) -> NDArray[np.float64]:
    subset: NDArray[np.int_] = PATTERN_MATRIX[word_list, :][:, possible]
    n: int = possible.sum()

    p: NDArray[np.float64] = np.zeros((word_list.sum(), N_PATTERNS))
    patts: NDArray[np.int_] = np.arange(N_PATTERNS)
    for i in range(word_list.sum()):
        w = subset[i]
        w.sort()
        idx = np.searchsorted(patts, w)
        idx[idx == patts.size] = 0
        mask = patts[idx] == w
        out = np.bincount(idx[mask], minlength=N_PATTERNS)
        p[i] = out / n

    logs: NDArray[np.float64] = np.log2(p, where=p > 0)
    entropies: NDArray[np.float64] = np.zeros(N_WORDS)
    entropies[word_list] = -np.sum(p * logs, axis=1)

    return entropies


# MAIN
def main() -> None:
    possible_words: NDArray[np.bool_] = np.ones((N_GAMES, N_WORDS), dtype=bool)
    victories: NDArray[np.bool_] = np.zeros(N_GAMES, dtype=bool)

    for i in range(1, N_GUESSES + 1):
        t: float = time()
        print(f"Computing suggestions for guess #{i}...")
        if VERBOSE >= 1:
            print(f" * {possible_words.sum()} possible guess(es) left.")
        print()

        x: tuple[NDArray[np.int_], NDArray[np.float64] | None] = get_suggestions(
            possible_words, victories, possible_only=(i > N_GAMES + 1)
        )
        suggestions: NDArray[np.int_] = x[0]
        entropies: NDArray[np.float64] | None = x[1]
        print_suggestions(suggestions, possible_words, entropies)
        if VERBOSE >= 2:
            print(f" > Done in {time() - t:.3f} second(s).")
        print()

        guess: int = input_guess(suggestions)
        print()

        for j in range(N_GAMES):
            if victories[j]:
                continue

            result: int = input_result(j + 1)
            print()

            if PATTERNS[result] == "=" * WORD_SIZE:
                victories[j] = True

            if victories.sum() == N_GAMES:
                print('Nice, "we" dit it!')
                break

            result_filter: NDArray[np.bool_] = PATTERN_MATRIX[guess] == result
            possible_words[j] = np.logical_and(possible_words[j], result_filter)

            if possible_words[j].sum() == 0:
                print(f"No possible guesses left for game {j + 1}...")
                break

        else:
            continue

        break

    else:
        print("Maybe next time...")


if __name__ == "__main__":
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        prog="python termo.py",
        description="Entropy-based Wordle-like solver."
        + "\nWhen prompted, input your guess by either choosing one of the suggestions"
        + " or typing a word.\nAfter that, input the result for each game by typing"
        + ' "-" for misses (black), "+" for correct letter (yellow)'
        + ' and "=" for correct position (green).',
    )

    parser.add_argument(
        "-W",
        "--word-list",
        help="Path to the file containing a list of all possible words."
        + " Words will be filtered by size.",
        required=True,
        metavar="WORD_LIST_PATH",
    )
    parser.add_argument(
        "-P",
        "--pattern-matrix",
        help="Path where the pattern matirx will be stored to or loaded from."
        + " If not given, the pattern matrix will be recomputed each time.",
        metavar="PATTERN_MATRIX_PATH",
        default="",
    )

    group: argparse._MutuallyExclusiveGroup = parser.add_mutually_exclusive_group(
        required=True
    )
    group.add_argument(
        "-1",
        "--termo",
        help="Default one word game.",
        action="store_true",
    )
    group.add_argument(
        "-2",
        "--dueto",
        help="Default two words game.",
        action="store_true",
    )
    group.add_argument(
        "-4",
        "--quarteto",
        help="Default four words game.",
        action="store_true",
    )
    group.add_argument(
        "-g",
        "--game",
        help="Game settings."
        + " Three arguments (int) are required:"
        + " word size, max number of guesses and number of simultaneous games.",
        nargs=3,
        type=int,
        metavar="N",
    )

    parser.add_argument(
        "-s",
        "--suggestions",
        help="Number of suggestions given for each guess. (Default: 3)",
        nargs=1,
        type=int,
        metavar="N",
        default=3,
    )

    parser.add_argument(
        "-v",
        "--verbose",
        help="Output information level."
        + " Either 0, 1 (shows entropy values for each guesses)"
        + " or 2 (shows execution times).",
        action="count",
        default=0,
    )

    args: argparse.Namespace = parser.parse_args()

    # PARAMETERS
    WORD_LIST_PATH: str = args.word_list
    PATTERN_MATRIX_PATH: str = args.pattern_matrix

    if args.termo:
        GAME_NAME: str = "termo"
        WORD_SIZE: int = 5
        N_GUESSES: int = 6
        N_GAMES: int = 1

    elif args.dueto:
        GAME_NAME: str = "dueto"
        WORD_SIZE: int = 5
        N_GUESSES: int = 7
        N_GAMES: int = 2

    elif args.quarteto:
        GAME_NAME: str = "quarteto"
        WORD_SIZE: int = 5
        N_GUESSES: int = 9
        N_GAMES: int = 4

    elif args.game:
        assert len(args.game) == 3
        for n in args.game:
            assert isinstance(n, int) and n > 0

        GAME_NAME: str = "custom game"
        WORD_SIZE: int = args.game[0]
        N_GUESSES: int = args.game[1]
        N_GAMES: int = args.game[2]

    else:
        exit(1)

    N_SUGGESTIONS: int = args.suggestions
    VERBOSE: int = args.verbose

    s: int = 40
    k: int = int((s - (10 + len(GAME_NAME))) / 2)
    s = k * 2 + 10 + len(GAME_NAME)
    print("+" + "-" * k + f"*** {GAME_NAME.upper()} ***" + "-" * k + "+")
    print(f"             word size = {WORD_SIZE}")
    print(f"           max guesses = {N_GUESSES}")
    print(f"    simultaneous games = {N_GAMES}")
    print("+" + "-" * (s - 2) + "+")
    print()
    print()

    # CONSTANTS
    t: float = time()
    print("Loading dictionary...")
    PATTERNS: NDArray[np.str_] = get_patterns()
    N_PATTERNS: int = PATTERNS.size
    WORD_LIST: NDArray[np.str_] = get_word_list()
    N_WORDS: int = WORD_LIST.size
    if VERBOSE >= 2:
        print(f" > {N_WORDS} words loaded in {time() - t:.3f} second(s).")
    print()

    t: float = time()
    print("Computing pattern matrix...")
    PATTERN_MATRIX: NDArray[np.int_] = compute_pattern_matrix()
    if VERBOSE >= 2:
        print(f" > Done in {time() - t:.3f} second(s).")
    print()

    main()
