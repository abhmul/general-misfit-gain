from typing import Collection, Iterable, Iterator, TypeVar, Container

S = TypeVar("S")

Number = float | int
JsonType = None | Number | str | bool | list["JsonType"] | dict[str, "JsonType"]


def peek(it: Iterable[S]) -> S:
    it = iter(it)
    first = next(it)
    return first


class SizedGenerator(Iterator[S]):
    def __init__(self, it: Iterator[S], length: int):
        self.it = it
        self.length = int(length)

    def __len__(self):
        return self.length

    def __next__(self):
        return next(self.it)
