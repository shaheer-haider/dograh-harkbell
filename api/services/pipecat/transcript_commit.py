"""Harkbell: the turn analyzer tells a manually committed transcriber when to commit.

A transcriber on manual commits (``elevenlabs_manual_stt.py``) used to commit
at every stop of the engine's own VAD, 0.2 s into any pause. A pause in the
middle of a sentence then cut the sentence in two, and the commit could land
on the first sound of the next word: on the Mirror Labs copy "It's in two
days. No, sorry, it's in two hours." reached the agent as "... Sorry, i / In
two hours." and it answered "you broke up there" (run 6583). The turn
analyzer already judges, at that same stop, whether the caller has finished
-- so it is the one that says when the audio is committed.

The stop strategy (``number_hold_turn_stop.py``) sends one of these upstream
at each VAD stop: a commit when the turn is over, a hold when it is not (the
analyzer heard more coming, or the turn ends on part of an order number). A
held commit is sent when the turn ends. The transcriber commits on its own a
moment after a stop that gets no answer, and some seconds into a hold that is
never released, so it never depends on these arriving.
"""

from dataclasses import dataclass

from pipecat.frames.frames import SystemFrame


@dataclass
class TranscriptCommitRequestFrame(SystemFrame):
    """The caller's turn is over: commit what the transcriber has heard."""


@dataclass
class TranscriptCommitHoldFrame(SystemFrame):
    """The caller has paused inside their turn: do not commit yet."""
